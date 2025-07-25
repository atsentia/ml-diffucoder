"""JAX implementation of Coupled Group Relative Policy Optimization (GRPO) trainer."""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, List, Union
from dataclasses import asdict
import warnings
import random

import jax
import jax.numpy as jnp
from jax import random as jax_random
import optax
from flax.training import train_state
from flax.core import freeze, unfreeze
import orbax.checkpoint as ocp
from tqdm import tqdm

from jax_lm.models.diffucoder import DiffuCoder, DiffuCoderConfig
from jax_lm.utils import (
    load_model,
    save_model,
    setup_tpu,
    get_tpu_mesh,
    shard_params,
)
from jax_lm.training.config import TrainingConfig
from jax_lm.training.trainer import Trainer, TrainState


def split_batch_dict(
    batch_dict: Dict[str, jnp.ndarray], num_chunks: int
) -> List[Dict[str, jnp.ndarray]]:
    """
    Split a dictionary of arrays along the first dimension into `num_chunks` equal parts.
    """
    first_key = next(iter(batch_dict.keys()))
    first_array = batch_dict[first_key]
    chunk_size = first_array.shape[0] // num_chunks
    
    return [
        {
            key: val[i * chunk_size : (i + 1) * chunk_size]
            for key, val in batch_dict.items()
        }
        for i in range(num_chunks)
    ]


def selective_log_softmax(logits, index, weights=None, mask=None):
    """
    JAX implementation of selective log softmax with weighted probabilities.
    
    Args:
        logits: Shape [num_iterations * 3 * batch_size, seq_len, vocab_size]
        index: Shape [num_iterations * batch_size, seq_len] 
        weights: Shape [num_iterations * 3] - weights for different versions
        mask: Shape [num_iterations * batch_size, seq_len] - mask indicating which tokens are masked
    
    Returns:
        Log probabilities with shape [num_iterations * batch_size, seq_len]
    """
    # Process sequences in chunks to reduce memory usage
    full_batch_size = logits.shape[0] // 3  # Each sequence has 3 versions
    num_iterations = weights.shape[0] // 3
    batch_size = full_batch_size // num_iterations
    per_token_logps = []
    
    # Process each sequence's three versions together
    for i in range(full_batch_size):
        # Get the three versions for this sequence
        seq_labels = index[i]  # [seq_len]
        
        chunk, offset = divmod(i, batch_size)
        base = chunk * 3 * batch_size
        indices = jnp.array([base + 0*batch_size + offset, base + 1*batch_size + offset, base + 2*batch_size + offset])
        
        seq_logits = logits[indices]  # [3, seq_len, vocab_size]
        
        # Compute log probabilities for all three versions
        seq_logps = jax.nn.log_softmax(seq_logits, axis=-1)
        seq_per_token_logps = jnp.take_along_axis(
            seq_logps, 
            seq_labels[None, :, None].repeat(3, axis=0), 
            axis=-1
        ).squeeze(-1)  # [3, seq_len]
        
        if weights is not None and mask is not None:
            # Get weights and mask for this sequence
            weight_idx = i // batch_size
            seq_weights = weights[weight_idx*3:(weight_idx+1)*3]  # [3]
            seq_mask = mask[i]  # [seq_len]
            
            # Weight the masked and unmasked versions
            weighted_logps = jnp.where(
                seq_mask,
                seq_per_token_logps[1] * seq_weights[1],  # p1 * t1
                seq_per_token_logps[2] * seq_weights[2]   # p2 * t2
            )
            
            # Combine with original probability and average
            final_logps = (seq_per_token_logps[0] + weighted_logps) / 2
        else:
            final_logps = seq_per_token_logps[0]  # Just use original probability
        
        per_token_logps.append(final_logps)
    
    return jnp.stack(per_token_logps)


class DiffuGRPOTrainer(Trainer):
    """
    JAX implementation of Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.
    
    This class extends the base Trainer to implement GRPO training with masked diffusion models,
    using efficient policy gradient estimation through conditional probabilities.
    
    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[DiffuCoder] = None,
        reward_funcs: Optional[List[Callable]] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        ref_model: Optional[DiffuCoder] = None,
    ):
        super().__init__(config, model, train_dataset, eval_dataset)
        
        # GRPO specific parameters
        self.reward_funcs = reward_funcs or []
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        
        # Training hyperparameters from config
        self.num_iterations = getattr(config, 'num_iterations', 4)
        self.num_generations = getattr(config, 'num_generations', 4)
        self.epsilon_low = getattr(config, 'epsilon_low', 0.2)
        self.epsilon_high = getattr(config, 'epsilon_high', 0.2)
        self.beta = getattr(config, 'beta', 0.01)  # KL penalty coefficient
        self.scale_rewards = getattr(config, 'scale_rewards', True)
        self.random_masking = getattr(config, 'random_masking', True)
        
        # Generation parameters
        self.max_completion_length = getattr(config, 'max_completion_length', 512)
        self.diffusion_steps = getattr(config, 'diffusion_steps', 256)
        self.generation_temperature = getattr(config, 'generation_temperature', 1.0)
        self.generation_batch_size = getattr(config, 'generation_batch_size', 4)
        
        # Internal state
        self._step = 0
        self._buffered_inputs = None
        self._metrics = {"train": {}, "eval": {}}
        self._textual_logs = {"prompt": [], "completion": [], "rewards": {}}
        
        # Initialize reference model if needed
        if self.beta > 0.0 and self.ref_model is None:
            self._setup_reference_model()
    
    def _setup_reference_model(self):
        """Setup reference model for KL penalty computation."""
        # Clone the current model parameters as reference
        self.ref_params = jax.tree_map(lambda x: x.copy(), self.state.params)
        print("Reference model initialized from current model parameters")
    
    def forward_process(self, batch, prompt_index, mask_id, key, accumulate=False):
        """
        Apply forward diffusion process to create masked versions of sequences.
        
        Args:
            batch: Input sequences [batch_size, seq_len]
            prompt_index: Boolean mask indicating prompt tokens [seq_len]
            mask_id: Token ID to use for masking
            key: JAX random key
            accumulate: Whether to handle gradient accumulation
            
        Returns:
            Tuple of (noisy_batches, weights, completion_mask)
        """
        b, l = batch.shape
        noisy_batch = []
        
        # Generate random mask ratio between 0.2 and 0.8
        mask_ratio = jax_random.uniform(key, (), minval=0.2, maxval=0.8)
        
        # Create random matrix to decide masking
        random_matrix = jax_random.uniform(key, (b, l))
        
        # 1. Always mask completion tokens (all completion tokens masked)
        is_mask = ~prompt_index[None, :]  # Broadcast to [batch_size, seq_len]
        noisy_batch.append(jnp.where(is_mask, mask_id, batch))
        
        # 2. Mask completion tokens with probability mask_ratio
        is_mask = ~prompt_index[None, :] & (random_matrix < mask_ratio)
        completion_mask = is_mask
        noisy_batch.append(jnp.where(is_mask, mask_id, batch))
        
        # 3. Mask completion tokens reversely (1 - mask_ratio probability)
        is_mask = ~prompt_index[None, :] & (random_matrix > mask_ratio)
        noisy_batch.append(jnp.where(is_mask, mask_id, batch))
        
        # Weights for the three versions
        weights = jnp.array([1.0, 1.0/mask_ratio, 1.0/(1.0-mask_ratio)])
        
        return noisy_batch, weights, completion_mask
    
    def get_logits(self, model_params, batch):
        """Get model logits for a batch of sequences."""
        outputs = self.model.apply(
            model_params,
            batch,
            deterministic=False,  # Use training mode for GRPO
        )
        logits = outputs["logits"]
        # Return logits shifted for next-token prediction
        return logits[:, :-1]
    
    def _get_per_token_logps(
        self, 
        model_params, 
        input_ids, 
        logits_to_keep, 
        mask_seeds
    ):
        """
        Calculate per-token log probabilities for diffusion model.
        
        Args:
            model_params: Model parameters
            input_ids: Input sequences [num_iterations, batch_size, seq_len]
            logits_to_keep: Number of completion tokens to compute logits for
            mask_seeds: List of random seeds for masking
            
        Returns:
            Per-token log probabilities [batch_size, num_iterations, logits_to_keep]
        """
        num_iterations, batch_size, seq_len = input_ids.shape
        
        # Ensure logits_to_keep is valid
        logits_to_keep = min(logits_to_keep, seq_len)
        per_token_logps = jnp.zeros((num_iterations, batch_size, logits_to_keep))
        
        prompt_length = seq_len - logits_to_keep
        prompt_index = jnp.zeros(seq_len, dtype=bool)
        prompt_index = prompt_index.at[:prompt_length].set(True)
        
        # Process each iteration
        all_perturbed_seqs = []
        all_weights = []
        all_expanded_inputs = []
        all_completion_masks = []
        
        for iter_idx, mask_seed in enumerate(mask_seeds):
            key = jax_random.PRNGKey(mask_seed)
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            
            perturbed_seq, t_weights, completion_mask = self.forward_process(
                expanded_input, 
                prompt_index, 
                self.tokenizer.mask_token_id,
                key,
                accumulate=(num_iterations > 1)
            )
            
            all_perturbed_seqs.extend(perturbed_seq)
            all_weights.extend(t_weights.tolist())
            all_expanded_inputs.append(expanded_input)
            all_completion_masks.append(completion_mask)
        
        # Concatenate all iterations into single batch
        perturbed_seq = jnp.concatenate(all_perturbed_seqs, axis=0)  # [num_iterations * 3 * batch_size, seq_len]
        completion_mask_seq = jnp.concatenate(all_completion_masks, axis=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = jnp.concatenate(all_expanded_inputs, axis=0)  # [num_iterations * batch_size, seq_len]
        all_weights_t = jnp.array(all_weights)  # [num_iterations * 3]
        
        # Get model predictions for the combined batch
        logits = self.get_logits(model_params, perturbed_seq)  # [num_iterations * 3 * batch_size, seq_len, vocab_size]
        
        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[:, -logits_to_keep:, :]  # [num_iterations * 3 * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[:, -logits_to_keep:]  # [num_iterations * batch_size, logits_to_keep]
        completion_loss_mask = completion_mask_seq[:, -logits_to_keep:]  # [num_iterations * batch_size, logits_to_keep]
        
        # Compute log probabilities using selective_log_softmax
        per_token_logps = selective_log_softmax(
            completion_logits,
            completion_targets,
            all_weights_t,
            completion_loss_mask
        ).reshape(num_iterations, batch_size, logits_to_keep).transpose(1, 0, 2)
        
        return per_token_logps.astype(jnp.float32)
    
    def compute_loss(
        self, 
        state: TrainState, 
        batch: Dict[str, jnp.ndarray], 
        rng: jax_random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """
        Compute GRPO loss for a batch.
        
        Args:
            state: Training state
            batch: Input batch
            rng: Random key
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"] 
        completion_ids = batch["completion_ids"]
        completion_mask = batch["completion_mask"]
        mask_seeds = batch["mask_seeds"]
        
        # Combine prompt and completion
        input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
        logits_to_keep = completion_ids.shape[1]
        
        # Get current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids_expanded = input_ids[None, :]  # Add iteration dimension
        
        per_token_logps = self._get_per_token_logps(
            state.params, input_ids_expanded, logits_to_keep, [this_itr_mask_seed]
        )
        
        # Compute KL divergence if beta > 0
        per_token_kl = jnp.zeros_like(per_token_logps)
        if self.beta > 0.0:
            ref_per_token_logps = batch["ref_per_token_logps"][:, this_itr_idx:this_itr_idx+1]
            per_token_kl = (
                jnp.exp(ref_per_token_logps - per_token_logps) - 
                (ref_per_token_logps - per_token_logps) - 1
            )
        
        # Compute GRPO loss
        advantages = batch["advantages"]
        old_per_token_logps = (
            batch["old_per_token_logps"][:, this_itr_idx:this_itr_idx+1]
            if self.num_iterations > 1
            else jax.lax.stop_gradient(per_token_logps)
        )
        
        coef_1 = jnp.exp(per_token_logps - old_per_token_logps)
        coef_2 = jnp.clip(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        
        per_token_loss1 = coef_1 * advantages[:, None, None]
        per_token_loss2 = coef_2 * advantages[:, None, None]
        per_token_loss = -jnp.minimum(per_token_loss1, per_token_loss2)
        
        if self.beta > 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # Compute final loss
        loss = (per_token_loss[:, 0, :] * completion_mask).sum() / jnp.maximum(completion_mask.sum(), 1.0)
        
        # Compute metrics
        metrics = {}
        if self.beta > 0.0:
            mean_kl = (per_token_kl * completion_mask[:, None, :]).sum() / jnp.maximum(completion_mask.sum(), 1.0)
            metrics["kl"] = mean_kl
        
        # Clipping statistics
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages[:, None, None] < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages[:, None, None] > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        
        low_clip = (is_low_clipped * completion_mask[:, None, :]).sum() / jnp.maximum(completion_mask.sum(), 1.0)
        high_clip = (is_high_clipped * completion_mask[:, None, :]).sum() / jnp.maximum(completion_mask.sum(), 1.0)
        clip_ratio = (is_region_clipped * completion_mask[:, None, :]).sum() / jnp.maximum(completion_mask.sum(), 1.0)
        
        metrics.update({
            "loss": loss,
            "clip_ratio_low": low_clip,
            "clip_ratio_high": high_clip,
            "clip_ratio_region": clip_ratio,
        })
        
        return loss, metrics
    
    def train_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: jax_random.PRNGKey,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Single GRPO training step."""
        
        def loss_fn(params):
            state_with_params = state.replace(params=params)
            loss, metrics = self.compute_loss(state_with_params, batch, rng)
            return loss, metrics
        
        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(global_step=state.global_step + 1)
        
        return new_state, metrics
    
    def _generate_completions(self, prompts_batch):
        """Generate completions for a batch of prompts using diffusion generation."""
        # This is a simplified version - in practice would use the actual diffusion generation
        # For now, return dummy completions
        batch_size = len(prompts_batch)
        max_length = self.max_completion_length
        
        # Dummy completion generation (replace with actual diffusion generation)
        completion_ids = jnp.ones((batch_size, max_length), dtype=jnp.int32) * self.tokenizer.pad_token_id
        completion_mask = jnp.ones((batch_size, max_length), dtype=jnp.int32)
        
        return completion_ids, completion_mask
    
    def _compute_rewards(self, prompts, completions):
        """Compute rewards for prompt-completion pairs."""
        rewards = []
        for reward_func in self.reward_funcs:
            batch_rewards = reward_func(prompts, completions)
            rewards.append(jnp.array(batch_rewards))
        
        if not rewards:
            # Return dummy rewards if no reward functions
            return jnp.zeros(len(prompts))
        
        # Average rewards across functions
        return jnp.stack(rewards).mean(axis=0)
    
    def _prepare_grpo_batch(self, raw_batch):
        """Prepare batch for GRPO training by generating completions and computing rewards."""
        prompts = raw_batch["prompts"]
        
        # Generate completions
        completion_ids, completion_mask = self._generate_completions(prompts)
        
        # Encode prompts
        # This is simplified - would use actual tokenizer
        prompt_ids = jnp.ones((len(prompts), 128), dtype=jnp.int32)  # Dummy
        prompt_mask = jnp.ones((len(prompts), 128), dtype=jnp.int32)  # Dummy
        
        # Compute rewards
        completions_text = ["dummy completion"] * len(prompts)  # Would decode completion_ids
        rewards = self._compute_rewards(prompts, completions_text)
        
        # Compute advantages (simplified group-wise advantages)
        rewards_grouped = rewards.reshape(-1, self.num_generations)
        sum_group = rewards_grouped.sum(axis=1, keepdims=True)
        baseline = (sum_group - rewards_grouped) / (self.num_generations - 1)
        advantages = (rewards_grouped - baseline).flatten()
        
        # Generate mask seeds
        if self.random_masking:
            mask_seeds = [random.randint(0, 2**12) for _ in range(self.num_iterations)]
        else:
            mask_seeds = [42] * self.num_iterations
        
        # Compute old and reference log probabilities (simplified)
        old_per_token_logps = jnp.zeros((len(prompts), self.num_iterations, completion_ids.shape[1]))
        ref_per_token_logps = jnp.zeros((len(prompts), self.num_iterations, completion_ids.shape[1]))
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,
        }
    
    def train(self):
        """Main GRPO training loop."""
        print(f"Starting GRPO training for {self.config.num_train_epochs} epochs")
        print(f"GRPO parameters: iterations={self.num_iterations}, generations={self.num_generations}")
        
        # Compile training step
        train_step_fn = jax.jit(self.train_step)
        
        # Training loop
        global_step = 0
        for epoch in range(self.config.num_train_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")
            
            # Training
            train_metrics = []
            epoch_start_time = time.time()
            
            # Create progress bar
            pbar = tqdm(
                self.train_dataset,
                desc=f"GRPO training epoch {epoch + 1}",
                total=len(self.train_dataset),
            )
            
            for batch_idx, raw_batch in enumerate(pbar):
                # Prepare GRPO batch (generate completions, compute rewards, etc.)
                batch = self._prepare_grpo_batch(raw_batch)
                
                # Split RNG
                self.rng, step_rng = jax_random.split(self.rng)
                
                # Training step
                with self.mesh:
                    self.state, metrics = train_step_fn(self.state, batch, step_rng)
                
                train_metrics.append(metrics)
                global_step += 1
                self._step += 1
                
                # Update progress bar
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_metrics = {
                        k: jnp.mean(jnp.array([m[k] for m in train_metrics[-self.config.logging_steps:]]))
                        for k in train_metrics[0].keys()
                    }
                    print(f"\nStep {global_step}: {avg_metrics}")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(self.state, {"train": avg_metrics})
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            
            # Update state
            self.state = self.state.replace(epoch=epoch + 1)
        
        # Final save
        self.save_checkpoint(self.state)
        print("\nGRPO training completed!")