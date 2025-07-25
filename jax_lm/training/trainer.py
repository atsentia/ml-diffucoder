"""Base trainer class for JAX DiffuCoder models."""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import asdict

import jax
import jax.numpy as jnp
from jax import random
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


class TrainState(train_state.TrainState):
    """Extended train state with additional fields."""
    
    # Add fields for training statistics
    loss_scale: Optional[float] = 1.0
    global_step: int = 0
    epoch: int = 0
    best_eval_metric: float = float("-inf")
    
    # Reference model parameters for GRPO
    ref_params: Optional[Dict[str, Any]] = None


class Trainer:
    """Base trainer for DiffuCoder models."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[DiffuCoder] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup devices and mesh
        self.devices = setup_tpu()
        self.mesh = get_tpu_mesh(self.devices)
        
        # Initialize random keys
        self.rng = random.PRNGKey(config.seed)
        
        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model if not provided
        if self.model is None:
            self._initialize_model()
        
        # Setup optimizer and train state
        self._setup_optimizer()
        self._setup_train_state()
        
        # Setup checkpointing
        self.checkpointer = ocp.PyTreeCheckpointer()
    
    def _initialize_model(self):
        """Initialize model from config or checkpoint."""
        model_path = Path(self.config.model_name_or_path)
        
        if model_path.exists():
            # Load from local checkpoint
            self.model, params = load_model(
                str(model_path),
                dtype=self._get_dtype(),
            )
            self.initial_params = params
        else:
            # Initialize new model
            # In production, this would download from HuggingFace
            raise NotImplementedError(
                "Model initialization from HuggingFace not yet implemented. "
                "Please provide a local checkpoint path."
            )
    
    def _get_dtype(self):
        """Get JAX dtype from config."""
        dtype_map = {
            "float32": jnp.float32,
            "float16": jnp.float16,
            "bfloat16": jnp.bfloat16,
        }
        return dtype_map.get(self.config.dtype, jnp.float32)
    
    def _setup_optimizer(self):
        """Setup Optax optimizer with learning rate schedule."""
        # Learning rate schedule
        if self.config.lr_scheduler_type == "cosine_with_min_lr":
            min_lr = self.config.learning_rate * self.config.lr_scheduler_kwargs.get("min_lr_rate", 0.1)
            schedule = optax.cosine_decay_schedule(
                init_value=self.config.learning_rate,
                decay_steps=self.config.num_train_epochs * len(self.train_dataset) // self.config.train_batch_size,
                alpha=min_lr / self.config.learning_rate,
            )
        elif self.config.lr_scheduler_type == "constant":
            schedule = self.config.learning_rate
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler_type}")
        
        # Optimizer chain
        optimizer_chain = [
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adamw(
                learning_rate=schedule,
                b1=self.config.adam_beta1,
                b2=self.config.adam_beta2,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            ),
        ]
        
        self.optimizer = optax.chain(*optimizer_chain)
    
    def _setup_train_state(self):
        """Initialize training state."""
        # Create train state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.initial_params,
            tx=self.optimizer,
            global_step=0,
            epoch=0,
        )
        
        # Shard parameters across devices
        with self.mesh:
            self.state = self.state.replace(
                params=shard_params(self.state.params, self.mesh)
            )
    
    def train_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: random.PRNGKey,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Single training step.
        
        This should be overridden by subclasses for specific training logic.
        """
        raise NotImplementedError("Subclasses must implement train_step")
    
    def eval_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: random.PRNGKey,
    ) -> Dict[str, float]:
        """Single evaluation step."""
        # Simple forward pass and loss computation
        outputs = state.apply_fn(
            state.params,
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            deterministic=True,
        )
        
        # Compute loss (simplified - actual implementation would compute proper loss)
        logits = outputs["logits"]
        labels = batch.get("labels", batch["input_ids"])
        
        # Cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits.reshape(-1, logits.shape[-1]),
            labels=labels.reshape(-1),
        ).mean()
        
        return {"loss": loss}
    
    def save_checkpoint(self, state: TrainState, metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and state
        save_model(self.model, state.params, checkpoint_dir)
        
        # Save training state
        train_state_path = checkpoint_dir / "train_state"
        self.checkpointer.save(
            train_state_path,
            {
                "state": state,
                "config": asdict(self.config),
                "metrics": metrics or {},
            }
        )
        
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.num_train_epochs} epochs")
        print(f"Total training steps: {self.config.num_train_epochs * len(self.train_dataset) // self.config.train_batch_size}")
        
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
                desc=f"Training epoch {epoch + 1}",
                total=len(self.train_dataset),
            )
            
            for batch_idx, batch in enumerate(pbar):
                # Split RNG
                self.rng, step_rng = random.split(self.rng)
                
                # Convert batch to JAX arrays
                batch = {k: jnp.array(v) for k, v in batch.items()}
                
                # Training step
                with self.mesh:
                    self.state, metrics = train_step_fn(self.state, batch, step_rng)
                
                train_metrics.append(metrics)
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_metrics = {
                        k: jnp.mean(jnp.array([m[k] for m in train_metrics[-self.config.logging_steps:]]))
                        for k in train_metrics[0].keys()
                    }
                    print(f"\nStep {global_step}: {avg_metrics}")
                
                # Evaluation
                if self.eval_dataset and global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    print(f"\nEvaluation at step {global_step}: {eval_metrics}")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(
                        self.state,
                        {"train": avg_metrics, "eval": eval_metrics if self.eval_dataset else {}},
                    )
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            
            # Update state
            self.state = self.state.replace(epoch=epoch + 1)
        
        # Final save
        self.save_checkpoint(self.state)
        print("\nTraining completed!")
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on eval dataset."""
        if not self.eval_dataset:
            return {}
        
        eval_step_fn = jax.jit(self.eval_step)
        eval_metrics = []
        
        for batch in tqdm(self.eval_dataset, desc="Evaluating"):
            # Split RNG
            self.rng, step_rng = random.split(self.rng)
            
            # Convert batch to JAX arrays
            batch = {k: jnp.array(v) for k, v in batch.items()}
            
            # Evaluation step
            with self.mesh:
                metrics = eval_step_fn(self.state, batch, step_rng)
            
            eval_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            k: jnp.mean(jnp.array([m[k] for m in eval_metrics]))
            for k in eval_metrics[0].keys()
        }
        
        return avg_metrics