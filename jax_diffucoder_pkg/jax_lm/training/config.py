"""Training configuration for JAX DiffuCoder."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for training DiffuCoder models."""
    
    # Model configuration
    model_name_or_path: str = "apple/DiffuCoder-7B-Base"
    dtype: str = "bfloat16"
    
    # Data configuration
    dataset_name: str = ""
    dataset_prompt_column: str = "question"
    system_prompt: str = "You are a helpful assistant."
    max_prompt_length: int = 200
    max_completion_length: int = 256
    
    # Training hyperparameters
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 0.2
    warmup_ratio: float = 0.0001
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {"min_lr_rate": 0.1})
    
    # Training settings
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 5
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    eval_steps: int = 500
    save_steps: int = 2000
    logging_steps: int = 10
    
    # GRPO specific settings
    sync_ref_model: bool = True
    ref_model_sync_steps: int = 64
    beta: float = 0.01
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    scale_rewards: bool = False
    num_generations: int = 10
    num_iterations: int = 2
    
    # Diffusion specific settings
    diffusion_steps: int = 256
    random_masking: bool = True
    p_mask_prompt: float = 0.0
    generation_batch_size: int = 10
    generation_temperature: float = 1.0
    
    # Reward configuration
    reward_funcs: List[str] = field(default_factory=lambda: ["code", "code_format"])
    reward_weights: List[float] = field(default_factory=lambda: [2.0, 0.5])
    e2b_router_url: str = "0.0.0.0:8000"
    
    # Output and logging
    output_dir: str = "./output"
    overwrite_output_dir: bool = True
    save_total_limit: int = 5
    save_strategy: str = "steps"
    logging_first_step: bool = True
    logging_strategy: str = "steps"
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    wandb_project: str = "jax-diffucoder"
    wandb_entity: Optional[str] = None
    
    # System settings
    seed: int = 42
    data_seed: Optional[int] = None
    
    # TPU specific settings
    tpu_num_cores: Optional[int] = None
    model_parallel_size: int = 1
    data_parallel_size: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.data_seed is None:
            self.data_seed = self.seed
        
        # Calculate effective batch size
        self.train_batch_size = (
            self.per_device_train_batch_size * 
            self.gradient_accumulation_steps
        )
        
        # Validate reward configuration
        if len(self.reward_funcs) != len(self.reward_weights):
            raise ValueError(
                f"Number of reward functions ({len(self.reward_funcs)}) "
                f"must match number of weights ({len(self.reward_weights)})"
            )


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Evaluation datasets
    eval_datasets: List[str] = field(default_factory=lambda: ["humaneval", "mbpp"])
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    num_samples: int = 1
    
    # Evaluation settings
    timeout: float = 10.0
    max_processes: int = 4
    
    # Output settings
    save_generations: bool = True
    save_metrics: bool = True