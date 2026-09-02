# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by BeindBeyond Ltd. and/or its affiliates. on 2026-01-10.

import functools
import gc
import json
import logging
import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from time import time
from typing import Optional
 
import pytz
import torch
import torch.distributed as dist
import yaml
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.optimization import (
    get_cosine_with_min_lr_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
 
from BeingH.dataset.base_dataset import PackedDataset, collate_wrapper, RobotDatasetConfig
from BeingH.model.beingvla import BeingH, BeingHConfig
from BeingH.train.train_utils import (
    create_logger,
    loading_ckpt_from_pretrained,
    add_special_tokens,
    get_latest_ckpt,
    #save_dataset_metadata
)
from BeingH.train.fsdp_utils import (
    FSDPCheckpoint,
    FSDPConfig,
    grad_checkpoint_check_fn,
    fsdp_wrapper,
)
from BeingH.train.rng_state import (
    capture_rng_state,
    load_rng_state,
    save_rng_state,
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# ==============================================================================
# Argument Definitions
# ==============================================================================

@dataclass
class ModelArguments:
    """Model architecture and configuration arguments."""

    # Model paths
    mllm_path: str = field(
        default="",
        metadata={"help": "Path of the pretrained MLLM model."}
    )
    vit_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    expert_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )

    # Architecture
    connector_arch: Optional[str] = field(
        default="internvl_connector",
        metadata={'help': 'connector name for MLLM.'}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    attn_mode: str = field(
        default="causal",
        metadata={'help': 'Set the attention mode (causal/full/noise).'}
    )

    # Action generation
    gen_action_type: str = field(
        default="prop_hidden", metadata={'help': 'Set generation action hidden state'}
    )
    layer_select_for_action: int = field(
        default=-1, metadata={'help': 'layer_select_for_action'}
    )
    action_token_num: int = field(
        default=16, metadata={'help': 'action_token_num'}
    )
    action_chunk_length: int = field(
        default=16, metadata={'help': 'action chunk.'}
    )
    num_inference_timesteps: int = field(
        default=4, metadata={'help': 'num_inference_timesteps'}
    )
    use_flow_matching: bool = field(
        default=False, metadata={'help': 'Use Flow Matching'}
    )

    # Action expert (MoT)
    use_expert: bool = field(
        default=True, metadata={'help': 'Use Expert'}
    )

    # Multi-view configuration
    max_view_num: int = field(
        default=-1,
        metadata={"help": "max image view we use for robot control episodes"}
    )
    use_fixed_view: bool = field(
        default=False,
        metadata={"help": "whether to use ego view only"}
    )
    
    # MPG enhancement
    use_mpg: bool = field(
        default=False,
        metadata={"help": "Enable MPG for action refinement"}
    )
    mpg_num_projections: int = field(
        default=32,
        metadata={"help": "Sliced Wasserstein projections for MPG"}
    )
    mpg_lambda: float = field(
        default=0.0,
        metadata={"help": "MPG residual strength (e.g., 0.1)"}
    )
    mpg_use_stop_gradient: bool = field(
        default=True,
        metadata={"help": "Stop gradient on MPG gate"}
    )
    mpg_refinement_iters: int = field(
        default=1,
        metadata={"help": "MPG refinement iterations at inference"}
    )
    mpg_gate_temperature: float = field(
        default=2.0,
        metadata={"help": "MPG gate temperature (higher = softer gating)"}
    )

    # Training-time RTC (Real-Time Control)
    use_training_time_rtc: bool = field(
        default=False,
        metadata={"help": "Enable training-time RTC for temporal continuity"}
    )
    simulated_delay: Optional[int] = field(
        default=None,
        metadata={"help": "Max simulated delay for RTC (e.g., 10 for ~200ms latency)"}
    )
    rtc_delay_exp_weight: bool = field(
        default=True,
        metadata={"help": "Use exponential weighting for RTC delay sampling"}
    )
    use_inference_prefix_overwrite: bool = field(
        default=False,
        metadata={"help": "Enable prefix overwriting at inference"}
    )


@dataclass
class DataTrainingArguments:
    """Data loading and preprocessing arguments."""

    # Dataset configuration
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prompt_template: str = field(
        default="long", metadata={'help': 'prompt_template'}
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )

    # Image preprocessing
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )

    # Data loading
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    
    # Augmentation
    vit_dropout_prob: float = field(
        default=0.,
        metadata={"help": "Probability of dropping ViT visual features during training for robotic tasks."}
    )
    state_dropout_prob: float = field(
        default=0.,
        metadata={"help": "Probability of dropping state features during training for robotic tasks."}
    )

    # Action space
    is_relative: bool = field(
        default=False,
        metadata={'help': 'Use relative action or not'},
    )
    is_abstract_action: bool = field(
        default=False,
        metadata={'help': 'Use abstract action or not'},
    )
    override_stats_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pre-computed, structured statistics JSON file to override automatic calculation."}
    )
    

@dataclass
class TrainingArguments(TrainingArguments):
    """Extended training arguments for FSDP and custom features."""

    # Data packing
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    max_num_tokens_per_sample: int = field(
        default=8192,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=32768,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )

    # Optimization
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    beta1: float = field(default=0.9, metadata={"help": "AdamW β₁"})
    beta2: float = field(default=0.999, metadata={"help": "AdamW β₂"})
    eps: float = field(default=1e-8, metadata={"help": "AdamW ε"})
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )

    # Learning rate schedule
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    
    # FSDP configuration
    use_gradient_checkpoint: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing, will slow down training speed."}
    )
    sharding_strategy: str = field(
        default="SHARD_GRAD_OP",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    
    # Module freezing
    freeze_mllm: bool = field(
        default=False,
        metadata={"help": "Keep the entire vlm model weights fixed (no gradient updates)."}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vit_mlp: bool = field(
        default=True,
        metadata={"help": "Keep ViT MLP weights fixed during training."
                  "It will be used only when freeze_mllm or freeze_vit is True"
                  "so default it is True (allow mllm or vit to be entirely frozen by default)"}
    )

    # Checkpoint management
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    save_model_only: bool = field(
        default=False,
        metadata={"help": "Save only model weights, ignoring optimizer/scheduler states."}
    )
    save_last: bool = field(
        default=False,
        metadata={"help": "whether to save checkpoint for the last step."}
    )
    save_steps_start: int = field(
        default=25000,
        metadata={"help": "Start saving checkpoints only after this step (exclusive)."}
    )

    logging_dir: str = field(
        default="results/tensorboard",
        metadata={"help": "tensorboard project dir" }
    )

    # --- dataset metadata ---
    save_merged_metadata: bool = field(
        default=False,
        metadata={"help": "Save merged statistics across all dataset variants (default: nested format for backward compatibility)."}
    )

# ==============================================================================
# Helper Functions
# ==============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return dist.get_rank() % torch.cuda.device_count()


def setup_logging(output_dir: str, rank: int) -> logging.Logger:
    """Create logger for current process."""
    return create_logger(output_dir, rank) if rank == 0 else create_logger(None, rank)


def setup_tensorboard(logging_dir: str, checkpoint_name: str, rank: int) -> Optional[SummaryWriter]:
    """Initialize TensorBoard writer for rank 0."""
    if rank != 0:
        return None
    
    tz = pytz.timezone('Asia/Shanghai')
    timestamp = datetime.now(tz).strftime(f"%Y-%m-%d-%H_{checkpoint_name}")
    return SummaryWriter(log_dir=f'{logging_dir}/{timestamp}')


def resolve_checkpoint_path(training_args: TrainingArguments, logger: logging.Logger):
    """Determine checkpoint path and mode based on arguments."""
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.output_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
        else:
            logger.info(f"Auto-resuming from: {resume_from}")
            resume_model_only = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
    
    return resume_from, resume_model_only


def apply_model_freezing(
    model: BeingH,
    training_args: TrainingArguments,
    mllm_layer_names: list,
    num_old_tokens: int,
    logger: logging.Logger
):
    """Apply module freezing based on training arguments."""

    # Freeze multimodal LLM (keep only expert/action parameters active)
    if training_args.freeze_mllm:
        for name, param in model.named_parameters():
            if name in mllm_layer_names:
                param.requires_grad = False
            if "connector" in name and training_args.freeze_vit_mlp:
                param.requires_grad = False

        model.vit_model.eval()
        logger.info("Froze multimodal LLM (except action expert)")
    
    # Freeze language model (optionally unfreeze new token embeddings)
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
        
        # Selectively unfreeze new token embeddings via gradient hook
        if model.gen_action_type == "action_token":
            # Most reliable method to unfreeze new token embeddings
            logger.info("Unfreezing new token embeddings via gradient hook")

            # 1. Get the input embedding layer and its weights
            input_embeddings = model.language_model.get_input_embeddings()
            input_embeddings_weight = input_embeddings.weight

            input_embeddings_weight.requires_grad = True
            # This hook is called after gradients are computed but before the optimizer uses them
            def selective_grad_hook(grad):
                # Multiply the computed gradients with our mask
                # This way, all old token gradients become 0, only new token gradients are preserved
                grad_mask = torch.zeros_like(grad)
                grad_mask[num_old_tokens:] = 1.0
                return grad * grad_mask

            input_embeddings_weight.register_hook(selective_grad_hook)
            
            logger.info(f"Only tokens from index {num_old_tokens} onwards will be updated")

    # Freeze vision transformer
    if training_args.freeze_vit:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
        logger.info("Froze vision transformer")


def _merge_statistics(statistics_list):
    """
    Merge multiple DatasetStatistics objects into one.

    Merge strategy:
    - mean: average across all statistics
    - std: average across all statistics
    - max: take maximum across all statistics
    - min: take minimum across all statistics
    - q01: take minimum (conservative lower bound)
    - q99: take maximum (conservative upper bound)

    Args:
        statistics_list: List of DatasetStatistics objects

    Returns:
        Merged statistics as a dict (JSON-serializable)
    """
    import numpy as np

    if len(statistics_list) == 1:
        return statistics_list[0].model_dump(mode="json")

    # Get the first statistics as template
    template = statistics_list[0].model_dump(mode="json")
    merged = {"state": {}, "action": {}}

    # Merge each modality (state, action)
    for modality in ["state", "action"]:
        if modality not in template:
            continue

        # Merge each feature key
        for feature_key in template[modality].keys():
            # Collect values from all statistics that have this feature
            all_values = {}
            for stat_key in ["mean", "std", "max", "min", "q01", "q99"]:
                all_values[stat_key] = []

            for stats in statistics_list:
                stats_dict = stats.model_dump(mode="json")
                if modality in stats_dict and feature_key in stats_dict[modality]:
                    feature_stats = stats_dict[modality][feature_key]
                    for stat_key in all_values.keys():
                        if stat_key in feature_stats:
                            all_values[stat_key].append(np.array(feature_stats[stat_key]))

            # Merge values
            merged_feature = {}
            for stat_key, values in all_values.items():
                if not values:
                    continue
                stacked = np.stack(values, axis=0)  # (N, dim)
                if stat_key == "mean":
                    merged_feature[stat_key] = np.mean(stacked, axis=0).tolist()
                elif stat_key == "std":
                    merged_feature[stat_key] = np.mean(stacked, axis=0).tolist()
                elif stat_key == "max":
                    merged_feature[stat_key] = np.max(stacked, axis=0).tolist()
                elif stat_key == "min":
                    merged_feature[stat_key] = np.min(stacked, axis=0).tolist()
                elif stat_key == "q01":
                    merged_feature[stat_key] = np.min(stacked, axis=0).tolist()
                elif stat_key == "q99":
                    merged_feature[stat_key] = np.max(stacked, axis=0).tolist()

            merged[modality][feature_key] = merged_feature

    return merged


def save_dataset_metadata(train_dataset, training_args, dataset_meta, logger):
    """
    Saves the metadata of the training dataset(s) to a JSON file.
    This function can handle raw datasets, or datasets wrapped in a ConcatDataset.

    Supports three formats:
    1. Default (nested): Preserve individual sub-dataset statistics
    2. Optional (merged): Save merged statistics + individual variants
    3. NEW (hierarchical): Save 2-level hierarchical statistics (embodiment → task)
       with {dataset_name}_variants key for inference-time variant selection
    """
    output_dir = Path(training_args.output_dir)
    exp_cfg_dir = output_dir / "experiment_cfg"
    exp_cfg_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() != 0:
        return

    grouped_dataset_meta_list = list(dataset_meta.items())

    for idx, grouped_dataset in enumerate(train_dataset.grouped_datasets):
        grouped_dataset_name = grouped_dataset_meta_list[idx][0]
        if not hasattr(grouped_dataset, 'dataset_metadatas'):
            continue

        specific_metadatas = grouped_dataset.dataset_metadatas

        # Serialize individual sub-dataset metadatas
        serializable_metadatas = {
            sub_dataset_name: metadata_obj.model_dump(mode="json")
            for sub_dataset_name, metadata_obj in specific_metadatas.items()
        }

        metadata_filepath = exp_cfg_dir / f"{grouped_dataset_name}_metadata.json"
        metadata_json = {}

        if os.path.exists(metadata_filepath):
            try:
                with open(metadata_filepath, "r") as f:
                    metadata_json = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Warning: Could not decode existing metadata file at {metadata_filepath}.")

        # Check for hierarchical stats or stats_sources (new format)
        has_hierarchical_stats = (
            hasattr(grouped_dataset, 'hierarchical_stats') and
            grouped_dataset.hierarchical_stats is not None
        )
        has_stats_sources = (
            hasattr(grouped_dataset, 'stats_sources') and
            grouped_dataset.stats_sources
        )

        if has_hierarchical_stats:
            # NEW: Save hierarchical metadata
            logger.info(f"Saving hierarchical metadata for '{grouped_dataset_name}'")

            hierarchical_stats = grouped_dataset.hierarchical_stats
            grouping_structure = getattr(grouped_dataset, 'grouping_structure', {})

            # Get merged metadata (use first dataset's modalities as template)
            merged_metadata = list(specific_metadatas.values())[0]

            # Build hierarchical metadata structure
            metadata_json.update({
                grouped_dataset_name: {
                    "statistics": hierarchical_stats['total']['statistics'],
                    "modalities": merged_metadata.modalities.model_dump(mode="json") if hasattr(merged_metadata, 'modalities') else {},
                    "embodiment_tag": getattr(merged_metadata, 'embodiment_tag', 'multi_embodiment'),
                    "hierarchical_stats": hierarchical_stats,
                    "grouping_structure": grouping_structure
                }
            })

            logger.info(f"✓ Hierarchical metadata saved:")
            logger.info(f"  - Total datasets: {hierarchical_stats['total']['dataset_count']}")
            logger.info(f"  - Embodiment groups: {list(hierarchical_stats['embodiment_groups'].keys())}")
            logger.info(f"  - Task datasets: {len(hierarchical_stats['task_datasets'])}")

        elif has_stats_sources:
            # NEW: Save hierarchical stats metadata (Level 1 + Level 2)
            logger.info(f"Saving hierarchical stats metadata for '{grouped_dataset_name}'")

            from configs.dataset_info import DATASET_INFO

            stats_sources = grouped_dataset.stats_sources
            stats_level = getattr(grouped_dataset, 'stats_level', 'auto')

            # === Step 1: Collect embodiment information ===
            embodiment_groups = {}  # {embodiment_name: [sub_names]}
            embodiment_tags = {}  # {embodiment_name: embodiment_tag}

            # Define priority order for registry search (prefer uni_posttrain for hierarchical datasets)
            REGISTRY_PRIORITY = ['uni_posttrain']

            for sub_name in specific_metadatas.keys():
                # Find embodiment info from DATASET_INFO with priority search
                matched_registry = None
                dataset_meta_info = None

                # First, try priority registries (e.g., uni_posttrain for hierarchical structure)
                for priority_reg in REGISTRY_PRIORITY:
                    if priority_reg in DATASET_INFO:
                        registry_datasets = DATASET_INFO[priority_reg]
                        if isinstance(registry_datasets, dict) and sub_name in registry_datasets:
                            matched_registry = priority_reg
                            dataset_meta_info = registry_datasets[sub_name]
                            break

                # If not found in priority registries, search all registries
                if not dataset_meta_info:
                    for registry_name, registry_datasets in DATASET_INFO.items():
                        if isinstance(registry_datasets, dict) and sub_name in registry_datasets:
                            matched_registry = registry_name
                            dataset_meta_info = registry_datasets[sub_name]
                            break

                # Process embodiment info if found
                if dataset_meta_info:
                    embodiment_name = dataset_meta_info.get('embodiment')
                    embodiment_tag = dataset_meta_info.get('embodiment_tag')

                    if embodiment_name:
                        embodiment_groups.setdefault(embodiment_name, []).append(sub_name)

                        # Store embodiment_tag mapping
                        if embodiment_tag:
                            embodiment_tags[embodiment_name] = embodiment_tag

            # === Step 2: Build variants (Level 1 + Level 2) ===
            variants = {}

            # Add Level 1: Task-specific variants (directly from training)
            for sub_name, meta in specific_metadatas.items():
                variants[sub_name] = {
                    **meta.model_dump(mode="json"),
                    "stats_level": "task"
                }

            # Add Level 2: Embodiment-merged variants
            for embodiment_name in embodiment_groups.keys():
                try:
                    subtask_names = embodiment_groups[embodiment_name]
                    template_meta = specific_metadatas[subtask_names[0]]

                    # Merge statistics from all subtasks under this embodiment
                    subtask_metas = [specific_metadatas[name] for name in subtask_names]
                    merged_statistics = _merge_statistics([m.statistics for m in subtask_metas])

                    # Get embodiment_tag from mapping (use template's tag if not found)
                    embodiment_tag_value = embodiment_tags.get(embodiment_name, template_meta.embodiment_tag.value)

                    variants[embodiment_name] = {
                        "statistics": merged_statistics,
                        "modalities": template_meta.modalities.model_dump(mode="json"),
                        "embodiment_tag": embodiment_tag_value,
                        "stats_level": "embodiment",
                    }

                    logger.info(f"  ✓ Level 2 (embodiment): {embodiment_name}")
                    logger.info(f"    Merged from {len(subtask_names)} subtasks: {subtask_names}")

                except Exception as e:
                    logger.warning(f"  ⚠ Failed to create embodiment variant for {embodiment_name}: {e}")

            # === Step 3: Set top-level default (first variant for backward compatibility) ===
            first_variant_name = list(specific_metadatas.keys())[0]
            first_variant_meta = specific_metadatas[first_variant_name]

            metadata_json.update({
                grouped_dataset_name: {
                    **first_variant_meta.model_dump(mode="json"),
                    "default_variant": first_variant_name,
                    "stats_level": stats_level,
                },
                f"{grouped_dataset_name}_variants": variants
            })

            logger.info(f"✓ Hierarchical stats saved:")
            logger.info(f"  - Default: {first_variant_name}")
            logger.info(f"  - Level 1 (task): {list(specific_metadatas.keys())}")
            logger.info(f"  - Level 2 (embodiment): {list(embodiment_groups.keys())}")
            logger.info(f"  - Total: {len(variants)} variants")

        else:
            # Legacy formats
            save_merged = getattr(training_args, 'save_merged_metadata', False)

            if save_merged:
                # User explicitly enabled merged metadata
                # Save both merged and individual variants (even for single dataset)
                from copy import deepcopy
                import numpy as np

                # Merge statistics
                merged_metadata = deepcopy(list(specific_metadatas.values())[0])

                # Get all modality keys (state, action, etc.)
                statistics_dict = vars(merged_metadata.statistics) if hasattr(merged_metadata.statistics, '__dict__') else {}

                for modality_key in statistics_dict.keys():
                    if modality_key.startswith('_'):
                        continue

                    modality_stats = getattr(merged_metadata.statistics, modality_key, None)
                    if modality_stats is None:
                        continue

                    # Get all feature keys (joint_position, etc.)
                    modality_stats_dict = vars(modality_stats) if hasattr(modality_stats, '__dict__') else {}

                    for feature_key in modality_stats_dict.keys():
                        if feature_key.startswith('_'):
                            continue
                        feature_stats = getattr(modality_stats, feature_key, None)
                        if feature_stats is None or not hasattr(feature_stats, 'min'):
                            continue

                        # Collect stats from all variants
                        all_mins, all_maxes = [], []
                        for metadata in specific_metadatas.values():
                            mod_stats = getattr(metadata.statistics, modality_key, None)
                            if mod_stats is None:
                                continue
                            feat_stats = getattr(mod_stats, feature_key, None)
                            if feat_stats is None:
                                continue
                            if hasattr(feat_stats, 'min') and feat_stats.min:
                                all_mins.append(np.array(feat_stats.min))
                            if hasattr(feat_stats, 'max') and feat_stats.max:
                                all_maxes.append(np.array(feat_stats.max))

                        # Merge: min of mins, max of maxes
                        if all_mins:
                            feature_stats.min = np.minimum.reduce(all_mins).tolist()
                        if all_maxes:
                            feature_stats.max = np.maximum.reduce(all_maxes).tolist()

                # Save merged + variants
                metadata_json.update({
                    grouped_dataset_name: merged_metadata.model_dump(mode="json"),
                    f"{grouped_dataset_name}_variants": serializable_metadatas
                })
                logger.info(f"Saved merged metadata + {len(specific_metadatas)} variant metadatas")
            else:
                # Default: Save individual sub-dataset metadatas (backward compatible)
                metadata_json.update({
                    grouped_dataset_name: serializable_metadatas
                })
                logger.info(f"Saved {len(specific_metadatas)} individual sub-dataset metadata(s)")

        with open(metadata_filepath, "w") as f:
            json.dump(metadata_json, f, indent=4)
        logger.info(f"Successfully saved dataset metadata to {metadata_filepath}")


# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():

    device = setup_distributed()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    world_size = dist.get_world_size()
    logger = setup_logging(training_args.output_dir, dist.get_rank())

    checkpoint_name = (
        "_".join(training_args.output_dir.split("/")[-2:])
        if "stage" in training_args.output_dir
        else training_args.output_dir.split("/")[-1]
    )
    tb_writer = setup_tensorboard(training_args.logging_dir, checkpoint_name, dist.get_rank())
    
    set_seed(training_args.seed)

    # Resolve checkpoint path
    resume_from, resume_model_only = resolve_checkpoint_path(training_args, logger)

    # =========================================================================
    # CRITICAL FIX: Sync action_token_num with action_chunk_length for flow matching
    # Must happen BEFORE add_special_tokens(), which depends on action_token_num.
    # =========================================================================
    if model_args.use_flow_matching:
        if model_args.action_token_num != model_args.action_chunk_length:
            logger.warning(
                f"[Flow Matching] action_token_num ({model_args.action_token_num}) != "
                f"action_chunk_length ({model_args.action_chunk_length}). "
                f"Overriding action_token_num to {model_args.action_chunk_length}."
            )
            model_args.action_token_num = model_args.action_chunk_length
        logger.info(f"[Flow Matching] action_token_num = action_chunk_length = {model_args.action_token_num}")

    language_model, vit_model, connector, llm_config, vit_config, Tokenizer, mllm_layer_names = loading_ckpt_from_pretrained(model_args, logger)

    logger.info(f"LLM Config: {llm_config}")
    logger.info(f"ViT Config: {vit_config}")

    # Setup tokenizer for model:
    tokenizer_path = model_args.mllm_path if model_args.mllm_path else model_args.llm_path
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)  
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer, new_token_ids, num_new_tokens, action_tokens = add_special_tokens(
        tokenizer, model_args
    )
    num_old_tokens = len(tokenizer) - num_new_tokens
    tokenizer.padding_side = "left"  
    
    logger.info(f"Tokenizer padding side: {tokenizer.padding_side}")
    logger.info(f"Tokenizer truncation side: {tokenizer.truncation_side}")

    config = BeingHConfig(
        llm_config=llm_config,
        vit_config=vit_config,
        connector_arch=model_args.connector_arch,
        template = data_args.conv_style,
        downsample_ratio = data_args.down_sample_ratio,
        force_image_size=data_args.force_image_size,
        select_layer = model_args.vision_select_layer,
        # action
        gen_action_type=model_args.gen_action_type,
        action_chunk_length=model_args.action_chunk_length,  # CRITICAL: Must match data loader!
        layer_select_for_action=model_args.layer_select_for_action,
        action_token_num=model_args.action_token_num,
        num_inference_timesteps=model_args.num_inference_timesteps,
        prompt_template=data_args.prompt_template,
        use_expert=model_args.use_expert,
        use_flow_matching=model_args.use_flow_matching,
        attn_mode=model_args.attn_mode,
        # New checkpoints record controlled invalid-dim Flow Matching explicitly.
        # Older checkpoints without this field also default to the fixed inference
        # path in BeingHConfig; set the field to False only to reproduce legacy behavior.
        mask_invalid_action_dims=True,
        # MPG parameters
        use_mpg=model_args.use_mpg,
        mpg_num_projections=model_args.mpg_num_projections,
        mpg_lambda=model_args.mpg_lambda,
        mpg_use_stop_gradient=model_args.mpg_use_stop_gradient,
        mpg_refinement_iters=model_args.mpg_refinement_iters,
        mpg_gate_temperature=model_args.mpg_gate_temperature,
        # Training-Time RTC parameters
        use_training_time_rtc=model_args.use_training_time_rtc,
        simulated_delay=model_args.simulated_delay,
        rtc_delay_exp_weight=model_args.rtc_delay_exp_weight,
        use_inference_prefix_overwrite=model_args.use_inference_prefix_overwrite,
    )
    config.llm_config._attn_implementation = 'flash_attention_2'

    # Initialize model
    model = BeingH(language_model, vit_model, connector, config)
    
    # Setup embeddings
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    logger.info(f"Added {num_new_tokens} new tokens, initialized with average embeddings")

    """Resize position embeddings if needed."""
    logger.info(f'model.config.vision_config.image_size: {model.config.vit_config.image_size}')
    if model.config.vit_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from {model.config.vit_config.image_size} to {data_args.force_image_size}')
        patch_size = model.config.vit_config.patch_size
        model.vit_model.resize_pos_embeddings(
            old_size=model.config.vit_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size
        )

        model.config.vit_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    
    # Configure gradient checkpointing
    model.language_model.config.use_cache = False
    model.vit_model.gradient_checkpointing = True
    model.vit_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            breakpoint()

    # Load dataset configuration
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    
    # Create robot dataset config
    robot_dataset_config = RobotDatasetConfig(
        max_view_num=model_args.max_view_num,
        use_fixed_view=model_args.use_fixed_view,
        gen_action_type=model_args.gen_action_type,
        action_chunk_length=model_args.action_chunk_length,
        is_relative=data_args.is_relative,
        is_abstract_action=data_args.is_abstract_action,
        prompt_template=data_args.prompt_template,
        vit_dropout_prob=data_args.vit_dropout_prob,
        state_dropout_prob = data_args.state_dropout_prob,
        override_stats_path = data_args.override_stats_path
    )
    
    # Create training dataset
    train_dataset = PackedDataset(
        tokenizer=tokenizer,
        template_name=data_args.conv_style,
        grouped_dataset_meta=dataset_meta,
        robot_config=robot_dataset_config,
        special_tokens=new_token_ids,
        force_image_size=data_args.force_image_size,
        img_patch_size=config.vit_config.patch_size,
        img_downsample_ratio=data_args.down_sample_ratio,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=training_args.max_num_tokens_per_sample,
        max_num_tokens=training_args.max_num_tokens,
        max_buffer_size=training_args.max_buffer_size,
        prefer_buffer_before=training_args.prefer_buffer_before,
        attn_mode=model_args.attn_mode,
        local_rank=dist.get_rank(),
        world_size=world_size,
        num_workers=training_args.num_workers,
        is_train=True,
        logger=logger,
    )

    apply_model_freezing(model, training_args, mllm_layer_names, num_old_tokens, logger)

    """Log model parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params / 1_000_000:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1_000_000:.2f}M")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # Save dataset metadata
    save_dataset_metadata(train_dataset, training_args, dataset_meta, logger)

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, 
        num_workers=training_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor if training_args.num_workers>0 else None,
        persistent_workers=True if training_args.num_workers>0 else False,
        multiprocessing_context='fork' if training_args.num_workers > 0 else None,  
    )

    # Setup FSDP
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    
    # Load checkpoint and wrap with FSDP
    model = FSDPCheckpoint.try_load_ckpt(resume_from, logger, model)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    
    if training_args.use_gradient_checkpoint:
        apply_activation_checkpointing(
            fsdp_model, 
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ), 
            check_fn=grad_checkpoint_check_fn
        )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_args.learning_rate, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=training_args.weight_decay,
    )
    warmup_steps = min(training_args.max_steps*training_args.warmup_ratio, training_args.warmup_steps)
    
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=training_args.max_steps,
                min_lr=training_args.min_lr,
            )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError
    
    train_step = 0
    if not resume_model_only:
        optimizer, scheduler, train_step = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Restore randomness only for a full resume.  A fresh run must honor its
    # configured seed even when another experiment used the same checkout.
    if resume_from is not None and not resume_model_only:
        restored_rng_path = load_rng_state(resume_from, rank=dist.get_rank())
        if restored_rng_path is None:
            logger.info("No RNG snapshot found; continuing from the configured seed")
        else:
            logger.info(f"Restored RNG snapshot from {restored_rng_path}")

    fsdp_model.train()

    logger.info(f"Starting training from step {train_step} to {training_args.max_steps}")
    loop_start_time = time() 
    optimizer.zero_grad()
    
    accum_action_loss, accum_ce_loss = 0, 0
    total_norm = torch.tensor(0.0, device=device)

    for micro_step, data in enumerate(train_loader, start=0):
        curr_step = micro_step // training_args.gradient_accumulation_steps

        # Move data to device
        data = data.cuda(device).to_dict()

        # Forward pass
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss_dict = fsdp_model(**data)
            action_loss = loss_dict['action_loss']
            und_loss = loss_dict['und_loss']

        # Backward pass
        loss = action_loss + und_loss
        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()
        
        accum_action_loss += action_loss / training_args.gradient_accumulation_steps
        accum_ce_loss += und_loss / training_args.gradient_accumulation_steps
        
        # Optimizer step
        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if curr_step>0 and curr_step % training_args.logging_steps==0:
                """Log training metrics to console and TensorBoard."""
                torch.cuda.synchronize()

                elapsed_time_total = time() - loop_start_time
                avg_steps_per_sec = curr_step / elapsed_time_total if elapsed_time_total > 0 else 0
                remaining_steps = training_args.max_steps - curr_step
                estimated_remaining_seconds = remaining_steps / avg_steps_per_sec if avg_steps_per_sec > 0 else 0
                eta_str = str(timedelta(seconds=int(estimated_remaining_seconds)))
                elapsed_time_str = str(timedelta(seconds=int(elapsed_time_total)))
                
                avg_act_loss = torch.tensor(accum_action_loss / training_args.logging_steps, device=device)
                dist.all_reduce(avg_act_loss, op=dist.ReduceOp.SUM)
                avg_act_loss = avg_act_loss.item() / world_size
                
                avg_und_loss = torch.tensor(accum_ce_loss / training_args.logging_steps, device=device)
                dist.all_reduce(avg_und_loss, op=dist.ReduceOp.SUM)
                avg_und_loss = avg_und_loss.item() / world_size
                
                mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
                dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)

                message = (
                        f"step: {curr_step:07d}/{training_args.max_steps} | "
                        f"act_loss: {avg_act_loss:.4f} | "
                        f"und_loss: {avg_und_loss:.4f} | "
                        f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                        f"steps/s: {avg_steps_per_sec:.2f} | "
                        f"ETA: {eta_str} | "
                        f"max_mem: {mem_cache.item():.0f}MB"
                    )

                if dist.get_rank() == 0:
                    logger.info(message)

                # Detailed logging every 10x
                if curr_step % (training_args.logging_steps*10)==0 and dist.get_rank()==0:
                    detail_msg = (
                            f"###########| elapsed_time: {elapsed_time_str}"
                            f"| num_und: {data['num_und_samples']*world_size}"
                            f"| num_gen: {data['num_gen_samples']*world_size}"
                    )
                    logger.info(detail_msg)

                if tb_writer is not None:
                    tb_writer.add_scalar('train/loss', action_loss, curr_step)
                    tb_writer.add_scalar('train/und_loss', und_loss, curr_step)
                    tb_writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], curr_step)
                    tb_writer.add_scalar('train/grad_norm', total_norm, curr_step)

                    # MPG metrics logging
                    try:
                        # Access the underlying model through FSDP wrapper
                        unwrapped_model = fsdp_model.module if hasattr(fsdp_model, 'module') else fsdp_model
                        if hasattr(unwrapped_model, 'use_mpg') and unwrapped_model.use_mpg:
                            if hasattr(unwrapped_model, 'last_mpg_gate') and unwrapped_model.last_mpg_gate is not None:
                                tb_writer.add_scalar('train/mpg_gate', unwrapped_model.last_mpg_gate, curr_step)
                            if hasattr(unwrapped_model, 'last_mpg_transport_cost') and unwrapped_model.last_mpg_transport_cost is not None:
                                tb_writer.add_scalar('train/mpg_transport_cost', unwrapped_model.last_mpg_transport_cost, curr_step)
                    except Exception as e:
                        # Silently ignore if we can't access MPG metrics (e.g., FSDP sharding)
                        pass

                # Clear accumulators after logging
                accum_action_loss = 0.
                accum_ce_loss = 0.
                
            if curr_step > training_args.save_steps_start and curr_step % training_args.save_steps == 0:
                # Get dataset name for metadata copying
                dataset_name = list(dataset_meta.keys())[0] if dataset_meta else None
                FSDPCheckpoint.fsdp_save_ckpt(
                    ckpt_dir=training_args.output_dir,
                    train_steps=curr_step,
                    model=fsdp_model,
                    tokenizer=tokenizer,
                    optimizer=None if training_args.save_model_only else optimizer,
                    scheduler=None if training_args.save_model_only else scheduler,
                    fsdp_config=fsdp_config,
                    logger=logger,
                    dataset_name=dataset_name,
                )
                # Keep both a checkpoint-local snapshot (for resuming that
                # checkpoint) and a latest snapshot in the run directory.
                rng_state = capture_rng_state()
                checkpoint_dir = Path(training_args.output_dir) / f"{curr_step:07d}"
                save_rng_state(checkpoint_dir, rank=dist.get_rank(), state=rng_state)
                save_rng_state(training_args.output_dir, rank=dist.get_rank(), state=rng_state)

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        if curr_step > training_args.max_steps:
            logger.info(f"Reached total_steps={training_args.max_steps}, stopping training.")
            break
    
    if training_args.save_last and curr_step > 0:  

        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = fsdp_model.state_dict()
        
        if dist.get_rank() == 0:
            logger.info("Saving model...")
            unwrapped_model = fsdp_model.module
            unwrapped_model.save_pretrained(
                training_args.output_dir,
                state_dict=model_state_dict,
                safe_serialization=False 
            )
            
            logger.info(f"✓ Model saved to {training_args.output_dir}")
            if tb_writer is not None:
                tb_writer.close()
            tokenizer.save_pretrained(training_args.output_dir)

        # ``save_last`` writes the final model directly into output_dir, so
        # its RNG snapshot lives there as well.
        save_rng_state(
            training_args.output_dir,
            rank=dist.get_rank(),
            state=capture_rng_state(),
        )
        
        gc.collect()
        dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        logger.info(f"Final checkpoint saved at step {curr_step}")
          
    logger.info("Done!")
    dist.destroy_process_group()

    
if __name__ == '__main__':
    main()
