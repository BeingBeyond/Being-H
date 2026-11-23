# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import glob
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
 
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoTokenizer

from configs.data_config import DATA_CONFIG_MAP, ModalityConfig
from BeingH.dataset.preprocess import build_vit_transform_base
from BeingH.model import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
from BeingH.model.beingvla import BeingH, BeingHConfig
from BeingH.model.layers import InternVLConnector
from BeingH.model.llm.qwen2 import Qwen2Tokenizer
from BeingH.model.vit_model.internvit_navit import InternVisionConfig, InternVisionModel
from BeingH.utils.schema import DatasetMetadata, EmbodimentTag
 
def get_default_device() -> torch.device:
    """Return CUDA if available, then Intel XPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


# Register custom config
AutoConfig.register("beingh", BeingHConfig)
AutoModel.register(BeingHConfig, BeingH)


# ==============================================================================
# Constants
# ==============================================================================
 
BLOCK_SIZE = 128
 
# Model version configurations
VERSION_CONFIGS = {
    "qwen2.5": (Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM, AutoTokenizer),
}


def load_safetensors(path):
    safetensor_files = glob.glob(f"{path}/*.safetensors")
    # if safetensor_files:
    print(f"Found {len(safetensor_files)} .safetensors files, loading...")
    state_dict = {}
    for file_path in safetensor_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    print("Weights loaded from .safetensors files.")
    return state_dict


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension (axis=0).
    NOTE: We use axis=0 explicitly to avoid removing other size-1 dimensions
    (e.g., gripper_position with shape (1, 16, 1) should become (16, 1), not (16,))
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Only remove batch dimension
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Only remove batch dimension
        else:
            squeezed_data[k] = v
    return squeezed_data


# ==============================================================================
# BeingH Policy
# ==============================================================================
class BasePolicy(ABC):
    """Abstract base class for robot control policies."""

    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError
    

class BeingHPolicy(BasePolicy):
    """
    Inference wrapper for trained Being-H VLA model.
    
    Features:
    - Multi-view image processing
    - State and action normalization
    - RTC (Real-Time Chunking) support
    - MPG parameter overrides
    - Hierarchical metadata variant selection
    """
    def __init__(
        self,
        model_path: str,
        data_config_name: str,  # Data configuration name
        dataset_name: str,
        embodiment_tag: str,
        instruction_template: str,
        prop_pos: str = "front",
        max_view_num: int = -1,
        use_fixed_view: bool = False,
        action_attn_mode: str = "causal",
        device: Optional[Union[int, str]] = None,
        # MPG overrides
        use_mpg: bool = None,
        mpg_lambda: float = None,
        mpg_num_projections: int = None,
        mpg_refinement_iters: int = None,
        mpg_gate_temperature: float = None,
        # Flow matching parameter override
        num_inference_timesteps: int = None,
        # RTC parameter
        enable_rtc: bool = True,
        # Metadata variant selection
        metadata_variant: str = None,
        stats_selection_mode: str = "auto",
    ):
        """
        Initialize BeingH policy.

        Args:
            model_path: Path to trained model checkpoint (self-contained directory)
            data_config_name: Data configuration name
            dataset_name: Dataset name for metadata loading
            embodiment_tag: Robot embodiment identifier
            instruction_template: Template for task instructions
            prop_pos: Proprioception position in sequence
            max_view_num: Maximum camera views (-1 = all)
            use_fixed_view: Use only ego view
            action_attn_mode: Attention mode for actions
            device: Inference device

            use_mpg: Override MPG enable flag
            mpg_lambda: Override MPG residual strength
            mpg_num_projections: Override Sliced Wasserstein projections
            mpg_refinement_iters: Override refinement iterations
            mpg_gate_temperature: Override MPG gate temperature

            num_inference_timesteps: Override diffusion steps
            enable_rtc: Enable Real-Time Chunking
            metadata_variant: Specific metadata variant to use
            stats_selection_mode: Auto-selection mode ('auto', 'task', 'embodiment')
        """
        self.device = torch.device(device) if device is not None else get_default_device()
        self.model_path = model_path
        self.data_config_name = data_config_name
        self.prop_pos = prop_pos
        self.action_attn_mode = action_attn_mode
        self.use_fixed_view = use_fixed_view
        self.max_view_num = max_view_num
        self.dataset_name = dataset_name
        self.metadata_variant = metadata_variant
        self.stats_selection_mode = stats_selection_mode
        self.instruction_template = instruction_template
        self.enable_rtc = enable_rtc

        self.embodiment_tag = EmbodimentTag(embodiment_tag)

        # ===========================
        # Set data config
        # ===========================
        DataConfigClass = DATA_CONFIG_MAP[self.data_config_name]

        self.data_config = DataConfigClass(
            embodiment_tag=self.embodiment_tag,
            use_fixed_view=self.use_fixed_view,
            max_view_num=self.max_view_num,
            obs_indices=[0],
            action_indices=list(range(16)),
        )

        self.unified_mapping = self.data_config.UNIFIED_MAPPING
        self._modality_transform = self.data_config.get_transforms()
        self._modality_transform.eval()

        self.language_key = self.data_config.LANGUAGE_KEYS[0]
        self.num_images = len(self.data_config.VIDEO_KEYS)

        # =====================================
        # Load model components and assemble
        # =====================================
        print(f"\n=== Loading model from {self.model_path} ===")

        config = BeingHConfig.from_pretrained(self.model_path)

        llm_version = self._detect_llm_version(config.llm_config)
        print(f"Detected LLM version: {llm_version}")

        QwenConfigClass, LanguageModelClass, _ = VERSION_CONFIGS[llm_version]
        print(f"Using {LanguageModelClass.__name__}")

        # Convert configs
        llm_config_dict = config.llm_config.to_dict()
        llm_config = QwenConfigClass.from_dict(llm_config_dict)

        # Setup expert config
        expert_config_dict = llm_config_dict.get('expert_config')
        if expert_config_dict:
            if not isinstance(expert_config_dict, dict):
                expert_config_dict = expert_config_dict.to_dict()
            expert_config = QwenConfigClass.from_dict(expert_config_dict)
            llm_config.expert_config = expert_config

        # Setup vision config
        vit_config_dict = config.vit_config.to_dict()
        vit_config = InternVisionConfig.from_dict(vit_config_dict)

        # Update main config
        config.llm_config = llm_config
        config.vit_config = vit_config
        
        # Initialize components
        language_model = LanguageModelClass(config.llm_config)
        vit_model = InternVisionModel(config.vit_config)
        connector = InternVLConnector(
            llm_hidden_size=config.llm_config.hidden_size,
            vit_hidden_size=config.vit_config.hidden_size,
            downsample_ratio=config.downsample_ratio,
        )

        # Assemble model
        self.model = BeingH(language_model, vit_model, connector, config)

        # Load weights
        print("Loading state dict...")
        state_dict = load_safetensors(self.model_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device, dtype=torch.bfloat16)
        self.model.eval()

        # Set core parameters
        # IMPORTANT: Set action_chunk_length from model config (not hardcoded)
        # This ensures inference uses the same chunk length as training
        self.action_chunk_length = self.model.config.action_chunk_length
        patch_size = self.model.config.vit_config.patch_size
        self.num_image_token = self.model.num_image_token = int(
            (self.model.config.force_image_size // patch_size) ** 2 * (0.5 ** 2)
        )

        self.gen_action_type = self.model.config.gen_action_type
        # For flow matching: action_token_num must equal action_chunk_length
        # (one token position per action step)
        self.action_token_num = self.action_chunk_length

        # =====================================
        # Setup and Apply Overrides
        # =====================================
        self._apply_mpg_overrides(
            use_mpg, mpg_lambda, mpg_num_projections, mpg_refinement_iters, mpg_gate_temperature
        )
        self._apply_flow_matching_overrides(num_inference_timesteps)
        self._setup_rtc()

        self._setup_tokenizer()

        _, self.image_transform = build_vit_transform_base(
            is_train=False,
            force_image_size=self.model.config.force_image_size,
            pad2square=False,
            normalize_type='imagenet'
        )

        # ===================
        # Load metadata
        # ===================
        # Load metadata from checkpoint directory (self-contained checkpoint)
        self._load_metadata(Path(self.model_path))

        print("✓ BeingHPolicy initialized successfully")

    def _detect_llm_version(self, llm_config) -> str:
        """Detect LLM version from config."""
        # Check layer module
        if hasattr(llm_config, 'layer_module') and llm_config.layer_module:
            if 'Qwen3' in llm_config.layer_module:
                version = "qwen3"
            elif 'Qwen2' in llm_config.layer_module:
                version = "qwen2.5"
            else:
                version = "qwen2.5"  # Default
        # Check architectures
        elif hasattr(llm_config, 'architectures') and llm_config.architectures:
            if 'Qwen3ForCausalLM' in llm_config.architectures:
                version = "qwen3"
            elif 'Qwen2ForCausalLM' in llm_config.architectures:
                version = "qwen2.5"
            else:
                version = "qwen2.5"


        return version
    
    def _apply_mpg_overrides(
        self,
        use_mpg: Optional[bool],
        mpg_lambda: Optional[float],
        mpg_num_projections: Optional[int],
        mpg_refinement_iters: Optional[int],
        mpg_gate_temperature: Optional[float],
    ):
        """Apply MPG parameter overrides."""
        if use_mpg is not None:
            print(f"Overriding use_mpg: {self.model.config.use_mpg} -> {use_mpg}")
            self.model.config.use_mpg = use_mpg
            self.model.use_mpg = use_mpg

        if mpg_lambda is not None:
            print(f"Overriding mpg_lambda: {self.model.config.mpg_lambda} -> {mpg_lambda}")
            self.model.config.mpg_lambda = mpg_lambda
            if hasattr(self.model, 'mpg') and self.model.mpg is not None:
                self.model.mpg.lambda_strength = mpg_lambda

        if mpg_num_projections is not None:
            print(f"Overriding mpg_num_projections: {self.model.config.mpg_num_projections} -> {mpg_num_projections}")
            self.model.config.mpg_num_projections = mpg_num_projections
            if hasattr(self.model, 'mpg') and self.model.mpg is not None:
                self.model.mpg.num_projections = mpg_num_projections
                self.model.mpg.sliced_wasserstein.num_projections = mpg_num_projections

        if mpg_refinement_iters is not None:
            print(f"Overriding mpg_refinement_iters: {self.model.config.mpg_refinement_iters} -> {mpg_refinement_iters}")
            self.model.config.mpg_refinement_iters = mpg_refinement_iters
            self.model.mpg_refinement_iters = mpg_refinement_iters

        if mpg_gate_temperature is not None:
            print(f"Overriding mpg_gate_temperature: {self.model.config.mpg_gate_temperature} -> {mpg_gate_temperature}")
            self.model.config.mpg_gate_temperature = mpg_gate_temperature
            if hasattr(self.model, 'mpg') and self.model.mpg is not None:
                self.model.mpg.gate_temperature = mpg_gate_temperature

        # Log final MPG configuration
        if hasattr(self.model.config, 'use_mpg') and self.model.config.use_mpg:
            print(f"MPG enabled: lambda={self.model.config.mpg_lambda}, "
                  f"projections={self.model.config.mpg_num_projections}, "
                  f"refinement_iters={self.model.config.mpg_refinement_iters}")

    def _apply_flow_matching_overrides(self, num_inference_timesteps: Optional[int]):
        """Apply flow matching parameter overrides."""
        if num_inference_timesteps is not None:
            print(f"Override num_inference_timesteps: "
                  f"{self.model.config.num_inference_timesteps} -> {num_inference_timesteps}")
            self.model.config.num_inference_timesteps = num_inference_timesteps
            self.model.num_inference_timesteps = num_inference_timesteps

    def _setup_rtc(self):
        """Setup Real-Time Chunking configuration."""
        if self.enable_rtc:
            # Enable inference-time prefix overwriting
            if not self.model.config.use_training_time_rtc:
                print(f"WARNING: Model was not trained with RTC, but RTC inference is enabled. May not work correctly.")

            self.model.config.use_inference_prefix_overwrite = True
            print(f"RTC enabled: Server returns full chunk_length={self.action_chunk_length} actions, "
                  f"trained_with_rtc={self.model.config.use_training_time_rtc}, "
                  f"prefix locking during denoising (client sends prev_chunk)")
        else:
            print("RTC disabled")

    def _setup_tokenizer(self):
        """Setup tokenizer and extract special token IDs."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        # Get special token IDs
        tokens = [
            '<|im_start|>', '<|im_end|>', '<img>', '</img>',
            '<|state_start|>', '<|state_end|>'
        ]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        (self.bos_token_id, self.eos_token_id, self.start_of_image,
         self.end_of_image, self.start_of_state, self.end_of_state) = token_ids
        
        newline_encoded = self.tokenizer.encode('\n')
        assert len(newline_encoded) == 1, "Newline should be single token"
        self.newline_token_id = newline_encoded[0]
   
    def _load_metadata(self, checkpoint_dir: Path):
        """
        Load dataset metadata with hierarchical stats and variant support.

        Args:
            checkpoint_dir: Checkpoint directory path (self-contained directory with metadata files)
        """
        metadata_filename = f"{self.dataset_name}_metadata.json"
        metadata_path = checkpoint_dir / metadata_filename

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found at {metadata_path}. "
                "This file is required for normalization statistics."
            )

        print(f"[Metadata] Loading from: {metadata_path}")

        with open(metadata_path, "r") as f:
            all_metadatas = json.load(f)

        metadata_dict = all_metadatas.get(self.dataset_name)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for dataset: '{self.dataset_name}' in {metadata_path}"
            )

        # Hierarchical Stats and Variant Selection 
        variants_key = f"{self.dataset_name}_variants"
        self.stats_level = "unknown"
        self.stats_source = "default"

        if variants_key in all_metadatas:
            """Select appropriate metadata variant."""

            available_variants = list(all_metadatas[variants_key].keys())
            print(f"\n[Metadata Variants] Available: {available_variants}")

            # Separate Level 1 (task) and Level 2 (embodiment) variants for display
            task_variants = [v for v in available_variants
                           if all_metadatas[variants_key][v].get('stats_level') == 'task']
            embodiment_variants = [v for v in available_variants
                                  if all_metadatas[variants_key][v].get('stats_level') == 'embodiment']

            if task_variants:
                print(f"  Level 1 (task): {task_variants}")
            if embodiment_variants:
                print(f"  Level 2 (embodiment): {embodiment_variants}")

            # Select metadata based on user preference
            if self.metadata_variant == "merged":
                # User requested merged statistics (use default top-level)
                print(f"  ✓ Using merged statistics from all variants")
                self.stats_level = metadata_dict.get('stats_level', 'merged')
                self.stats_source = "merged"

            elif self.metadata_variant and self.metadata_variant != "merged":
                # User requested specific variant
                if self.metadata_variant in all_metadatas[variants_key]:
                    metadata_dict = all_metadatas[variants_key][self.metadata_variant]
                    self.stats_level = metadata_dict.get('stats_level', 'unknown')
                    self.stats_source = f"variant:{self.metadata_variant}"
                    print(f"  ✓ Using variant: '{self.metadata_variant}' (level: {self.stats_level})")
                else:
                    print(f"  ⚠ Requested variant '{self.metadata_variant}' not found, using default")
                    # Fall through to auto-select

            if not self.metadata_variant or (self.metadata_variant and self.metadata_variant not in all_metadatas[variants_key] and self.metadata_variant != "merged"):
                # Auto-select (default behavior)
                # Priority: task-specific > embodiment-merged > default
                if self.stats_selection_mode == "task" and task_variants:
                    first_task = task_variants[0]
                    metadata_dict = all_metadatas[variants_key][first_task]
                    self.stats_level = "task"
                    self.stats_source = f"auto:task:{first_task}"
                    print(f"  → Auto-selected (task): '{first_task}'")

                elif self.stats_selection_mode == "embodiment" and embodiment_variants:
                    first_emb = embodiment_variants[0]
                    metadata_dict = all_metadatas[variants_key][first_emb]
                    self.stats_level = "embodiment"
                    self.stats_source = f"auto:embodiment:{first_emb}"
                    print(f"  → Auto-selected (embodiment): '{first_emb}'")

                elif self.stats_selection_mode == "auto":
                    # Default: use first available variant (usually task-specific)
                    first_variant = available_variants[0]
                    metadata_dict = all_metadatas[variants_key][first_variant]
                    self.stats_level = metadata_dict.get('stats_level', 'auto')
                    self.stats_source = f"auto:{first_variant}"
                    print(f"  → Auto-selected: '{first_variant}' (level: {self.stats_level})")

                else:
                    # Use default top-level
                    self.stats_level = metadata_dict.get('stats_level', 'default')
                    self.stats_source = "default"
                    print(f"  → Using default top-level metadata")

        else:
            # Legacy format - no variants
            if "statistics" not in metadata_dict and isinstance(metadata_dict, dict):
                print(f"Warning: Nested metadata detected, Keys: {list(metadata_dict.keys())}")
                first_key = next(iter(metadata_dict))
                print(f"Unalcking key: {first_key}")
                metadata_dict = metadata_dict[first_key]
                self.stats_source = f"legacy:{first_key}"
            else:
                self.stats_source = "legacy:direct"

        if "embodiment_tag" in metadata_dict:
            print(f"[Metadata] Embodiment tag: {metadata_dict['embodiment_tag']}")

        print(f"[Metadata] Stats level: {self.stats_level}, Source: {self.stats_source}")

        # Validate and set metadata
        metadata = DatasetMetadata.model_validate(metadata_dict)
        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

        print(f"✓ Metadata loaded for '{self.dataset_name}'")

    @torch.no_grad()
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get action with RTC support.
        
        For RTC mode, observations should contain:
        - 'prev_chunk': Previous action chunk (unified 200-dim, normalized)
        - 'inference_delay': Number of prefix actions
        
        Server returns full chunk; client handles composition/shifting.
        
        Args:
            observations: Environment observations
            
        Returns:
            Action dictionary with unnormalized actions
        """
        # ===== 1. Extract RTC Parameters (sent by client) =====
        prev_chunk = observations.pop('prev_chunk', None)
        inference_delay = observations.pop('inference_delay', 0)

        # Prepare prev_chunk if RTC enabled
        if self.enable_rtc and prev_chunk is not None:
            # Client should send unified 200-dim actions (normalized)
            # Convert to tensor and move to device
            if isinstance(prev_chunk, np.ndarray):
                prev_chunk = torch.from_numpy(prev_chunk).to(dtype=torch.float32, device=self.device)
            elif isinstance(prev_chunk, torch.Tensor):
                prev_chunk = prev_chunk.to(dtype=torch.float32, device=self.device)

            # Ensure 3D shape: (B, chunk_length, action_dim)
            if prev_chunk.ndim == 2:
                prev_chunk = prev_chunk.unsqueeze(0)

            # Verify dimensions
            expected_dim = self.model.unified_action_dim  # Note: on model, not config
            if prev_chunk.shape[-1] != expected_dim:
                raise ValueError(
                    f"prev_chunk dimension mismatch: got {prev_chunk.shape[-1]}, "
                    f"expected {expected_dim}. Client must send unified actions, not raw actions!"
                )

        # ===== 2. Prepare Observations =====
        is_batch = self._check_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        processed_obs = self._modality_transform(observations)

        # ===== 3. Prepare Packed Inputs =====
        packed_inputs = self._prepare_packed_inputs(
            processed_obs, observations[self.language_key]
        )
        for key, value in packed_inputs.items():
            if isinstance(value, torch.Tensor):
                packed_inputs[key] = value.to(self.device, non_blocking=True)

        if getattr(self.model, "mask_invalid_action_dims", False):
            action_valid_mask = torch.zeros(
                1,
                self.action_chunk_length,
                self.model.unified_action_dim,
                dtype=torch.float32,
                device=self.device,
            )
            for key, (start, end) in self.unified_mapping.items():
                if key.startswith('action.'):
                    action_valid_mask[:, :, start:end] = 1.0
            packed_inputs['action_valid_mask'] = action_valid_mask

            if self.enable_rtc and prev_chunk is not None:
                prev_chunk = prev_chunk * action_valid_mask

        # ===== 4. Add RTC Parameters to Model Input =====
        if self.enable_rtc and prev_chunk is not None and inference_delay > 0:
            packed_inputs['prev_chunk'] = prev_chunk
            packed_inputs['inference_delay'] = inference_delay

        # ===== 5. Generate Action Chunk =====
        model_pred = self.model.get_action(**packed_inputs)
        action_pred = model_pred["action_pred"]

        # Reshape to 3D for processing, from (B * chunk_length, action_dim) to (1, chunk_length, action_dim)
        action_chunk = action_pred.reshape(1, self.action_chunk_length, -1)  # (1, chunk_length, action_dim)

        # ===== 6. Unnormalize Actions =====
        action_pred_cpu = action_chunk.cpu().to(torch.float32)

        sliced_action_dict = {}
        for key, (start, end) in self.unified_mapping.items():
            if key.startswith('action.'):
                sliced_action_dict[key] = action_pred_cpu[:, :, start:end]

        unnormalized_action_dict = self._modality_transform.unapply(sliced_action_dict)

        if not is_batch:
            unnormalized_action_dict = squeeze_dict_values(unnormalized_action_dict)

        # ===== 7. Return Actions =====
        result = {k: v.tolist() for k, v in unnormalized_action_dict.items()}

        # For RTC mode, also return unified actions for next query
        if self.enable_rtc:
            if not is_batch:
                result['action_unified'] = action_pred_cpu.squeeze(0).tolist()  # (chunk_length, action_dim)
            else:
                result['action_unified'] = action_pred_cpu.tolist()  # (batch, chunk_length, action_dim)

        return result

    def _prepare_packed_inputs(self, processed_obs: Dict[str, Any], instructions: List[str]) -> Dict[str, Any]:
        """Prepare packed inputs for model."""
        first_state_key = next((k for k in processed_obs if k.startswith('state.')), None)
        N, _ = processed_obs[first_state_key].shape
        
        state_tensor = torch.zeros(N, self.model.unified_state_dim, dtype=torch.float32)

        # Fill state tensor from mapping
        for key, (start, end) in self.unified_mapping.items():
            if key.startswith('state.') and key in processed_obs:
                source_tensor = processed_obs[key]
                
                expected_dim = end - start
                if source_tensor.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Dimension mismatch for '{key}': "
                        f"expected {expected_dim}, got {source_tensor.shape[-1]}"
                    )
                
                state_tensor[:, start:end] = source_tensor

        # Prepare vision inputs
        all_frames = []
        for key in self.data_config.VIDEO_KEYS:
            video_np = processed_obs[key]
            for f_idx in range(video_np.shape[0]):
                all_frames.append(Image.fromarray(video_np[f_idx]))
            
        pixel_values = torch.stack([self.image_transform(img) for img in all_frames])

        # Build prompt components
        system_prompt = f"system\n{self.model.system_message}"
        system_ids = self.tokenizer.encode(system_prompt)

        user_prompt = "user\n"
        user_ids = self.tokenizer.encode(user_prompt)
        
        assistant_prompt = "assistant\n"
        assistant_ids = self.tokenizer.encode(assistant_prompt)

        inst_formatted = self.instruction_template.format(
            task_description=instructions[0], k=self.action_chunk_length
        )
        inst_ids = self.tokenizer.encode(inst_formatted)
        
        # CORE: Build packed sequence
        packed_text_ids, packed_text_indexes = [], []
        packed_vit_token_indexes, packed_state_indexes, packed_action_indexes = [], [], []
        packed_position_ids = []
        split_lens, attn_modes = [], []
        
        curr = 0
        curr_rope_id = 0

        # === Block 1: System Turn ===
        block_ids = [self.bos_token_id] + system_ids + [self.eos_token_id, self.newline_token_id]
        packed_text_ids.extend(block_ids)
        packed_text_indexes.extend(range(curr, curr + len(block_ids)))
        packed_position_ids.extend(range(curr_rope_id, curr_rope_id + len(block_ids)))
        curr += len(block_ids)
        curr_rope_id += len(block_ids)
        split_lens.append(len(block_ids))
        attn_modes.append("causal")

        # === Block 2: Content (user + vision + state + instruction) ===
        curr_split_start = curr

        # User prompt
        # is_bos: True, is_eos: False
        block_ids = [self.bos_token_id] + user_ids
        packed_text_ids.extend(block_ids)
        packed_text_indexes.extend(range(curr, curr + len(block_ids)))
        curr += len(block_ids)
        
        # Vision. Match the training data layout: each image is wrapped by its
        # own vision start/end tokens instead of sharing one span across views.
        for _ in all_frames:
            packed_text_ids.append(self.start_of_image)
            packed_text_indexes.append(curr)
            curr += 1

            packed_vit_token_indexes.extend(range(curr, curr + self.num_image_token))
            curr += self.num_image_token

            packed_text_ids.append(self.end_of_image)
            packed_text_indexes.append(curr)
            curr += 1
        
        # State
        packed_text_ids.extend([self.start_of_state])
        packed_text_indexes.append(curr)
        curr += 1

        packed_state_indexes.append(curr) # State occupies one token position
        curr += 1

        packed_text_ids.extend([self.end_of_state])
        packed_text_indexes.append(curr)
        curr += 1
        
        # Instruction
        # is_bos: False, is_eos: True
        block_ids = inst_ids + [self.eos_token_id, self.newline_token_id]
        packed_text_ids.extend(block_ids)
        packed_text_indexes.extend(range(curr, curr + len(block_ids)))
        curr += len(block_ids)
        
        # Finalize Content Block
        curr_split_len = curr - curr_split_start
        packed_position_ids.extend(range(curr_rope_id, curr_rope_id + curr_split_len))
        curr_rope_id += curr_split_len
        split_lens.append(curr_split_len)
        attn_modes.append("causal")

        # === # Block 3: Action (assistant + action placeholders) ===
        curr_split_start = curr

        # Assistant prompt
        # is_bos: True, is_eos: False
        block_ids = [self.bos_token_id] + assistant_ids
        packed_text_ids.extend(block_ids)
        packed_text_indexes.extend(range(curr, curr + len(block_ids)))
        curr += len(block_ids)
        
        # Action placeholders
        packed_action_indexes.extend(range(curr, curr + self.action_token_num))
        curr += self.action_token_num

        # Final EOS
        packed_text_ids.append(self.eos_token_id)
        packed_text_indexes.append(curr)
        curr += 1
        
        # Finalize Action Block
        curr_split_len = curr - curr_split_start
        packed_position_ids.extend(range(curr_rope_id, curr_rope_id + curr_split_len))
        # curr_rope_id += curr_split_len # No need to update rope_id for the last block
        split_lens.append(curr_split_len)
        attn_modes.append("causal")

        # Padding for block attention
        sequence_length = curr
        padding_len = (BLOCK_SIZE - (sequence_length % BLOCK_SIZE)) % BLOCK_SIZE
        sample_lens = [sequence_length]

        if padding_len > 0:
            split_lens.append(padding_len)
            attn_modes.append("causal")
            sample_lens.append(padding_len)

        # Assemble final dict
        return {
            "sequence_length": sequence_length,
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "sample_lens": sample_lens,
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "split_lens": split_lens,
            "attn_modes": attn_modes,
            "packed_vit_tokens": pixel_values.to(dtype=torch.bfloat16),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_action_indexes": torch.tensor(packed_action_indexes, dtype=torch.long),
            "padded_state": state_tensor.to(dtype=torch.bfloat16),
            "packed_state_indexes": torch.tensor(packed_state_indexes, dtype=torch.long),
            "embodiment_ids": torch.tensor([31], dtype=torch.long),
        }

    def _check_is_batched(self, obs: Dict[str, Any]) -> bool:
        first_val = next(iter(obs.values()))

        if isinstance(first_val, np.ndarray):
            return first_val.ndim > 1 and ("state" in next(iter(obs.keys())) or "video" in next(iter(obs.keys())))
        if isinstance(first_val, list):  # Applies to instruction lists
            return True
        return False
        
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.data_config.modality_config()
