# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""BeingH VLA Inference Server - Entry point for running the inference server."""

import argparse
import torch
import numpy as np
import random
import tyro
from typing import Optional

from .beingh_policy import BeingHPolicy, get_default_device
from .beingh_service import BeingHInferenceServer

def set_seed(seed: int):
    """Set seed for all random number generators to ensure reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"--- Random seed set to {seed} for reproducibility ---")

# --- 2. Define command line arguments using dataclass ---
from dataclasses import dataclass

@dataclass
class ServerArgs:
    """Command line arguments for the inference server."""
    model_path: str
    """Path to the trained model checkpoint (self-contained directory with model.safetensors and metadata)."""

    port: int = 5555
    """Port for the server to run on."""

    host: str = "0.0.0.0"
    """Host address for the server to bind to."""

    api_token: Optional[str] = None
    """Optional API token for authentication."""

    seed: int = 42
    """Random seed."""

    device: Optional[str] = None
    """Device to run on, e.g. cuda, xpu or cpu (default: CUDA, then XPU, then CPU)."""

    prompt_template: str = "long"
    prop_pos: str = "front"

    data_config_name: str = ""
    embodiment_tag: str = ""

    dataset_name: str = ""

    max_view_num: int = -1
    use_fixed_view: bool = False

    # MPG Parameter Overrides
    # =====================================================
    use_mpg: Optional[bool] = None
    """Override: Enable/disable MPG enhancement at inference."""

    mpg_lambda: Optional[float] = None
    """Override: MPG residual strength (e.g., 0.1)."""

    mpg_num_projections: Optional[int] = None
    """Override: Number of Sliced Wasserstein projections."""

    mpg_refinement_iters: Optional[int] = None
    """Override: MPG refinement iterations at inference."""

    mpg_gate_temperature: Optional[float] = None
    """Override: MPG gate temperature (higher = softer gating)."""

    # Flow Matching Parameter Override
    # =====================================================
    num_inference_timesteps: Optional[int] = None
    """Override: Number of flow matching denoising steps (default: use model config)."""

    # RTC (Real-Time Chunking) Parameter
    # =====================================================
    enable_rtc: bool = True
    """Enable Training-Time RTC support (requires model trained with RTC)."""

    # Metadata Variant Selection
    # =====================================================
    metadata_variant: Optional[str] = None
    """Metadata variant to use: None (auto), 'merged', or specific variant name like 'adamu_pick_simple' or 'PND_AdamU'"""

    stats_selection_mode: str = "auto"
    """Stats selection mode for hierarchical metadata: 'auto' (default), 'task', 'embodiment', 'total'"""


# --- 3. Main function ---
def main(args: ServerArgs):

    set_seed(args.seed)

    # Determine device
    device = args.device if args.device is not None else str(get_default_device())
    print(f"Running policy on device: {device}")

    if args.prompt_template == "short":
        instruction_template = "{task_description}"
    else:
        instruction_template = "According to the instruction '{task_description}', what's the micro-step actions in the next {k} steps?"

    # Initialize BeingHPolicy
    # Policy handles model loading, transform loading, metadata loading, etc.
    print("--- 2. Initializing BeingHPolicy ---")
    policy = BeingHPolicy(
        model_path=args.model_path,
        data_config_name=args.data_config_name,
        embodiment_tag=args.embodiment_tag,
        dataset_name=args.dataset_name,
        instruction_template=instruction_template,
        prop_pos=args.prop_pos,
        max_view_num=args.max_view_num,
        use_fixed_view=args.use_fixed_view,
        device=device,
        # MPG parameter overrides
        use_mpg=args.use_mpg,
        mpg_lambda=args.mpg_lambda,
        mpg_num_projections=args.mpg_num_projections,
        mpg_refinement_iters=args.mpg_refinement_iters,
        mpg_gate_temperature=args.mpg_gate_temperature,
        # Flow matching parameter override
        num_inference_timesteps=args.num_inference_timesteps,
        # RTC parameter
        enable_rtc=args.enable_rtc,
        # Metadata variant selection
        metadata_variant=args.metadata_variant,
        stats_selection_mode=args.stats_selection_mode,
    )

    # Create and run server
    # Server only needs a policy object that implements get_action method
    print(f"--- 3. Starting Inference Server on {args.host}:{args.port} ---")
    server = BeingHInferenceServer(
        policy=policy,
        port=args.port,
        host=args.host,
        api_token=args.api_token
    )
    server.run()

if __name__ == "__main__":
    # Use tyro to parse command line arguments
    args = tyro.cli(ServerArgs)
    main(args)
