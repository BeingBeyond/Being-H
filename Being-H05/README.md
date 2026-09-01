# Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization

<p align="center">
    <img src="assets/being-h05.png" width="300"/>
<p>

<div align="center">

[![Blog](https://img.shields.io/badge/Blog-Being--H05-green)](https://research.beingbeyond.com/being-h05)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/pdf/2601.12993)
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/BeingBeyond/being-h05)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

</div>

Being-H0.5 is BeingBeyond's flagship **VLA** model, scaling human-centric learning with a unified action space to enable robust cross-embodiment robot control.
This directory contains the released H0.5 codebase inside the unified `Being-H` repository layout.

<div align="center">
<video src="https://github.com/user-attachments/assets/36714389-e737-4b11-8dcf-9076cc9f1d69" controls>
</video>
</div>

## Model Checkpoints

Download models from Hugging Face [Model Collections](https://huggingface.co/collections/BeingBeyond/being-h05):

| Model Type | Model Name | Parameters | Description |
|------------|------------|------------|-------------|
| **VLA Pretrained** | [Being-H05-2B](https://huggingface.co/BeingBeyond/Being-H05-2B) | 2B | Base vision-language-action model (preview) |
| **VLA Specialist** | [Being-H05-2B_libero](https://huggingface.co/BeingBeyond/Being-H05-2B_libero) | 2B | Post-trained on LIBERO benchmark |
| **VLA Specialist** | [Being-H05-2B_robocasa](https://huggingface.co/BeingBeyond/Being-H05-2B_robocasa) | 2B | Post-trained on RoboCasa kitchen tasks |
| **VLA Generalist** | [Being-H05-2B_libero_robocasa](https://huggingface.co/BeingBeyond/Being-H05-2B_libero_robocasa) | 2B | Post-trained on both LIBERO and RoboCasa |

Note: the vision part is 224px by default.

## Quick Start

### Installation

```bash
git clone https://github.com/BeingBeyond/Being-H.git
cd Being-H/Being-H05
conda create -n beingh python=3.10
conda activate beingh
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Training

```bash
# Single-embodiment training (e.g., LIBERO)
bash scripts/train/train_libero_example.sh

# Cross-embodiment training (multiple robots)
bash scripts/train/train_cross_emb_example.sh
```

**Important for cross-embodiment training:** Enable `--save_merged_metadata True` to save hierarchical metadata for inference. See [docs/training.md](docs/training.md) for details.

### Inference

```python
from BeingH.inference.beingh_policy import BeingHPolicy

# Load a pre-trained policy
policy = BeingHPolicy(
    model_path="<path-to-checkpoint>",      # Path to Being-H checkpoint
    data_config_name="<config-name>",       # e.g., "libero_nonorm", "robocasa_human"
    dataset_name="<dataset-name>",          # For loading normalization stats
    embodiment_tag="<robot-tag>",           # Robot identifier
    instruction_template="<prompt>",        # Task instruction template
)

# Run inference
actions = policy.get_action(observations)
```

See [docs/inference.md](docs/inference.md) for the complete API reference.

## Supported Robots

Being-H currently provides example configurations for **LIBERO** and **RoboCasa** benchmarks. We will gradually release more pre-built configurations for additional robot platforms.

To add your own robot, refer to our example configurations and the [Unified Action Space](docs/unified_action_space.md) slot layout, then follow the guide in [Data Configuration](docs/data_configuration.md).

Don't see your robot? [Open an issue](https://github.com/BeingBeyond/Being-H/issues) with your robot specs and a data sample - we're happy to help add support.

## How It Works: Unified Action Space

Being-H uses a **200-dimensional unified action space** that maps different robots to a shared semantic representation. This is what enables cross-embodiment transfer.

**The key insight**: Similar robot components (e.g., end-effector position) always map to the same dimensions, regardless of the robot type. This allows knowledge to transfer between robots.

For most users, you don't need to understand the details - just use one of the pre-built configurations. For advanced users who want to add custom robots, see the complete documentation:

**[Unified Action Space Guide](docs/unified_action_space.md)** - Complete slot layout and configuration examples

## Cross-Embodiment Metadata

For cross-embodiment models, Being-H saves **metadata** during training that is essential for inference. This metadata contains normalization statistics for each task/embodiment.

When running inference on a cross-embodiment model, specify which metadata variant to use:

```python
policy = BeingHPolicy(
    model_path="<path-to-checkpoint>",
    dataset_name="uni_posttrain",              # Cross-embodiment dataset
    metadata_variant="<task-or-embodiment>",   # Select normalization stats
    stats_selection_mode="task",               # "task", "embodiment", or "auto"
    # ... other parameters
)
```

See [docs/inference.md](docs/inference.md#cross-embodiment-metadata) for details.

## Documentation

| Document | Description |
|----------|-------------|
| [Unified Action Space](docs/unified_action_space.md) | How cross-embodiment transfer works |
| [Data Configuration](docs/data_configuration.md) | Adding custom robots and datasets |
| [Training](docs/training.md) | Training parameters and scripts |
| [Inference](docs/inference.md) | BeingHPolicy API reference |
| [Evaluation](docs/evaluation.md) | LIBERO and RoboCasa benchmarks |

## TODO

The following features are planned for future implementation:

- [ ] Out-of-the-box real robot pretrained checkpoints
- [ ] Complete pretraining scripts and documentation
- [x] Complete post-training scripts for all benchmarks
- [x] Detailed training and data documentation
- [x] Benchmark evaluation scripts for all supported tasks

## Contributing and Building on Being-H

We encourage researchers and practitioners to leverage Being-H as a foundation for their own experiments and applications. Whether you're adapting Being-H to new robotic platforms, exploring novel manipulation tasks, or extending the model to new domains, our modular codebase is designed to support your innovations. We welcome contributions of all kinds - from bug fixes and documentation improvements to new features and model architectures. By building on Being-H together, we can advance the field of vision-language-action modeling and enable robots to perform more complex and diverse manipulation tasks. Join us in making robotic manipulation more capable, robust, and accessible to all.

## Acknowledgments

Being-H builds on the following excellent open-source projects:

- [InternVL](https://github.com/OpenGVLab/InternVL): Vision-Language model backbone
- [Bagel](https://github.com/ByteDance-Seed/Bagel): Training framework
- [Qwen](https://github.com/QwenLM/Qwen): Language model and MoE expert
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): Benchmark for lifelong robot learning
- [RoboCasa](https://github.com/robocasa/robocasa): Large-scale simulation benchmark for everyday tasks

We thank the authors for their contributions to the robotics and machine learning communities.

## License

Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.

SPDX-License-Identifier: Apache-2.0

## Citation

If you find our work useful, please consider citing us and give a star to our repository! 🌟🌟🌟

**Being-H0.5**

```bibtex
@article{beingbeyond2026beingh05,
  title={Being-H0. 5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization},
  author={Luo, Hao and Wang, Ye and Zhang, Wanpeng and Zheng, Sipeng and Xi, Ziheng and Xu, Chaoyi and Xu, Haiweng and Yuan, Haoqi and Zhang, Chi and Wang, Yiqing and others},
  journal={arXiv preprint arXiv:2601.12993},
  year={2026}
}
```
