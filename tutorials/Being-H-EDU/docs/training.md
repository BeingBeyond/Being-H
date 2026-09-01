# Training

The public SO101 training entrypoint is `examples/so101/train_so101_beingh.sh`.

## Required Inputs

- `PRETRAIN_MODEL`: InternVL/Qwen VLM checkpoint.
- `EXPERT_MODEL`: Qwen expert checkpoint.
- `RESUME_PATH`: Being-H checkpoint used as the starting point.
- `configs/posttrain/so101/so101_example.yaml` or a copied local YAML: dataset config pointing to local SO101 data.

For the public SO101 dataset, follow [so101_data_processing.md](so101_data_processing.md) first. It covers the Hugging Face download location, the `SO101_DATASET` path, and the required local action delta conversion.

## Dataset Config

```yaml
so101_posttrain:
  dataset_names:
  - so101.pick_cube_plate
  dataset_path_overrides:
    so101.pick_cube_plate: /local/path/to/Being-H-EDU_SO101/pick_cube_plate_trimmed
  data_config_names:
  - "so101"
  embodiment_tags:
  - "so101"
```

`dataset_names` must match a key in `configs/dataset_info.py`. The default key for the current SO101 example is `so101.pick_cube_plate`.

## Launch

```bash
PRETRAIN_MODEL=/path/to/InternVL3_5-2B \
EXPERT_MODEL=/path/to/Qwen3-0.6B \
RESUME_PATH=/path/to/being-h05-checkpoint \
bash examples/so101/train_so101_beingh.sh --gpus 4 --steps 20000
```

Use `--dataset-config configs/posttrain/so101/so101_local.yaml` if you generated a local config with the downloaded Hugging Face dataset path.

## Common Options

```bash
--gpus N
--steps N
--lr LR
--save-steps N
--action-chunk-length N
--dataset-config PATH
--run-name NAME
--output-root PATH
--log-dir PATH
```

Extra arguments after `--` are passed through to `BeingH/train/train.py`.

`BeingH/train/train.py` has no Weights & Biases integration, so the launcher's `--wandb*` options are
accepted but cannot enable logging; passing one exits with an error. Training metrics are written to
TensorBoard under `--log-dir`.

## Outputs

By default, checkpoints and logs are written to:

```text
outputs/<run-name>/
logs/tensorboard/
```
