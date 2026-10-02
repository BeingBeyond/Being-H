# Data Configuration

Being-H uses a three-layer configuration system to manage datasets and embodiments. Currently, we provide example configurations for **LIBERO** and **RoboCasa** - refer to these examples when adding your own robot.

## Configuration Layers

1. **YAML Config** (`configs/posttrain/*.yaml`): Specifies which datasets to use for training
2. **DataConfig Class** (`configs/data_config.py`): Defines how to parse and transform data
3. **Dataset Registry** (`configs/dataset_info.py`): Maps dataset names to file paths

See [Training Guide](training.md) for how these configs are used during training, and [Inference Guide](inference.md) for how metadata is loaded during inference.

## DataConfig Class

Each embodiment requires a `DataConfig` class that defines:

```python
class MyRobotDataConfig(BaseDataConfig):
    # Camera views available
    VIDEO_KEYS = ['video.front_cam', 'video.wrist_cam']
    VIDEO_SOURCE_COLUMNS = {
        'video.front_cam': 'observation.images.front',
        'video.wrist_cam': 'observation.images.wrist',
    }

    # State modalities
    STATE_KEYS = ['state.eef_position', 'state.eef_rotation', 'state.gripper']

    # Action modalities
    ACTION_KEYS = ['action.eef_position', 'action.eef_rotation', 'action.gripper']

    # Language instruction key
    LANGUAGE_KEYS = ['language.instruction']
```

### UNIFIED_MAPPING

Maps modalities to the 200-dimensional unified space (see [Unified Action Space](unified_action_space.md)):

```python
    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        'state.eef_position': (0, 3),
        'state.eef_rotation': (3, 6),
        'state.gripper': (18, 19),
        'action.eef_position': (0, 3),
        'action.eef_rotation': (3, 6),
        'action.gripper': (18, 19),
    }
```

### ModalityDef

Defines how to extract data from Parquet columns:

```python
    def define_modalities(self) -> Dict[str, ModalityDef]:
        return {
            'language.instruction': ModalityDef(
                source_column='task_index', start=0, end=0),
            'state.eef_position': ModalityDef(
                source_column='observation.state', start=0, end=3),
            'state.eef_rotation': ModalityDef(
                source_column='observation.state', start=3, end=6,
                rotation_type="axis_angle"),
            'state.gripper': ModalityDef(
                source_column='observation.state', start=6, end=7),
            'action.eef_position': ModalityDef(
                source_column='action', start=0, end=3, absolute=False),
            'action.eef_rotation': ModalityDef(
                source_column='action', start=3, end=6, absolute=False,
                rotation_type="axis_angle"),
            'action.gripper': ModalityDef(
                source_column='action', start=6, end=7),
        }
```

**ModalityDef Fields:**

| Field | Description |
|-------|-------------|
| `source_column` | Parquet column name |
| `start` | Start index in the column |
| `end` | End index (exclusive) |
| `absolute` | `True` for absolute values, `False` for deltas |
| `rotation_type` | `"axis_angle"`, `"quaternion"`, or `None` |
| `continuous` | `True` for continuous data (default) |

## Adding a Custom Robot

### Step 1: Create DataConfig Class

Add to `configs/data_config.py`:

```python
class MyRobotDataConfig(BaseDataConfig):
    VIDEO_KEYS = ['video.front_cam']
    VIDEO_SOURCE_COLUMNS = {'video.front_cam': 'observation.images.front'}
    STATE_KEYS = ['state.eef_position', 'state.eef_rotation', 'state.gripper']
    ACTION_KEYS = ['action.eef_position', 'action.eef_rotation', 'action.gripper']
    LANGUAGE_KEYS = ['language.instruction']

    UNIFIED_MAPPING = {
        'state.eef_position': (0, 3),
        'state.eef_rotation': (3, 6),
        'state.gripper': (18, 19),
        'action.eef_position': (0, 3),
        'action.eef_rotation': (3, 6),
        'action.gripper': (18, 19),
    }

    state_normalization_modes = {}
    action_normalization_modes = {'action.gripper': 'binary'}
```

### Step 2: Register in DATA_CONFIG_MAP

```python
DATA_CONFIG_MAP = {
    "libero_nonorm": LiberoNoNormDataConfig,
    "robocasa_human": RobocasaHumanDataConfig,
    "my_robot": MyRobotDataConfig,  # Add here
}
```

### Step 3: Add to Dataset Registry

Edit `configs/dataset_info.py`. `DATASET_REGISTRY` maps a dataset *group* name to the dataset class
used to read it, and `DATASET_INFO` lists the datasets in that group with their paths:

```python
DATASET_REGISTRY = {
    ...
    'my_robot_posttrain': LeRobotIterableDataset,
}

DATASET_INFO = {
    ...
    'my_robot_posttrain': {
        'my_robot_task': {
            'dataset_path': "/path/to/my_robot_data",
        },
    },
}
```

### Step 4: Create YAML Config

Create `configs/posttrain/my_robot/my_robot.yaml`. The top-level key is the group name from
`DATASET_REGISTRY`, and `dataset_names` / `data_config_names` / `embodiment_tags` are parallel lists
with one entry per dataset:

```yaml
my_robot_posttrain:
  dataset_names:
  - my_robot_task
  data_config_names:
  - "my_robot"
  embodiment_tags:
  - "new_embodiment"
  sampling_strategy: "step"   # 'step' or 'trajectory'
  video_backend: "torchvision_av"
  vit_transform_args:
    type: "beingh"
    normalize_type: "imagenet"
    use_color_jitter: true
  frame_step_size:
  - 1
  weight: 1
```

### Step 5: Verify Setup

Run a quick training test:

```bash
torchrun --nproc_per_node=1 BeingH/train/train.py \
    --resume_from /path/to/Being-H05-2B \
    --dataset_config_file configs/posttrain/my_robot/my_robot.yaml \
    --max_steps 10 \
    --output_dir /tmp/test_my_robot
```

## Common Pitfalls

1. **Dimension mismatch**: Ensure UNIFIED_MAPPING indices don't overlap
2. **Missing rotation_type**: Specify for rotation modalities
3. **Wrong source_column**: Must match Parquet column names exactly
4. **Binary normalization**: Use for gripper open/close actions

---

## Example Configurations

We provide example configurations for LIBERO and RoboCasa:

- **LIBERO**: `configs/data_config.py` → `LiberoNoNormDataConfig`
- **RoboCasa**: `configs/data_config.py` → `RobocasaHumanDataConfig`
- **Cross-embodiment YAML**: `configs/posttrain/cross-embodiment/libero_robocasa.yaml`

Refer to these examples when creating configurations for your own robot.

## Need Help?

If you're having trouble configuring your robot, [open an issue](https://github.com/BeingBeyond/Being-H/issues) with:
- Your robot's action/state dimensions
- A sample of your data format
- The slot mapping you're trying to use

We're happy to help add support for new robots.
