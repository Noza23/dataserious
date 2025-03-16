# DataSerious

![Tests](https://github.com/Noza23/dataserious/actions/workflows/tests.yaml/badge.svg)
[![codecov](https://codecov.io/gh/Noza23/dataserious/graph/badge.svg?token=m9yHQyL0sQ)](https://codecov.io/gh/Noza23/dataserious)

`dataserious` is a Python package that enhances dataclasses with type validation, serialization, and search space generation. It builds on top of the standard `dataclasses` module to provide additional functionality for configuration management, it only has a single dependency on `pyyaml` for YAML support.

## Features

- **Type Validation**: Ensures that the attributes of the dataclass instances match their type annotations.
- **Serialization**: Supports serialization to and from JSON and YAML formats.
- **Search Space Generation**: Generates search trees for hyperparameter tuning in grid and random search.

## Installation

You can install `dataserious` with YAML support using pip.

```sh
pip install "dataserious[yaml] @ git+https://github.com/Noza23/dataserious.git"
```
## Usage

### Defining a Configuration Class

To define a configuration class, inherit from `BaseConfig` and use `ConfigField` for fields that require additional metadata.

```python
from dataserious import BaseConfig, ConfigField

class ModelConfig(BaseConfig):
    name: str
    """Name of the model."""
    n_layers: int = ConfigField(searchable=True, description="Number of layers in the model.")
    n_heads: int = ConfigField(searchable=True)
    """Number of heads in the model."""
```

### Creating Template Configurations
```python
ModelConfig.to_schema()
ModelConfig.schema_to_yaml("schema.yaml")
```
```yaml
name: 'str: Name of the model.'
n_layers: 'int: Number of layers in the model.'
n_heads: 'int: Number of heads in the model.'
```

### Loading and Saving Configurations

You can load and save configurations in JSON and YAML formats.

```python
config = ModelConfig(name="GPT", n_layers=12, n_heads=12)
config.to_yaml("config.yaml")
loaded_config = ModelConfig.from_yaml("config.yaml")
```

### Generating Search Trees

Generate search trees for hyperparameter tuning.

```python
print(config.to_search_tree())
config.search_tree_to_yaml("search_tree.yaml")
```

```yaml
n_layers:
- int
n_heads:
- int
```

### Serving Grid and Random Search Spaces

1. Filled out search tree *test.yaml*:
    ```yaml
    n_layers:
    - 12
    - 24
    - 36
    n_heads:
    - 12
    - 24
    - 36
    ```
2. Loading the search tree and generating search spaces:
    ```python
    configs = config.get_configs_grid_from_path("test.yaml")
    configs_random = config.get_configs_random_from_path("test.yaml", n=2, seed=42)
    ```

3. Resulting Configs:
    ```python
    [print(json.dumps(config.to_dict(), indent=4)) for config in configs]
    print("\nRandom configs:")
    [print(json.dumps(config.to_dict(), indent=4)) for config in configs_random]
    ```


### Validating Configurations

Ensure that the configurations match the expected types.

```python
try:
    config = ModelConfig(name="GPT", n_layers="twelve", n_heads=12)
except TypeError as e:
    print(e)
```

## Contributing

Contributions are welcome!
