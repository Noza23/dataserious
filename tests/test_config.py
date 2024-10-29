import tempfile
from dataclasses import field
from typing import Any

import pytest
import yaml

from dataserious import BaseConfig, BaseIntEnum, BaseStrEnum
from dataserious.base import JsonSerializableDict, JsonType, SearchTreeType, check_type


class Devices(BaseIntEnum):
    CPU = 0
    GPU = 1
    TPU = 2


class Model(BaseStrEnum):
    GPT = "GPT"
    BERT = "BERT"
    BERTRetrieval = "BERTRetrieval"


class TransformationFunction(BaseStrEnum):
    OMIT_OUTLIERS = "OMIT_OUTLIERS"
    NORMALIZE = "NORMALIZE"

    def apply(self, data: Any, args: dict[str, Any]) -> Any:
        return data


class Transformation(BaseConfig):
    name: TransformationFunction = field(
        metadata={"description": "Transformation name"}
    )
    args: dict[str, Any] = field(
        default_factory=dict, metadata={"description": "Transformation arguments"}
    )


class ModelConfig(BaseConfig):
    name: Model = field(metadata={"description": "Model name"})
    n_layers: int = field(
        metadata={"description": "Number of layers", "searchable": True}
    )
    n_heads: int = field(
        metadata={"description": "Number of heads", "searchable": True}
    )
    transformations: list[Transformation] = field(
        default_factory=list, metadata={"description": "Transformations to apply"}
    )


class Config(BaseConfig):
    experiment: str = field(metadata={"description": "Experiment name"})
    device: Devices = field(metadata={"description": "Device to use"})
    model: ModelConfig = field(
        default_factory=ModelConfig, metadata={"description": "Model configuration"}
    )
    seed: int = field(
        default=42, metadata={"description": "Random seed", "searchable": True}
    )

    def apply_transformations(self, data: Any) -> Any:
        for transformation in self.model.transformations:
            data = transformation.name.apply(data, transformation.args)
        return data


@pytest.fixture
def config():
    return Config.from_yaml("tests/assets/config.yaml")


@pytest.fixture
def search_tree():
    with open("tests/assets/search_tree.yaml", "r") as f:
        return yaml.safe_load(f)


def test_check_type(config):
    """Test type checking."""

    assert check_type([1, "a"], list[int | str])
    assert not check_type([1, "a"], list[int])
    assert check_type([{1: "a"}], list[dict[int, str]])
    assert not check_type([{1: "a"}], list[dict[str, str]])
    assert check_type({"a": [1, 2], "b": ["a", "b"]}, JsonType)
    assert check_type(config.to_dict(), JsonSerializableDict)

    class NonSerialObj:
        pass

    assert not check_type(attr={"test": NonSerialObj()}, annot=JsonSerializableDict)
    assert not check_type(
        attr={"test": bytes("a", "utf-8")}, annot=JsonSerializableDict
    )
    assert check_type(attr={"test": [[[{"a": [[1, 2]]}]]]}, annot=JsonSerializableDict)


def test_serialization(config):
    """Test Serialization."""
    assert isinstance(config.to_dict(), dict)
    assert config == Config.from_dict(config.to_dict())


def test_json_serialization(config):
    """Test JSON Serialization."""
    with tempfile.NamedTemporaryFile() as f:
        config.to_json(f.name)
        ref = Config.from_json(f.name)
    assert config == ref


def test_yaml_serialization(config):
    """YAML Serialization."""
    with tempfile.NamedTemporaryFile() as f:
        config.to_yaml(f.name)
        ref = Config.from_yaml(f.name)
    assert config == ref


def test_replace():
    class SomeConfig(BaseConfig):
        x: int
        y: int

    c = SomeConfig(x=1, y=2)
    c_new = c.replace(x=3)
    assert c_new.x == 3 and c_new.y == 2
    assert id(c) != id(c_new)


def test_apply_transformations(config):
    """Test the apply_transformations method."""
    assert config.apply_transformations(1) == 1


def test_schema():
    """Test the schema generation."""
    schema = Config.to_schema()
    assert isinstance(schema, dict)
    assert "<class 'str'>" in schema["experiment"]
    assert "{0, 1, 2}" in schema["device"]
    assert schema["seed"] == 42


def test_enum(config):
    """Test the Enum Class Parsing."""
    assert isinstance(config.device, Devices)


def test_type_error(config):
    """Test the type error."""
    cfg = config.to_dict()
    cfg["seed"] = "42"
    with pytest.raises(TypeError):
        config.from_dict(cfg)
    cfg["seed"] = 42
    cfg["model"]["transformations"] = [2, 3]
    with pytest.raises(TypeError):
        config.from_dict(cfg)


def test_search_tree(config):
    """Test the search tree method."""
    search_tree = config.to_search_tree()
    assert check_type(search_tree, SearchTreeType)
    assert search_tree["model"]["n_layers"] == ["int"]

    configs_grid = config.get_configs_grid_from_path(
        search_tree_path="tests/assets/search_tree.yaml"
    )
    assert len(configs_grid) == 24

    configs_random = config.get_configs_random_from_path(
        search_tree_path="tests/assets/search_tree.yaml", n=10, seed=42
    )
    assert len(configs_random) == 10
