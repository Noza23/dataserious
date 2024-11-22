import tempfile
from enum import Enum

import pytest

from dataserious import BaseConfig
from dataserious.base import JsonSerializableDict, JsonType, SearchTreeType, check_type


class NonSerialObj:
    pass


def test_check_type(config):
    """Test type checking."""

    assert check_type([1, "a"], list[int | str])
    assert not check_type([1, "a"], list[int])
    assert check_type([{1: "a"}], list[dict[int, str]])
    assert not check_type([{1: "a"}], list[dict[str, str]])
    assert check_type({"x": [1, 2], "y": ["a", "b"]}, JsonType)
    assert check_type(config.to_dict(), JsonSerializableDict)
    assert not check_type(attr={"x": NonSerialObj()}, annot=JsonSerializableDict)
    assert not check_type(attr={"x": bytes("a", "utf-8")}, annot=JsonSerializableDict)
    assert check_type(attr={"x": [[[{"a": [[1, 2]]}]]]}, annot=JsonSerializableDict)


def test_serialization(config):
    """Test Serialization."""
    assert isinstance(config.to_dict(), dict)
    assert config == config.__class__.from_dict(config.to_dict())


def test_json_serialization(config):
    """Test JSON Serialization."""
    with tempfile.NamedTemporaryFile() as f:
        config.to_json(f.name)
        ref = config.__class__.from_json(f.name)
    assert config == ref


def test_yaml_serialization(config):
    """YAML Serialization."""
    with tempfile.NamedTemporaryFile() as f:
        config.to_yaml(f.name)
        ref = config.__class__.from_yaml(f.name)
    assert config == ref


class SomeConfig(BaseConfig):
    x: int
    y: int


def test_replace():
    c = SomeConfig(x=1, y=2)
    c_new = c.replace(x=3)
    assert c_new.x == 3 and c_new.y == 2
    assert id(c) != id(c_new)


def test_apply_config_method(config):
    """Test the apply_transformations method."""
    assert config.apply_transformations(1) == 1


def test_schema(config):
    """Test the schema generation."""
    schema = config.__class__.to_schema()
    assert isinstance(schema, dict)
    assert "Literal | NoneType" in schema["experiment"]
    assert "{0, 1, 2}" in schema["device"]
    assert schema["seed"] == 42


def test_enum(config):
    """Test the Enum Class Parsing."""
    assert check_type(config.device, list[Enum])


def test_type_error(config):
    """Test the type error."""
    cfg = config.to_dict()
    cfg["seed"] = "42"
    with pytest.raises(TypeError):
        config.from_dict(cfg)
    cfg["seed"] = 42
    assert config == config.from_dict(cfg)


def test_search_tree(config, search_tree_path):
    """Test the search tree method."""
    search_tree = config.to_search_tree()
    assert check_type(search_tree, SearchTreeType)
    assert search_tree["model"]["n_layers"] == ["int"]

    configs_grid = config.get_configs_grid_from_path(search_tree_path)
    configs_grid_yield = [
        cfg for cfg in config.yield_configs_grid_from_path(search_tree_path)
    ]
    assert configs_grid == configs_grid_yield
    assert len(configs_grid) == 24

    configs_rand = config.get_configs_random_from_path(search_tree_path, n=10, seed=42)
    assert len(configs_rand) == 10
