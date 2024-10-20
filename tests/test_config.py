import tempfile
import unittest
from dataclasses import field
from typing import Any

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


class TestConfig(unittest.TestCase):
    """Test the Config Class."""

    config = Config.from_yaml("tests/assets/config.yaml")

    with open("tests/assets/search_tree.yaml", "r") as f:
        search_tree = yaml.safe_load(f)

    def test_check_type(self):
        """Test type checking."""

        class NonSerialObj:
            pass

        self.assertTrue(check_type([1, "a"], list[int | str]))
        self.assertFalse(check_type([1, "a"], list[int]))
        self.assertTrue(check_type([{1: "a"}], list[dict[int, str]]))
        self.assertFalse(check_type([{1: "a"}], list[dict[str, str]]))
        self.assertTrue(check_type({"a": [1, 2], "b": ["a", "b"]}, JsonType))
        self.assertTrue(check_type(self.config.to_dict(), JsonSerializableDict))
        self.assertFalse(
            check_type(attr={"test": NonSerialObj()}, annot=JsonSerializableDict)
        )
        self.assertFalse(
            check_type(attr={"test": bytes("a", "utf-8")}, annot=JsonSerializableDict)
        )
        self.assertTrue(
            check_type(attr={"test": [[[{"a": [[1, 2]]}]]]}, annot=JsonSerializableDict)
        )

    def test_serialization(self):
        """Test Serialization."""
        self.assertIsInstance(self.config.to_dict(), dict)
        self.assertEqual(self.config, Config.from_dict(self.config.to_dict()))

    def test_json_serialization(self):
        """Test JSON Serialization."""
        with tempfile.NamedTemporaryFile() as f:
            self.config.to_json(f.name)
            ref = Config.from_json(f.name)
        self.assertEqual(self.config, ref)

    def test_yaml_serialization(self):
        """YAML Serialization."""
        with tempfile.NamedTemporaryFile() as f:
            self.config.to_yaml(f.name)
            ref = Config.from_yaml(f.name)
        self.assertEqual(self.config, ref)

    def test_replace(self):
        class SomeConfig(BaseConfig):
            x: int
            y: int

        c = SomeConfig(x=1, y=2)
        c_new = c.replace(x=3)
        assert c_new.x == 3 and c_new.y == 2
        assert id(c) != id(c_new)

    def test_apply_transformations(self):
        """Test the apply_transformations method."""
        data = 1
        assert self.config.apply_transformations(data) == 1

    def test_schema(self):
        """Test the schema generation."""
        schema = Config.to_schema()
        self.assertIsInstance(schema, dict)
        assert "<class 'str'>" in schema["experiment"]
        assert "{0, 1, 2}" in schema["device"]
        assert schema["seed"] == 42

    def test_enum(self):
        """Test the Enum Class Parsing."""
        assert isinstance(self.config.device, Devices)

    def test_type_error(self):
        """Test the type error."""
        cfg = self.config.to_dict()
        cfg["seed"] = "42"
        with self.assertRaises(TypeError):
            self.config.from_dict(cfg)
        cfg["seed"] = 42
        cfg["model"]["transformations"] = [2, 3]
        with self.assertRaises(TypeError):
            self.config.from_dict(cfg)

    def test_search_tree(self):
        """Test the search tree method."""
        search_tree = self.config.to_search_tree()
        check_type(search_tree, SearchTreeType)
        assert search_tree["model"]["n_layers"] == ["int"]

        configs_grid = self.config.get_configs_grid_from_path(
            search_tree_path="tests/assets/search_tree.yaml"
        )
        assert len(configs_grid) == 24

        configs_random = self.config.get_configs_random_from_path(
            search_tree_path="tests/assets/search_tree.yaml", n=10, seed=42
        )
        assert len(configs_random) == 10
