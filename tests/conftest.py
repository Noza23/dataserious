from enum import Enum
from typing import Any, Literal

import pytest

from dataserious import BaseConfig
from dataserious.fields import ConfigField


class Devices(int, Enum):
    CPU = 0
    GPU = 1
    TPU = 2


class Model(str, Enum):
    GPT = "GPT"
    BERT = "BERT"
    BERTRetrieval = "BERTRetrieval"


class TransformationFunction(str, Enum):
    OMIT_OUTLIERS = "OMIT_OUTLIERS"
    NORMALIZE = "NORMALIZE"

    def apply(self, data: Any, args: dict[str, Any]) -> Any:
        return data


class Transformation(BaseConfig):
    name: TransformationFunction = TransformationFunction.OMIT_OUTLIERS
    """Transformation function to apply to the data."""
    args: dict[str, Any] = ConfigField(default_factory=dict)
    """Arguments for the transformation function."""


class ModelConfig(BaseConfig):
    name: Model

    n_layers: int = ConfigField(searchable=True)
    """Number of layers in the model."""
    n_heads: int = ConfigField(searchable=True)
    """Number of heads in the model."""
    transformations: list[Transformation] = ConfigField(default_factory=list)
    """Transformations to apply to the data."""


class Config(BaseConfig):
    experiment: Literal["exp1", "exp2"] | None
    device: list[Devices] | str
    """Device to run the experiment on."""
    model: ModelConfig
    """Model configuration."""
    seed: int = ConfigField(default=42, searchable=True)
    """Seed for reproducibility."""

    def apply_transformations(self, data: Any) -> Any:
        for transformation in self.model.transformations:
            data = transformation.name.apply(data, transformation.args)
        return data


@pytest.fixture
def config():
    return Config.from_yaml("tests/assets/config.yaml")


@pytest.fixture
def search_tree_path():
    return "tests/assets/search_tree.yaml"
