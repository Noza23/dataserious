from typing import Any

import pytest

from dataserious import BaseConfig, BaseIntEnum, BaseStrEnum
from dataserious.fields import ConfigField


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
    name: TransformationFunction = TransformationFunction.OMIT_OUTLIERS
    args: dict[str, Any] = ConfigField(default_factory=dict)


class ModelConfig(BaseConfig):
    name: Model

    n_layers: int = ConfigField(description="Number of layers", searchable=True)
    n_heads: int = ConfigField(description="Number of heads", searchable=True)
    transformations: list[Transformation] = ConfigField(default_factory=list)


class Config(BaseConfig):
    experiment: str
    device: Devices = ConfigField(description="Device to run the experiment on")
    model: ModelConfig = ConfigField(default_factory=ModelConfig)
    seed: int = ConfigField(default=42, searchable=True)

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
