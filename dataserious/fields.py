"""Fields for dataserious configs."""

from dataclasses import field
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def _extend_signature(func: Callable[P, T]):
    def _ConfigField(
        searchable: bool = False,
        description: str = "",
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        metadata = kwargs.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["searchable"] = searchable
            metadata["description"] = description
        return field(**kwargs)  # type: ignore

    return _ConfigField


# ConfigField is a constructor for fields of a BaseConfig class extending
# dataclasses.field with some additional arguments.
ConfigField = _extend_signature(field)
