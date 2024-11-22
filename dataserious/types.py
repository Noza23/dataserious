"""Types used in the dataserious package."""

import typing
from types import GenericAlias, UnionType
from typing import ForwardRef, TypeAlias, Union

UnionTypes = (UnionType, typing._UnionGenericAlias)  # type: ignore[name-defined, attr-defined]
GenericAliasTypes = (GenericAlias, typing._GenericAlias)  # type: ignore[name-defined, attr-defined]

Annotation: TypeAlias = (
    type
    | UnionType
    | GenericAlias
    | ForwardRef
    | typing._UnionGenericAlias  # type: ignore[name-defined, attr-defined]
    | typing._GenericAlias  # type: ignore[name-defined, attr-defined]
)

# Json Types
JsonPrimitive: TypeAlias = int | float | str | bool | None
JsonArrayType: TypeAlias = list[Union[JsonPrimitive, "JsonObjectType", "JsonArrayType"]]
JsonObjectType: TypeAlias = dict[
    str, Union[JsonPrimitive, JsonArrayType, "JsonObjectType"]
]
JsonType: TypeAlias = JsonPrimitive | JsonArrayType | JsonObjectType

JsonSerializableDict: TypeAlias = JsonObjectType


# Types used for defining and performing Random or Grid Search on BaseConfig
# Tree having list of possible configuration values as leafs
SearchLeafType: TypeAlias = list[JsonType]
SearchTreeType: TypeAlias = dict[str, Union[SearchLeafType, "SearchTreeType"]]
