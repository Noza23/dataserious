"""Utility functions for the dataserious package."""

import inspect
from enum import EnumMeta
from typing import Any, Dict, ForwardRef, List, Literal, Set, get_args, get_origin

from dataserious.types import *  # noqa
from dataserious.types import Annotation, GenericAliasTypes, UnionTypes


def check_type(attr, annot: Annotation) -> bool:
    """Check the type of the attribute recursively.

    Args:
        attr (Any): Attribute to be checked.
        annot (Annotation): Type hint to be checked against the attribute.

    Returns:
        bool: True if the attribute matches the type hint, False otherwise.

    Examples::
    >>> check_type("abc", int)
    False
    >>> check_type([1, "a"], list[int | str])
    True
    >>> check_type({'a': [1, 2], 'b': ["a", "b"]}, JsonType)
    True
    >>> check_type([BaseConfig()], list[BaseConfig])
    True

    """
    if annot in (Any, object):
        return True

    if isinstance(annot, UnionTypes):
        return any([check_type(attr, t) for t in get_args(annot)])

    if isinstance(annot, GenericAliasTypes):
        origin = get_origin(annot)
        args = get_args(annot)
        if isclasssubclass(origin, (List, Set)):
            return isinstance(attr, (list, set)) and all(
                [check_type(element, args[0]) for element in attr]
            )
        elif isclasssubclass(origin, Dict):
            return isinstance(attr, dict) and all(
                [
                    check_type(key, args[0]) and check_type(value, args[1])
                    for key, value in attr.items()
                ]
            )
        elif origin is Literal:
            return attr in args
        else:
            return isinstance(attr, origin)

    if isinstance(annot, ForwardRef):
        return check_type(attr, eval(annot.__forward_arg__))

    return isinstance(attr, annot)


def isclasssubclass(obj: object, cls: type[object] | tuple[type[object], ...]) -> bool:
    """Check if the object is a subclass of the class, return `False` if not a class."""
    return inspect.isclass(obj) and issubclass(obj, cls)


def type_to_view_string(annot: Annotation):
    """Convert the type hint to a view string."""
    if annot is None or annot is type(None):
        return str(None)
    if isinstance(annot, EnumMeta):
        return str(set(annot._value2member_map_.keys())).replace("'", "")
    if isinstance(annot, UnionTypes):
        return " | ".join([type_to_view_string(t) for t in get_args(annot)])
    if isinstance(annot, GenericAliasTypes):
        origin = get_origin(annot)
        args = get_args(annot)
        if isclasssubclass(origin, (List, Set)):
            return f"{origin.__name__}[{type_to_view_string(args[0])}]"
        if isclasssubclass(origin, Dict):
            return (
                f"dict[{type_to_view_string(args[0])}, {type_to_view_string(args[1])}]"
            )
        if origin is Literal:
            return " | ".join(a for a in args)
    if isinstance(annot, ForwardRef):
        return type_to_view_string(eval(annot.__forward_arg__))
    return annot.__name__
