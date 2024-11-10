"""Fields for dataserious configs."""

import ast
import inspect
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
        kwargs.setdefault("metadata", {})
        if isinstance(kwargs["metadata"], dict):
            kwargs["metadata"]["searchable"] = searchable
            kwargs["metadata"]["description"] = description
        return field(**kwargs)  # type: ignore

    return _ConfigField


# ConfigField is a constructor for fields of a BaseConfig class extending
# dataclasses.field with some additional arguments.
ConfigField = _extend_signature(field)


class DescriptionVisitor(ast.NodeVisitor):
    """Extracts attribute docstrings as descriptions from a class definition."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.descriptions: dict[str, str] = {}
        self.target: str | None = None

    def visit(self, node):  # noqa: D102
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            self.target = node.target.id
            self.descriptions[self.target] = ""
        super().visit(node)

    def visit_Expr(self, node):  # noqa: D102
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            desc = inspect.cleandoc(node.value.value)
            if self.target is not None:
                self.descriptions[self.target] = desc.replace("\n", " ")


def get_attr_descriptions(cls: type[object]) -> dict[str, str]:
    """Goes through the class definition and extracts attribute docstrings.

    Args:
        cls: The class to extract the attribute docstrings from.

    Returns:
        A dictionary with the attribute names as keys and the docstrings as values.
    """
    try:
        source = inspect.getsource(cls)
    except OSError:
        return {}
    visitor = DescriptionVisitor()
    visitor.visit(ast.parse(source))
    return visitor.descriptions
