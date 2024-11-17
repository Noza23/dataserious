"""Utility functions for the dataserious package."""

import inspect


def isclasssubclass(obj: object, cls: type[object]) -> bool:
    """Check if the object is a subclass of the class, return `False` if not a class."""
    return inspect.isclass(obj) and issubclass(obj, cls)
