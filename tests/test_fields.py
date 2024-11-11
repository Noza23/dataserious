from unittest.mock import patch

from dataserious import BaseConfig
from dataserious.base import MISSING
from dataserious.fields import ConfigField, get_attr_descriptions


class Coords(BaseConfig):
    x: int = ConfigField(searchable=True, description="xy")
    y: int
    """This is an attribute docstring for y."""
    z: str = ConfigField(False, description="yz")
    """This wont override the description in ConfigField."""
    ref: "Coords"
    """This is an multiline
    attribute docstring for ref.
    """


def test_config_field():
    assert Coords.__dataclass_fields__["x"].metadata.get("searchable") is True
    assert Coords.__dataclass_fields__["x"].metadata.get("description") == "xy"
    assert Coords.__dataclass_fields__["y"].metadata.get("searchable") is None
    assert Coords.__dataclass_fields__["y"].default is MISSING
    assert Coords.__dataclass_fields__["z"].metadata.get("searchable") is False
    assert Coords.__dataclass_fields__["z"].metadata.get("description") == "yz"
    assert Coords.__dataclass_fields__["ref"].metadata.get("searchable") is None


def test_attr_descriptions():
    Coords.__dataclass_fields__["x"].metadata["description"] == "xy"
    Coords.__dataclass_fields__["y"].metadata[
        "description"
    ] == "This is an attribute docstring for y."
    Coords.__dataclass_fields__["z"].metadata["description"] == "yz"
    Coords.__dataclass_fields__["ref"].metadata[
        "description"
    ] == "This is an multiline attribute docstring for ref."


def test_get_attr_descriptions_oserror():
    with patch("inspect.getsource", side_effect=OSError):
        assert get_attr_descriptions(object) == {}
