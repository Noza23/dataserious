from dataserious import BaseConfig
from dataserious.base import MISSING
from dataserious.fields import ConfigField


class Coords(BaseConfig):
    x: int = ConfigField(searchable=True, description="xy")
    y: int
    z: str = ConfigField(False, description="xy")


def test_config_field():
    assert Coords.__dataclass_fields__["x"].metadata["searchable"] is True
    assert Coords.__dataclass_fields__["x"].metadata["description"] == "xy"
    assert Coords.__dataclass_fields__["y"].metadata.get("searchable") is None
    assert Coords.__dataclass_fields__["y"].metadata.get("description") is None
    assert Coords.__dataclass_fields__["y"].default is MISSING
    assert Coords.__dataclass_fields__["z"].metadata["searchable"] is False
    assert Coords.__dataclass_fields__["z"].metadata["description"] == "xy"
    # Searchable Field with Description
