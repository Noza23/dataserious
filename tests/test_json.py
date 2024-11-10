import pytest

from dataserious.json import _serialize


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


def test_conifg_json_encoder():
    pytest.raises(TypeError, _serialize, Point(1, 2))

    def config_to_json(self):
        return {"x": self.x, "y": self.y}

    Point.config_to_json = config_to_json
    assert _serialize(Point(1, 2)) == '{"x": 1, "y": 2}'
