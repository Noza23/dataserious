"""Json De/Serialization Utilities for config classes."""

import json


class ConfigJSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder Allowing Serialization for Enum and Custom Non-JSON Objects.

    Note:
        The Encoder extend the JSONEncoder class to provide custom serialization.
        objects by calling a `config_to_json` method if available.

    Raises:
        TypeError: If the object is not JSON serializable and does not have
        a `config_to_json` method implemented.

    """

    def default(self, obj):  # noqa: D102
        if getattr(obj, "config_to_json", None) and callable(obj.config_to_json):
            return obj.config_to_json()

        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable."
            f" You can implement a `.config_to_json()` method for "
            f"{obj.__class__.__name__} to make it JSON serializable. "
            "The implemented method should return a serializable object in context "
            "of `json.dumps` of base python `json` module."
            "But note that in loading the configuration back from file,"
            "customly serialized objects should be deserialized back to the "
            "original object by extending the `__post_init__` method "
            "of the BaseConfig class. This is usually not recommended."
        )


def _serialize(obj):
    """Serialize the object using the custom JSON Encoder."""
    return json.dumps(obj, cls=ConfigJSONEncoder)
