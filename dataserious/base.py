"""Dataclasses based Configuration Class with validation, serialization and search space generation."""

import dataclasses
import enum
import importlib.util
import inspect
import itertools
import json
import operator
import random
import sys
import typing
from functools import reduce
from json.encoder import JSONEncoder
from pathlib import Path
from types import GenericAlias, UnionType

YAML_AVAILABLE = importlib.util.find_spec("yaml")
if YAML_AVAILABLE:
    import yaml

YAML_SUFFIXES = (".yaml", ".yml")
JSON_SUFFIXES = (".json",)


JSON_TYPES = (int, float, str, bool, list, dict, type(None))

JsonPrimitive: typing.TypeAlias = int | float | str | bool | None
JsonArrayType: typing.TypeAlias = list[
    typing.Union[JsonPrimitive, "JsonObjectType", "JsonArrayType"]
]
JsonObjectType: typing.TypeAlias = dict[
    str, typing.Union[JsonPrimitive, JsonArrayType, "JsonObjectType"]
]
JsonType: typing.TypeAlias = JsonPrimitive | JsonArrayType | JsonObjectType

JsonSerializableDict: typing.TypeAlias = JsonObjectType

# Types used for defining and performing Random or Grid Search on BaseConfig
# Tree having list of possible configuration values as leafs
SearchLeafType: typing.TypeAlias = list[JsonType]
SearchTreeType: typing.TypeAlias = dict[
    str, typing.Union[SearchLeafType, "SearchTreeType"]
]

Annotation: typing.TypeAlias = (
    type
    | UnionType
    | GenericAlias
    | typing._UnionGenericAlias  # type: ignore[name-defined, attr-defined]
    | typing._GenericAlias  # type: ignore[name-defined, attr-defined]
    | typing.ForwardRef
)

UnionInstance = (UnionType, typing._UnionGenericAlias)  # type: ignore[name-defined, attr-defined]
GenericAliasInstance = (GenericAlias, typing._GenericAlias)  # type: ignore[name-defined, attr-defined]


class ConfigJSONEncoder(JSONEncoder):
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
            json_str = obj.config_to_json()
            if not isinstance(json_str, str):
                raise TypeError(
                    "`config_to_json` method must return a json string, "
                    f"got {type(json_str).__name__}"
                )
            return json_str

        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable."
            f" You can implement a `.config_to_json()` method for "
            f"{obj.__class__.__name__} to make it JSON serializable. "
            "But note that in loading the configuration back from file,"
            "customly serialized objects should be deserialized back to the "
            "original object by extending the `__post_init__` method "
            "of the BaseConfig class. This is usually not recommended."
        )


def _serialize(obj):
    """Serialize the object using the custom JSON Encoder."""
    return json.dumps(obj, cls=ConfigJSONEncoder).strip('"')


class CustomEnumMeta(enum.EnumMeta):
    """Custom Enum Meta Class for Better Error Handling."""

    def __call__(cls, value, **kwargs):
        """Extend the __call__ method to raise a ValueError."""
        if value not in cls._value2member_map_:
            raise ValueError(
                f"{cls.__name__} must be one of {[*cls._value2member_map_.keys()]}"
            )
        return super().__call__(value, **kwargs)


class BaseEnum(enum.Enum, metaclass=CustomEnumMeta):
    """BaseEnum Class for implementing custom Enum classes."""


if sys.version_info >= (3, 11):
    dataclass_transform = typing.dataclass_transform

    class BaseStrEnum(enum.StrEnum, metaclass=CustomEnumMeta):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value

    class BaseIntEnum(enum.IntEnum, metaclass=CustomEnumMeta):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value.__str__()

else:
    from typing_extensions import dataclass_transform

    class BaseStrEnum(str, BaseEnum):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value

    class BaseIntEnum(int, BaseEnum):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value.__str__()


C = typing.TypeVar("C", bound="BaseConfig")


@dataclass_transform(kw_only_default=True)
class BaseConfig:
    """Base Configuration class extending python `dataclasses` module.

    Note:
        It builds on top of the dataclass module to provide additional
        functionality. Such as loading and saving configurations from
        YAML and JSON files, generating schemas and validating
        type Annotations post initialize in `__post_init__` dataclass method.

        - The configuration class should inherit from `BaseConfig`.
        - 'BaseEnum' 'BaseStrEnum' and 'BaseIntEnum' provide enhanced Enum classes.
        - Field annotations should be json serializable types to ensure proper I/O.

        - Specifing `{'description': 'some description'}` in the field metadata is
        recommended and will be very useful in generating configuration templates
        and type error messages.

    Example::

     class SomeConfig(BaseConfig):
        z: list[int] = field(metadata={'description': 'list of integers'})

    """

    def __init_subclass__(cls, **kwargs):
        """Subclass initialization."""
        kwargs.setdefault("kw_only", True)
        dataclasses.dataclass(cls, **kwargs)

    def __post_init__(self):
        """Run the post initialization checks and modifications.

        In the `BaseConfig` implementation it checks the type hints and parses
        Enum fields. For additional checks, custom parsing and modifications
        the method can be extended in the child classes as follows:

        Example::

         class SomeConfig(BaseConfig):
             x: int

         def __post_init__(self):
            super().__post_init__()
            assert x < 2 # Additional check
        """
        for field in self.fields():
            self._modify_field(
                **{field.name: parse(getattr(self, field.name), field.type)}
            )

            if not check_type(getattr(self, field.name), field.type):
                raise TypeError(
                    '\n'
                    f'| loc: {self.__class__.__name__}.{field.name}\n'
                    f'| expects: {field.type}\n'
                    f'| got: {type(getattr(self, field.name))}\n'
                    f'| description: {field.metadata.get("description")}\n'
                )

    def __contains__(self, item):
        """Check if the item is in the configuration."""
        return item in self.to_dict()

    def __getitem__(self, item: str):
        """Get the item from the configuration."""
        return getattr(self, item)

    def __str__(self):
        """Serialize Configuration to a json string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=3)

    @classmethod
    def fields(cls) -> tuple[dataclasses.Field, ...]:
        """Get the fields of the configuration class."""
        return dataclasses.fields(cls)  # type: ignore[arg-type]

    def get_by_path(self, path: list[str] | str):
        """Get the value in the configuration by point separated path or list of keys.

        Args:
            path (list[str] | str): Path to the value in the configuration.

        Example::

         class SomeOtherConfig(BaseConfig):
             z: int

         class SomeConfig(BaseConfig):
             x: SomeOtherConfig
             y: int

         c = SomeConfig(SomeOtherConfig(z=1), 2)
         assert c.get_by_path('x.z') == 1

        """
        if isinstance(path, str):
            path = path.split(".")
        return get_by_path(self, path)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Get the field names of the configuration class."""
        return tuple(f.name for f in cls.fields())

    @classmethod
    def get_all_subclasses(cls: type[C]) -> set[type[C]]:
        """Get all subclasses of given class recursively."""
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in c.get_all_subclasses()]
        )

    @classmethod
    def config_validate(cls: type[C], obj: str | dict | C) -> C:
        """Validate the object against the configuration class.

        Args:
            obj (str | dict | C): Object to be validated.

        """
        if isinstance(obj, cls):
            return cls.from_dict(obj.to_dict(), allow_extra=True)
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        if isinstance(obj, str):
            return cls.from_dict(json.loads(obj))
        raise ValueError(
            f"Object must be an instance of {cls.__name__}, dict, or json string, "
            f"got {type(obj).__name__}"
        )

    def _modify_field(self, /, **changes):
        """Replace the fields of the configuration in place. (Insecure)."""
        for name, value in changes.items():
            if name not in self.field_names():
                raise ValueError(
                    f"Invalid field name: `{name}` in `{self.__class__.__name__}`"
                )
            object.__setattr__(self, name, value)

    def replace(self: C, /, **changes) -> C:
        """Replace the fields and return a new object with replaced values (Secure).

        Example::

         class SomeConfig(BaseConfig):
            x: int
            y: int

         c = SomeConfig(1, 2)
         c_new = c.replace(x=3)
         assert c_new.x == 3 and c_new.y == 2
        """
        return dataclasses.replace(self, **changes)  # type: ignore[type-var]

    @classmethod
    def from_yaml(cls, path: Path | str):
        """Load configuration from a YAML file."""
        return cls.from_dict(_yaml_load(path))

    def to_yaml(self, path: Path | str):
        """Deserialize the configuration to a YAML file."""
        return _yaml_dump(self.to_dict(), path)

    @classmethod
    def from_json(cls, path: Path | str):
        """Deserialize the configuration from a JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: Path | str):
        """Serilize the configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=3)

    @classmethod
    def schema_to_json(cls, path: Path | str):
        """Dump the JSON schema for the configuration.

        Args:
            path (Path | str): Path to the file to write the `json` schema.

        Note:
            This method is useful for generating configuration templates.
            As it writes a schema for the configuration with type hints,
            desciptions and default values (if available) for each field.
            The user only needs to fill in the values in the schema
            and load the config using `from_json` classmethod.

        """
        with open(path, "w") as f:
            json.dump(cls.to_schema(), f, indent=3)

    @classmethod
    def schema_to_yaml(cls, path: Path | str):
        """Dump the YAML schema for the configuration.

        Args:
            path (Path | str): Path to the file to write the `yaml` schema.

        Note:
            This method is useful for generating configuration templates.
            As it writes a schema for the configuration with type hints,
            desciptions and default values (if available) for each field.
            The user only needs to fill in the values in the schema
            and load the config using `from_yaml` classmethod

        """
        return _yaml_dump(cls.to_schema(), path)

    @classmethod
    def from_dict(cls, config: dict[str, typing.Any], allow_extra: bool = False):
        """Parse the configuration from a python dictionary.

        Args:
            config (dict[str, typing.Any]): Configuration dictionary to be parsed.
            allow_extra (bool): Whether to allow extra fields in the configuration.

        """
        if not allow_extra:
            _check_extra_names(config, {f.name for f in cls.fields()})
        return cls(
            **{
                f.name: cls.handle_field_from(f.type, config.get(f.name))
                for f in cls.fields()
                if f.name in config
            }
        )

    def to_dict(self) -> JsonSerializableDict:
        """Convert the configuration to json serializable python dictionary."""
        return {
            f.name: self.handle_field_to(getattr(self, f.name)) for f in self.fields()
        }

    @classmethod
    def from_dir(cls, path: str | Path):
        """Load all configuration from a selected directory.

        Args:
            path (Path | str): Path to the directory containing configuration files.

        Note:
            In the directory `yaml` and `json` files can be mixed, the method will
            load all of them.

        """
        if not isinstance(path, Path):
            path = Path(path)
        patt = f'*[{"|".join(YAML_SUFFIXES + JSON_SUFFIXES)}]'
        return [cls.from_file(p) for p in path.glob(patt)]

    @classmethod
    def from_file(cls, path: str | Path):
        """Load configuration from a file."""
        if not isinstance(path, Path):
            path = Path(path)
        if path.suffix in YAML_SUFFIXES:
            return cls.from_yaml(path)
        if path.suffix in JSON_SUFFIXES:
            return cls.from_json(path)
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            f"Please use one of {YAML_SUFFIXES + JSON_SUFFIXES}"
        )

    @classmethod
    def to_schema(cls):
        """Generate a dictionary schema for the configuration.

        Note:
            The method generates a schema for the configuration class with
            field names as keys and as values we have either a default value
            (if available) a type hint with a description or a list of possible
            values for `Enum` fields.

        """
        return {f.name: _handle_schema(f) for f in cls.fields()}

    @staticmethod
    def handle_field_from(annot: Annotation, value) -> typing.Any:
        """Handle from serialization for the field."""
        return _handle_field_from(annot, value)

    @staticmethod
    def handle_field_to(value) -> typing.Any:
        """Handle to serialization for the field."""
        return _handle_field_to(value)

    def to_search_tree(self, prune_null: bool = True) -> SearchTreeType:
        """Generate a search tree template for the configuration class instance.

        Args:
            prune_null (bool): Whether to prune the null paths from the search tree.

        Returns:
            SearchTreeType: A tree structure of the configuration class where the leafes
            are `list` of the possible values to use in Grid Search or Random Search.

        Note:
            - Configurations which are not searchable are not included in the tree.
            - We mark the searchable fields with a metadata key 'searchable'.

        """
        search_tree = _to_search_tree(self)
        if prune_null:
            search_tree = prune_null_path(search_tree)
        return search_tree

    def search_tree_to_yaml(self, path: str | Path):
        """Dump the Search Tree Template to a YAML file."""
        return _yaml_dump(self.to_search_tree(), path)

    def search_tree_to_json(self, path: str | Path):
        """Dump the Search Tree Template to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_search_tree(), f, indent=3)

    def get_configs_grid(self, search_tree: SearchTreeType):
        """Generate a grid of configurations from the Search Tree.

        Args:
            search_tree (SearchTreeType): Search Tree that defines the search space.

        Returns:
            list[BaseConfig]: configs generated from the SearchTree in Grid Search.

        """
        return [
            self.from_dict(cfg)
            for cfg in _yield_config_search_space(
                config_tree=self.to_dict(), search_tree=search_tree
            )
        ]

    def yield_configs_grid(self, search_tree: SearchTreeType):
        """Yield a grid of configurations one-by-one from the Search Tree.

        Args:
            search_tree (SearchTreeType): Search Tree that defines the search space.

        Returns:
            Generator[BaseConfig]: configs generated from the SearchTree in Grid Search.

        """
        for cfg in _yield_config_search_space(
            config_tree=self.to_dict(), search_tree=search_tree
        ):
            yield self.from_dict(cfg)

    def get_configs_random(self, search_tree: SearchTreeType, n: int, seed: int):
        """Generate a configurations from the Search Tree for Random Search.

        Args:
            search_tree (SearchTreeType): Search Tree that defines the search space.
            n (int): Number of random configurations to generate.
            seed (int): Seed for the random number generator.

        Returns:
            list[BaseConfig]: configs generated from the SearchTree in Random Search.

        """
        return [
            self.from_dict(cfg)
            for cfg in _yield_config_search_space(
                config_tree=self.to_dict(),
                search_tree=search_tree,
                random_n=n,
                seed=seed,
            )
        ]

    def get_configs_grid_from_path(self, search_tree_path: str | Path):
        """Generate a grid of configurations from the Search Tree in Grid Search."""
        return [
            self.from_dict(cfg)
            for cfg in _yield_config_search_space(
                config_tree=self.to_dict(),
                search_tree=_yaml_load(search_tree_path),
            )
        ]

    def yield_configs_grid_from_path(self, search_tree_path: Path | str):
        """Generate a grid of configurations from the Search Tree in Grid Search."""
        for cfg in _yield_config_search_space(
            config_tree=self.to_dict(), search_tree=_yaml_load(search_tree_path)
        ):
            yield self.from_dict(cfg)

    def get_configs_random_from_path(
        self, search_tree_path: Path | str, n: int, seed: int
    ):
        """Generate random config from the Search Tree in Random Search."""
        return [
            self.from_dict(cfg)
            for cfg in _yield_config_search_space(
                config_tree=self.to_dict(),
                search_tree=_yaml_load(search_tree_path),
                random_n=n,
                seed=seed,
            )
        ]


def _to_search_tree(obj: BaseConfig) -> SearchTreeType:
    """Convert the configuration class to a search tree."""
    return {
        field.name: _handle_searchtree_schema(getattr(obj, field.name))
        for field in obj.fields()
        if isbaseconfig(field.type) or field.metadata.get("searchable")
    }


def _yield_config_search_space(
    config_tree: JsonSerializableDict,
    search_tree: SearchTreeType,
    random_n: int | None = None,
    seed: int | None = None,
):
    """Generate a grid of configs from the Search Tree.

    Args:
        config_tree (JsonSerializableDict): Configuration dictionary to be used as base.
        search_tree (SearchTreeType): Search Tree that defines the search space.
        random_n (int | None): Number of random configurations to generate.
        seed (int | None): Seed for the random number generator.

    Returns:
        Generator[JsonSerializableDict]: Generated config dict from the SearchTree.

    Note:
        - If `random_n` is given `seed` must also be provided.
        - If they are both `None` the function will generate a grid search space.

    """
    mapping = _get_grid_mapping(search_tree)
    product = itertools.product(*mapping.values())
    if random_n:  # Random Search might get slow for large search spaces.
        assert seed is not None, "Seed must be provided for Random Search."
        product_list = list(product)
        random.Random(seed).shuffle(product_list)
        product = product_list[:random_n]  # type: ignore[assignment]
    for values in product:
        for k, values_ in zip(mapping.keys(), values):
            set_config_value_by_path(config_tree, k, values_)
        yield config_tree


def _get_grid_mapping(search_tree: SearchTreeType):
    """Get the mapping of the search tree for grid search.

    Args:
        search_tree (SearchTreeType): Search Tree to be mapped.

    Returns:
        Mapping where the keys are point separeted string paths
        and the values are the tree leafes.

    """
    search_tree = prune_null_path(search_tree)
    tree_paths = get_all_string_path(search_tree)
    mapping = {k: get_search_tree_leaf_by_path(search_tree, k) for k in tree_paths}
    return mapping


def prune_null_path(tree: SearchTreeType) -> SearchTreeType:
    """Prune the null values from the search tree."""
    keys_to_delete = []

    for key, branch in tree.items():
        if isinstance(branch, dict):
            prune_null_path(branch)
        if not branch:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del tree[key]
    return tree


def get_search_tree_leaf_by_path(
    search_tree: SearchTreeType,
    path: list[str] | str,
) -> SearchLeafType:
    """Access a value in the search tree by path.

    Args:
        search_tree (SearchTreeType): Search Tree to access the value from.
        path (list[str] | str): Path to the value in the search tree.

    """
    if isinstance(path, str):
        path = path.split(".")
    return get_by_path(search_tree, path)


def set_config_value_by_path(
    config_dict: JsonSerializableDict,
    path: list[str] | str,
    value: JsonType,
) -> JsonSerializableDict:
    """Set the Configuration value by path in the configuration dictionary.

    Args:
        config_dict (JsonSerializableDict): Configuration dictionary to be modified.
        path (list[str] | str): Path to the value in the configuration.
        value (JsonType): Value to be set in the configuration.

    """
    if isinstance(path, str):
        path = path.split(".")
    return set_by_path(config_dict, path, value)


def get_all_string_path(tree: dict, sep: str = ".") -> list[str]:
    """Build the string path of the tree leafes recursively with a separator.

    Args:
        tree (dict): Tree to get the string path from.
        sep (str): Separator for the keys in the path.

    Returns:
        list[str]: List of string paths to the leafes of the tree.

    """
    keys = []
    for k, v in tree.items():
        if isinstance(v, dict):
            keys.extend([f"{k}{sep}{kk}" for kk in get_all_string_path(v)])
        else:
            keys.append(k)
    return keys


def get_by_path(tree: BaseConfig | dict, path: list) -> typing.Any:
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, path, tree)


def set_by_path(tree: dict, path: list, value):
    """Set a value in a pytree by path."""
    reduce(operator.getitem, path[:-1], tree)[path[-1]] = value
    return tree


def parse(attr, annot: Annotation) -> typing.Any:
    """Parse custom non-json serializable objects, like `Enum` or keep it as it is."""
    if inspect.isclass(annot) and issubclass(annot, enum.Enum):
        return annot(attr)

    if isinstance(annot, UnionInstance):
        for t in typing.get_args(annot):
            if issubclass(t, enum.Enum) and attr is not None:
                return t(attr)

    if isinstance(annot, GenericAliasInstance):
        if issubclass(typing.get_origin(annot), typing.List):
            if not isinstance(attr, typing.List):
                return
            return [parse(element, typing.get_args(annot)[0]) for element in attr]

        elif issubclass(typing.get_origin(annot), typing.Dict):
            if not isinstance(attr, typing.Dict):
                return
            return {
                parse(key, typing.get_args(annot)[0]): parse(
                    value, typing.get_args(annot)[1]
                )
                for key, value in attr.items()
            }

    if isinstance(annot, typing.ForwardRef):
        return parse(attr, eval(annot.__forward_arg__))
    return attr


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
    if annot == typing.Any:
        return True
    if isinstance(annot, UnionInstance):
        return any([check_type(attr, t) for t in typing.get_args(annot)])

    if isinstance(annot, GenericAliasInstance):
        if issubclass(typing.get_origin(annot), typing.List):
            if not isinstance(attr, typing.List):
                return False
            return all(
                [check_type(element, typing.get_args(annot)[0]) for element in attr]
            )
        elif issubclass(typing.get_origin(annot), typing.Dict):
            if not isinstance(attr, typing.Dict):
                return False
            return all(
                [
                    check_type(key, typing.get_args(annot)[0])
                    and check_type(value, typing.get_args(annot)[1])
                    for key, value in attr.items()
                ]
            )
        else:
            raise ValueError(
                f"Unsupported Generic Type: {typing.get_origin(annot)}. "
                "Contact the maintainer to add more type support."
            )
    if isinstance(annot, typing.ForwardRef):
        return check_type(attr, eval(annot.__forward_arg__))

    return isinstance(attr, annot)


def _handle_field_from(annot: Annotation, value) -> typing.Any:
    """Handle from serialization for the field."""
    if value is None:
        return None

    if inspect.isclass(annot) and issubclass(annot, BaseConfig):
        if isinstance(value, BaseConfig):
            return value
        if isinstance(value, dict):
            return annot.from_dict(value)

    if isinstance(annot, UnionInstance):
        for t in typing.get_args(annot):
            return _handle_field_from(t, value)

    if isinstance(annot, GenericAliasInstance):
        if issubclass(typing.get_origin(annot), typing.List):
            if not isinstance(value, typing.List):
                return
            return [
                _handle_field_from(typing.get_args(annot)[0], element)
                for element in value
            ]

        if issubclass(typing.get_origin(annot), typing.Dict):
            if not isinstance(value, typing.Dict):
                return
            return {
                key: _handle_field_from(typing.get_args(annot)[1], val)
                for key, val in value.items()
            }

    return value


def _handle_field_to(value) -> typing.Any:
    """Handle to serialization for the field."""
    if isinstance(value, BaseConfig):
        return value.to_dict()
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, list):
        return [_handle_field_to(element) for element in value]
    if isinstance(value, dict):
        return {
            _handle_field_to(key): _handle_field_to(val) for key, val in value.items()
        }
    if type(value) in JSON_TYPES:
        return value
    return _serialize(value)


def _handle_schema(field: dataclasses.Field):
    """Handle the schema for the field."""
    desc = field.metadata.get("description", "No description given")
    if inspect.isclass(field.type) and issubclass(field.type, BaseConfig):
        return field.type.to_schema()

    if isinstance(field.type, enum.EnumMeta):
        return (
            str(set(field.type._value2member_map_.keys())).replace("'", "")
            + f": {desc}"
        )

    if not isinstance(field.default, dataclasses._MISSING_TYPE):
        return field.default
    return f"{field.type}: {desc}"


def _handle_searchtree_schema(obj):
    """Handle the field for the search tree schema."""
    if isinstance(obj, BaseConfig):
        return obj.to_search_tree()
    if isinstance(obj, enum.Enum):
        return str(set(obj._value2member_map_.keys())).replace("'", "")
    return [type(obj).__name__]


def _check_extra_names(config: dict[str, typing.Any], field_names: set[str]):
    """Check for extra field names in the configuration.

    Args:
        config (dict[str, typing.Any]): Configuration dictionary to be checked.
        field_names (set[str]): Set of field names in the configuration class.

    Raises:
        ValueError: If there are extra fields in the configuration.

    """
    if extra_names := set(config.keys()) - field_names:
        raise ValueError(
            f"Extra fields in the configuration: {extra_names}. "
            f"Allowed fields are: {field_names}"
        )


def isbaseconfig(annot: Annotation) -> bool:
    """Check if the class is a subclass of BaseConfig."""
    return inspect.isclass(annot) and issubclass(annot, BaseConfig)


def _yaml_dump(obj, path: str | Path):
    """Dump the object to a YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError(
            'Please install "pyyaml" module to use the YAML serialization, '
            "otherwise use .to_json() method."
        )
    with open(path, "w") as f:
        yaml.dump(obj, f, sort_keys=False)


def _yaml_load(path: str | Path):
    """Load the object from a YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError(
            'Please install "pyyaml" module to use the YAML deserialization, '
            "otherwise use .from_json() method."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)
