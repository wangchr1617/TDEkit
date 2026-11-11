from .tde import generate_direction_table, generate_uniform_direction_table, save_direction_table_to_csv, read_direction_table_from_csv
from .tde import TDESearch
from importlib import import_module

__version__ = "0.1.2"
__author__ = "Changrui Wang <wangchr1617@gmail.com>, Yidong Li <liyidong2023@buaa.edu.cn>"
__license__ = "MIT"

__all__ = [
    "TDESearch", "generate_direction_table", "generate_uniform_direction_table",
    "save_direction_table_to_csv", "read_direction_table_from_csv", "ForceAnalyzer",
    "FarthestPointSample", "DescriptorAnalyzer",
]

_OPTIONAL_EXPORTS = {
    "ForceAnalyzer": "active",
    "FarthestPointSample": "fps",
    "DescriptorAnalyzer": "fps",
}


def __getattr__(name):
    module = _OPTIONAL_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
