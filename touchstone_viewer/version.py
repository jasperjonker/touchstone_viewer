from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

try:
    from ._version import __version__
except ImportError:
    try:
        __version__ = package_version("touchstone-viewer")
    except PackageNotFoundError:
        __version__ = "0+unknown"

__all__ = ["__version__"]
