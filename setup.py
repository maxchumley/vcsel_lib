"""Compatibility entry point for tools that still invoke ``setup.py``.

Project metadata, including the release version, lives in ``pyproject.toml``
so the two build paths cannot drift apart.
"""

from setuptools import setup


setup()
