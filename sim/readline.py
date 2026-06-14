"""Local pytest capture shim for macOS/conda readline crashes.

Pytest imports ``readline`` during capture initialization. On this workstation's
conda Python, importing the native extension segfaults before tests collect.
Keeping this shim in the simulator project directory makes ``cd sim && uv run
pytest`` resolve a harmless local module instead. The file is outside ``src/``
and is not included in the package build.
"""
