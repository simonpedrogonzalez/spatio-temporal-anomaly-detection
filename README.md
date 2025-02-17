# spatio-temporal-anomaly-detection

Spatio-temporal anomaly detection for Utah air pollution.

## Environment setup

### Using uv manager

uv is my favorite because:
- it covers all needs (managing python versions, virtual envs and dependency resolution, etc)
- it's ridiculous how fast it is
- the `.venv` where all packages are stored is created by default in the project directory which:
    - allows inspecting library code easily
    - keeps each project's dependencies explicitly isolated

1. uv installation https://docs.astral.sh/uv/getting-started/installation/#standalone-installer (tldr `pip install uv` using the base python)
2. uv python version installation: `uv python install 3.12` (recommended because is about 60% faster than 3.10 or previous versions). However we may downgrade if we need an old dependency futher down the line.
3. venv creation and dependency installation: cd to the project directory and run `uv sync`

Other things you may want to do with uv:
- `uv python list` to see all installed python versions
- `uv add <package>` to install a package in the current venv

### Using other tools

If you don't wanna use uv, I exported a requirements.txt file with the dependencies, which you can use to install dependencies.

## Running code

- Vscode: setup the python interpreter used for your debugger: open command pallete (ctrl+shift+p) and search for "Python: Select Interpreter", then select the one in the .venv directory, which should be something like `Python 3.12.8 ('.venv')`. Run and Debug as usual.
- Terminal: uv `run my_script.py`

## GIS data visualization
QGIS is a foss that can be used to visualize and analyze GIS data. Recommended for exploring the data.
https://qgis.org/download/#debian-ubuntu
