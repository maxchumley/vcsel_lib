from setuptools import setup

setup(
    name="vcsel-lib",          # distribution name (pip/conda sees this)
    version="0.2.0",
    description="Functions to Simulate VCSEL Array Dynamics",
    author="Max Chumley",
    py_modules=["vcsel_lib"],  # ← the single .py file
    install_requires=[
        "numpy",
        "sympy",
        "matplotlib",
        "IPython",
        "jupyter",
        "scipy",
        "tqdm",
        "latex"
    ],
)