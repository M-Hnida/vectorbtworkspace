from setuptools import setup, find_packages

setup(
    name="vectorflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "vectorbt",
        "plotly",
        "scipy",
        "pandas_ta",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "vectorflow=vectorflow.cli:main",
        ],
    },
)
