import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eiv-grid-id",
    version="1.0.0",
    author="Jean-Sebastien Brouillon",
    author_email="jean-sebastien.brouillon@epfl.ch",
    description="Simulation and identification software package for distribution grids",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DecodEPFL/eiv-grid-id",
    project_urls={
        "EIV Grid ID": "https://github.com/DecodEPFL/eiv-grid-id",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6",
        "matplotlib>=3.3",
        "pandapower>=2.6",
        "pandas>=1.2",
        "tqdm>=4.56",
        "click>=8.0"
    ],
    python_requires=">=3.8",
)
# TODO: complete requirements list
