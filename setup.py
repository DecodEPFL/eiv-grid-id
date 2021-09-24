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
        "click>=8.0",
        "proplot==0.6.4",
        "pytest==4.6.9",
        "pytest-cov==2.8.1",
        "seaborn==0.11.1",
        "tikzplotlib==0.9.6",

        # "mlflow==1.13.1",

        # Running GPU operations require
        # "cupy-cuda113==9.2.0",  # use to be "pycuda==2020.1"

        # Running Jupyter notebooks may require
        # "cov-core==1.15.0",
        # "coverage==4.5.2",
        # "Flask==1.1.2",
        # "ipykernel==5.4.3",
        # "ipython==7.19.0",
        # "ipython-genutils==0.2.0",
        # "ipywidgets==7.6.3",
        # "jupyter==1.0.0",
        # "jupyter-client==6.1.11",
        # "jupyter-console==6.2.0",
        # "jupyter-core==4.7.0",
        # "jupyter-server==1.2.1",
        # "jupyterlab==3.0.4",
        # "jupyterlab-pygments==0.1.2",
        # "jupyterlab-server==2.1.2",
        # "jupyterlab-widgets==1.0.0",
        # "nbconvert==6.0.7",
        # "nbformat==5.0.8",
        # "Pygments==2.7.4"
    ],
    python_requires=">=3.8",
)
