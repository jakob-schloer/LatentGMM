# LatentGMM
Code for the paper Schloer et al., A multi-modal representation of El Niño Southern Oscillation Diversity (2023)

## Install

1. Create conda environment with packages specified in conda YAML-file
``conda create -f latgmmenv.yml``
2. Activate environment ``conda activate latgmmenv``
3. Run ``pip install -e . `` in the root directory to use this latgmm package
4. Use correof-package whereever you want in your environment by ``import latgmm``

## Data

Put your datafiles in the ``./data`` folder. The location of the files are: 


## Contribute and Guidelines

You are welcome to contribute. Please keep in mind the following guidelines:

- Datasets are stored in the `/data` folder (for large datasets store a link to the original location in the data folder)
- Outputs and plots are stored in `/output`
- Unittests are placed in `/test`. Please make sure that all unit tests start with `test_...`
- Please use relative paths in all scripts only
- Comment your code!!! Use the [google docstring standard](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).
- Please use the following linters:
	- pylint
