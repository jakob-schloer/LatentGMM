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

└── reanalysis
    ├── 6-hourly
    │   └── ERA5
    │       ├── era5_u10_anom_HF_1940_2021_5N5S_130E80W_1deg.nc
    │       └── era5_u10_anom_LF_1940_2021_5N5S_130E80W_1deg.nc
    └── monthly
        ├── CERA-20C
        │   ├── oceanvars_CERA20C_1x1.nc
        │   └── sst_cera20c_1901-2009_r1x1.nc
        ├── COBE
        │   └── sst_cobe2_month_1850-2019.nc
        ├── ERA5
        │   ├── 2m_temperature_era5_monthly_sp_1940-2022_2.5x2.5.nc
        │   ├── olr_era5_monthly_sp_1940-2022_2.5x2.5.nc
        │   └── sea_surface_temperature_era5_monthly_sp_1940-2022_1.0x1.0.nc
        ├── ERSSTv5
        │   └── sst_ersstv5_month_1854-present.nc
        ├── GODAS
        │   ├── oceanvars_GODAS_1x1.nc
        │   └── sst_godas_month_1980-present.nc
        ├── HadISST
        │   └── sst_hadisst_month_1870-present.nc
        ├── ORAS5
        │   └── oceanvars_ORAS5_1x1.nc
        └── SODA
            ├── oceanvars_SODA_1x1.nc
            └── sst_SODA_month_1980-2017.nc



## Contribute and Guidelines

You are welcome to contribute. Please keep in mind the following guidelines:

- Datasets are stored in the `/data` folder (for large datasets store a link to the original location in the data folder)
- Outputs and plots are stored in `/output`
- Unittests are placed in `/test`. Please make sure that all unit tests start with `test_...`
- Please use relative paths in all scripts only
- Comment your code!!! Use the [google docstring standard](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).
- Please use the following linters:
	- pylint
