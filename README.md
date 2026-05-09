# Reproducing Wogan et al. (2026), Version 1.0.0

This repository contains most of the code used in Wogan et al. (2026), titled "Toward Inferring the Surface Fluxes of Biosignature Gases on Rocky Exoplanets from Telescope Spectra". The article demonstrates how to infer surface gas fluxes given a spectrum of an exoplanet, for the purposes of figuring out if the planet hosts life.

This code cannot be feasibly run on a laptop, and will instead require a high performance computing cluster (multiple nodes).

## Step 1: Installation and Setup

If you do not have Anaconda on your system, install it here or in any way you prefer: [https://www.anaconda.com/download](https://www.anaconda.com/download). Next, run the following code to create a conda environment `flux`, then install `picaso` and an opacity file.

```sh
conda env create -f environment.yaml
conda activate flux

# Install picaso
wget https://github.com/natashabatalha/picaso/archive/4d5eded20c38d5e0189d49f643518a7b336a5768.zip
unzip 4d5eded20c38d5e0189d49f643518a7b336a5768.zip
cd picaso-4d5eded20c38d5e0189d49f643518a7b336a5768
python -m pip install . -v "numpy==1.26.4" "scipy==1.11.4"
# Get reference
cd ../
cp -r picaso-4d5eded20c38d5e0189d49f643518a7b336a5768/reference picasofiles/
rm -rf picaso-4d5eded20c38d5e0189d49f643518a7b336a5768
rm 4d5eded20c38d5e0189d49f643518a7b336a5768.zip

# Opacity DB
wget https://zenodo.org/records/17381172/files/opacities_photochem_0.1_250.0_R15000.db.zip
unzip opacities_photochem_0.1_250.0_R15000.db.zip
rm opacities_photochem_0.1_250.0_R15000.db.zip
mv opacities_photochem_0.1_250.0_R15000.db picasofiles/reference/opacities/

# setup
export picaso_refdata=$(pwd)"/picasofiles/reference/"
export PYSYN_CDBS="NOT_A_PATH_1234567890"
```

## Step 2: Input files

With the `flux` environment active, and the above environment variables set (for `picaso_refdata` and `PYSYN_CDBS`), run the following script to generate needed input files.

```sh
python input_files.py
```

## Step 3: Climate and Photochemical grid

Next, run the following scripts to generate the climate and photochemical grids. Again ensure you have the `flux` environment active, and the above environment variables set. This step will require a high performance computer outfitted with MPI (i.e., access to >100 cores). In the commands below, you should replace `NUMBER_OF_CORES` with however cores you want to distribute the calculation across (e.g., 10 nodes w/ 40 cores/node, then `NUMBER_OF_CORES` should be replaced with 400).

```sh
mpiexec -n NUMBER_OF_CORES python climate_grid.py
mpiexec -n NUMBER_OF_CORES python photochemical_grid.py
```

## Step 4: Run the retrievals

Next, run any retrievals with the following. Again you will need a HPC with MPI.

```sh
mpiexec -n NUMBER_OF_CORES python retrieval_run.py
```

## Step 5: Figures

Finally, make some of the key figures in the paper with.

```sh
python figures.py
```