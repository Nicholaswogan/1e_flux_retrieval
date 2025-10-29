
```sh
conda env create -f environment.yaml
conda activate flux

# build Photochem
wget https://github.com/nicholaswogan/photochem/archive/6775d3ef2075428a9cbdf909ead9b6f5d460000d.zip
unzip 6775d3ef2075428a9cbdf909ead9b6f5d460000d.zip
cd photochem-6775d3ef2075428a9cbdf909ead9b6f5d460000d
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
python -m pip install --no-deps --no-build-isolation . -v
cd ..
rm -rf photochem-6775d3ef2075428a9cbdf909ead9b6f5d460000d
rm 6775d3ef2075428a9cbdf909ead9b6f5d460000d.zip

# Install picaso
wget https://github.com/natashabatalha/picaso/archive/4d5eded20c38d5e0189d49f643518a7b336a5768.zip
unzip 4d5eded20c38d5e0189d49f643518a7b336a5768.zip
cd picaso-4d5eded20c38d5e0189d49f643518a7b336a5768
python -m pip install . -v
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