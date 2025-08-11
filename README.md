
```sh
conda create -n flux -c conda-forge -c bokeh photochem=0.6.7 numpy=1.24 mpi4py dill tqdm astropy matplotlib pandas pip xarray pathos bokeh=2.4.3 wget unzip tar pymultinest=2.12
conda activate flux

# Install picaso
pip install picaso==3.1.2 -v

# Get reference
wget https://github.com/natashabatalha/picaso/archive/4d90735.zip
unzip 4d90735.zip
cp -r picaso-4d907355da9e1dcca36cd053a93ef6112ce08807/reference picasofiles/
rm -rf picaso-4d907355da9e1dcca36cd053a93ef6112ce08807
rm 4d90735.zip

# Opacity DB
wget https://zenodo.org/records/14861730/files/opacities_0.3_15_R15000.db.tar.gz
tar -xvzf opacities_0.3_15_R15000.db.tar.gz
rm opacities_0.3_15_R15000.db.tar.gz
mv opacities_0.3_15_R15000.db picasofiles/reference/opacities/

# 
export picaso_refdata=$(pwd)"/picasofiles/reference/"
```


```sh
# Install pymultinest
pip install pymultinest==2.12 -v

# Compile multinest, and put somewhere useful
wget https://github.com/JohannesBuchner/MultiNest/archive/refs/tags/v3.10b.tar.gz
tar -xvzf v3.10b.tar.gz
cd MultiNest-3.10b/build
cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_Fortran_FLAGS="-std=legacy"
make
cp ../lib/libmultinest* ../../lib
cd ../..
rm -rf MultiNest-3.10b
rm v3.10b.tar.gz

export DYLD_LIBRARY_PATH=$(pwd)"/lib/"
export LD_LIBRARY_PATH=$(pwd)"/lib/"
```