
import os
from photochem.utils import stars
from photochem.utils import zahnle_rx_and_thermo_files
from photochem.utils import climate
import requests
import tempfile
import numpy as np
from astropy.io import fits
from astropy import constants
import planets

def create_climate_inputs():
    # Species file
    climate.species_file_for_climate(
        filename='inputs/species_climate.yaml', 
        species=['H2O','CO2','N2','H2','CH4','CO','O2'], 
        condensates=['H2O','CO2','CH4']
    )

    # Settings file
    climate.settings_file_for_climate(
        filename='inputs/settings_climate.yaml', 
        planet_mass=float(planets.TRAPPIST1e.mass*constants.M_earth.cgs.value), 
        planet_radius=float(planets.TRAPPIST1e.radius*constants.R_earth.cgs.value), 
        surface_albedo=0.1, 
        number_of_layers=50, 
        number_of_zenith_angles=4, 
        photon_scale_factor=1.0
    )

def create_zahnle_HNOC():
    zahnle_rx_and_thermo_files(
        atoms_names=['H', 'N', 'O', 'C'],
        rxns_filename='inputs/zahnle_HNOC.yaml',
        thermo_filename=None
    )

def create_TRAPPIST1e_stellar_flux():
    _ = stars.hazmat_spectrum(
        star_name='TRAPPIST-1',
        model='1a',
        outputfile='inputs/TRAPPIST1e_hazmat.txt',
        stellar_flux=planets.TRAPPIST1e.stellar_flux,
    )

def create_TRAPPIST1_flux_picaso():

    if os.path.isfile('inputs/TRAPPIST1e_stellar_flux_picaso.txt'):
        return

    url = 'http://archive.stsci.edu/hlsps/hazmat/hlsp_hazmat_phoenix_synthspec_trappist-1_1a_v1_fullres.fits'
    response = requests.get(url)

    with tempfile.TemporaryFile() as f:
        f.write(response.content)
        data = fits.getdata(f)

    wv = data['wavelength']/10/1e3 # convert from Angstroms to um
    # (erg/cm2/s/Ang) = Flam
    F = data['flux_density']

    wv, inds = np.unique(wv, return_index=True)
    F = F[inds]

    out = np.empty((2,wv.shape[0]))
    out[0,:] = wv
    out[1,:] = F
    np.savetxt('inputs/TRAPPIST1e_stellar_flux_picaso.txt', out.T)

def main():
    create_zahnle_HNOC()
    create_TRAPPIST1e_stellar_flux()
    create_TRAPPIST1_flux_picaso()
    create_climate_inputs()

if __name__ == '__main__':
    main()

