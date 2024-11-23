
import os
from photochem.utils import stars
from photochem.utils import zahnle_rx_and_thermo_files
import requests
import tempfile
import numpy as np
from astropy.io import fits

def create_zahnle_HNOC():
    "Creates a reactions file with H, N, O, C species."
    if not os.path.isfile('inputs/zahnle_HNOC.yaml'):
        zahnle_rx_and_thermo_files(
            atoms_names=['H', 'N', 'O', 'C'],
            rxns_filename='inputs/zahnle_HNOC.yaml',
            thermo_filename=None
        )

def create_TRAPPIST1e_stellar_flux():
    "Creates TRAPPIST-1 spectrum at planet e"
    if not os.path.isfile('inputs/TRAPPIST1e_stellar_flux.txt'):
        _ = stars.hazmat_spectrum(
            star_name='TRAPPIST-1',
            model='1a',
            outputfile='inputs/TRAPPIST1e_stellar_flux.txt',
            stellar_flux=0.646*1361.0, # W/m^2 (Agol et al. 2021)
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

if __name__ == '__main__':
    main()

