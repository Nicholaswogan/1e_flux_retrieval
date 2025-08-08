
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

def create_TRAPPIST1e_flux_hazmat():
    _ = stars.hazmat_spectrum(
        star_name='TRAPPIST-1',
        model='1a',
        outputfile='inputs/TRAPPIST1e_hazmat.txt',
        stellar_flux=planets.TRAPPIST1e.stellar_flux,
    )

def create_TRAPPIST1_flux_hazmat_picaso():

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
    np.savetxt('inputs/TRAPPIST1e_hazmat_picaso.txt', out.T)

def create_TRAPPIST1e_flux_muscles():

    url = 'https://zenodo.org/records/4556130/files/trappist-1_model_const_res_v07.ecsv'
    response = requests.get(url)
    with open('inputs/trappist-1_model_const_res_v07.ecsv','w') as f:
        f.write(response.content.decode())

    ###First, load the spectrum. Must return wavelength in nm and flux in mW/m^2/nm
    filename = 'inputs/trappist-1_model_const_res_v07.ecsv'
    spec_wav, spec_flux = np.genfromtxt(filename, skip_header=7, skip_footer=0,usecols=(0,1), unpack=True) #A, erg/s/cm2/A; fluxes are at Earth-star distance                
    wv = spec_wav/10 # convert from Angstroms to nm
    # (erg/cm2/s/Ang)*(1 W/1e7 erg)*(1e3 mW/1 W)*(1e4 cm^2/1 m^2)*(10 Ang/1 nm) = mW/m^2/nm
    F = spec_flux*(1/1e7)*(1e3/1)*(1e4/1)*(10/1) # convert from erg/cm2/s/Ang to mW/m^2/nm
    
    # Remove duplicated wavelengths
    wv, inds = np.unique(wv, return_index=True)
    F = F[inds]
    
    # Rescale the spectrum so that it's total bolometric flux matches Teff
    Teff = stars.MUSCLES_STARS['TRAPPIST-1']['st_teff']
    factor = stars.stefan_boltzmann(Teff)/stars.energy_in_spectrum(wv, F)
    F *= factor
    
    # Rescale to planet
    F = stars.scale_spectrum_to_planet(wv, F, None, planets.TRAPPIST1e.stellar_flux)
    
    # Only consider needed resolution
    wv, F = stars.rebin_to_needed_resolution(wv, F)

    # Save the spectrum to a file, if desired
    stars.save_photochem_spectrum(wv, F, 'inputs/TRAPPIST1e_muscles.txt', scale_to_planet=False)

def main():
    create_zahnle_HNOC()
    create_TRAPPIST1e_flux_hazmat()
    create_TRAPPIST1_flux_hazmat_picaso()
    create_TRAPPIST1e_flux_muscles()
    create_climate_inputs()

    # _ = stars.solar_spectrum(
    #     outputfile='inputs/Sun4.0Ga.txt',
    #     age=4.0,
    #     stellar_flux=1370,
    #     scale_before_age=True
    # )
    # climate.settings_file_for_climate(
    #     filename='inputs/settings_climate_earth.yaml', 
    #     planet_mass=float(constants.M_earth.cgs.value), 
    #     planet_radius=float(constants.R_earth.cgs.value), 
    #     surface_albedo=0.1, 
    #     number_of_layers=50, 
    #     number_of_zenith_angles=4, 
    #     photon_scale_factor=1.0
    # )


if __name__ == '__main__':
    main()

