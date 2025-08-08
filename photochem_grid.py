import warnings
warnings.filterwarnings('ignore')

from gridutils import make_grid, GridInterpolator
import numpy as np
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import climate_grid
import utils

def get_gridvals():
    log10PCO2 = np.append([-9,-7],np.arange(-5,1.01,1))
    log10PO2 = np.append([-15,-11],np.arange(-7,1.01,2))
    log10PCO = np.append([-11],np.arange(-7,1.01,2))
    log10PH2 = np.append([-9],np.arange(-6,0.01,2))
    log10PCH4 = np.append([-11,-9],np.arange(-7,1.01,1))
    gridvals = (log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4)
    return gridvals

def make_climate_interpolators():
    gridvals = climate_grid.get_gridvals()
    g = GridInterpolator('results/climate_v2.h5',gridvals)
    PRESS = g.make_interpolator('P',logspace=True)
    TEMP = g.make_interpolator('T')
    MIX = {}
    species = ['H2O','CO2','N2','O2','CO','H2','CH4']
    for sp in species:
        MIX[sp] = g.make_interpolator(sp,logspace=True)
    return PRESS, TEMP, MIX, g

def make_photochemical_model(stellar_flux='inputs/TRAPPIST1e_hazmat.txt'):
    pc = utils.EvoAtmosphereRobust(
        'inputs/zahnle_HNOC.yaml',
        'inputs/settings.yaml',
        stellar_flux
    )

    # Default is 1 micron
    particle_radius = pc.var.particle_radius
    particle_radius[:,:] = 1.0e-4
    pc.var.particle_radius = particle_radius
    pc.update_vertical_grid(TOA_alt=pc.var.top_atmos)

    # Specifics
    pc.set_particle_radii({
        'CO2aer': 1.0e-2, # 100 microns
        'H2Oaer': 1.0e-2,
        'HCaer1': 1.0e-4, # 1 micron
        'HCaer2': 1.0e-4,
        'HCaer3': 1.0e-4,
    })
    pc.rdat.max_total_step = 10_000
    pc.rdat.verbose = False
    return pc

PRESS, TEMP, MIX, GRIDINTERPOLATOR = make_climate_interpolators()
PHOTOCHEMICAL_MODEL = make_photochemical_model()

def model(x):

    pc = PHOTOCHEMICAL_MODEL

    log10PN2 = 0.0
    log10Kzz = 5.0

    # Parameters
    species_var = ['CO2','O2','CO','H2','CH4']
    log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4 = x

    # Species that we will fix at surface
    species_bc = ['N2','CO2','O2','CO','H2','CH4']
    x_bc = np.array([log10PN2,log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4])

    # Rough surface P
    P_surf = np.sum([10.0**a for a in x_bc])
    
    # P-T-mix profile from climate model
    P = PRESS(x)[:]
    T = TEMP(x)[:]
    mix = {}
    for key in MIX:
        mix[key] = MIX[key](x)[:]
    # Lower the mix if outside the climate grid.
    for i,sp in enumerate(species_var):
        if x[i] < GRIDINTERPOLATOR.min_gridvals[i]:
            mix[sp] = np.ones(mix[sp].shape[0])*((10.0**x[i])/P_surf)
    
    # Normalize
    ftot = np.zeros(P.shape[0])
    for key in mix:
        ftot += mix[key]
    for key in mix:
        mix[key] /= ftot

    # Constant Kzz
    Kzz = np.ones(P.shape[0])*10.0**log10Kzz

    # Surface boundary conditions
    Pi = x_to_press(species_bc, x_bc) # dynes/cm^2

    pc.initialize_to_PT_bcs(P, T, Kzz, mix, Pi)
    converged = pc.find_steady_state_robust()

    result = make_result(x, pc, converged)

    return result

def x_to_press(species, x):
    Pi = {
        'H2O': 270.0e6
    }
    for i,key in enumerate(species):
        Pi[key] = 1e6*10.0**x[i]
    return Pi

def make_result(x, pc, converged):
    P = pc.wrk.pressure
    z = pc.var.z
    T = pc.var.temperature
    sol = pc.mole_fraction_dict()
    surf, top = pc.gas_fluxes()
    for key in surf:
        surf[key] = np.array(surf[key].astype(np.float32))
        top[key] = np.array(top[key].astype(np.float32))

    result = {}
    result['converged'] = np.array(converged)
    result['x'] = x.astype(np.float32)
    result['P'] = P.astype(np.float32)
    result['z'] = z.astype(np.float32)
    result['T'] = T.astype(np.float32)
    for i,sp in enumerate(pc.dat.species_names[:-2]):
        result[sp] = sol[sp].astype(np.float32)
    for key in surf:
        result[key+'_BOA_flux'] = surf[key]
        result[key+'_TOA_flux'] = top[key]
    result['max_time'] = np.array(float(pc.rdat.max_time))

    return result

if __name__ == "__main__":
    # mpiexec -n X python photochem_grid.py
    make_grid(
        model_func=model, 
        gridvals=get_gridvals(), 
        filename='results/photochem_v1.h5', 
        progress_filename='results/photochem_v1.log'
    )
    


