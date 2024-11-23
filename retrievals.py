import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import ultranest
import pickle
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import grid_utils
import photochem_grid
import utils
import planets

# def get_gridvals():
#     log10PCO2 = np.append([-9,-8,-7,-6],np.arange(-5,1.01,0.5))
#     log10PO2 = np.append([-7, -6, -5, -4, -3, -2],np.arange(-1,1.01,0.5))
#     log10PCO = np.append([-7, -6, -5, -4, -3, -2],np.arange(-1,1.01,0.5))
#     log10PH2 = np.append([-6, -5, -4, -3, -2],np.arange(-1,0.01,0.5))
#     log10PCH4 = np.append([-11,-10,-9,-8, -7],np.arange(-6,1.01,0.5))
#     log10Pcloud = np.arange(-5,0.01,1)
#     gridvals = (log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4,log10Pcloud)
#     return gridvals

def make_interpolators():
    filename = 'results/photochem_v1.2.pkl'
    gridvals = photochem_grid.get_gridvals()
    g = grid_utils.GridInterpolator(filename, gridvals)

    pc = photochem_grid.PHOTOCHEMICAL_MODEL
    all_species = pc.dat.species_names[pc.dat.np:-2]
    MIX = {}
    for i,sp in enumerate(all_species):
        MIX[sp] = g.make_array_interpolator(sp,logspace=True)
    all_particles = pc.dat.species_names[:pc.dat.np]
    PARTICLES = {}
    for i,sp in enumerate(all_particles):
        PARTICLES[sp] = g.make_array_interpolator(sp,logspace=True)
    FLUXES = {}
    for i,sp in enumerate(all_species):
        FLUXES[sp] = g.make_value_interpolator('BOA flux',sp)

    PRESS = g.make_array_interpolator('P',logspace=True)
    ALT = g.make_array_interpolator('z')
    TEMP = g.make_array_interpolator('T')

    return PRESS, ALT, TEMP, MIX, PARTICLES, FLUXES

def make_picaso():
    filename_db = None
    M_planet = planets.TRAPPIST1e.mass
    R_planet = planets.TRAPPIST1e.radius
    R_star = planets.TRAPPIST1.radius
    star_kwargs = {
        'filename': 'inputs/TRAPPIST1e_stellar_flux_picaso.txt',
        'w_unit': 'um',
        'f_unit': 'FLAM'
    }
    opannection_kwargs = {'wave_range': [0.5, 5.4]}
    p = utils.Picaso(
        filename_db, 
        M_planet, 
        R_planet, 
        R_star,
        opannection_kwargs=opannection_kwargs,
        star_kwargs=star_kwargs
    )
    return p

def make_picaso_atm(x, species=None):
    atm = {}
    atm['pressure'] = PRESS(x)/1e6
    atm['temperature'] = TEMP(x)
    if species is None:
        species = list(MIX.keys())
    ftot = np.zeros(atm['pressure'].shape[0])
    for key in species:
        atm[key] = MIX[key](x)
        ftot += atm[key]
    for key in species:
        atm[key] /= ftot
    for key in atm:
        atm[key] = atm[key][::-1].copy()
    atm = pd.DataFrame(atm)
    return atm

# Over all models, these are species with concentrations above 1e-5 vmr
PICASO_SPECIES = [
    'C2H2', 'H2O2', 'C4H3', 'O2', 'H2CO', 
    'C2H3OH', 'OH', 'HCN', 'CH3O2', 'C2H4', 
    'CH2CO', 'H2O', 'CH3OH', 'C', 'N2D', 'H2', 
    'O3', 'CH3CHO', 'N', 'C4H2', 'O', 'N2', 'H', 
    'CH3', 'C2H6', 'CH4', 'C4H4', 'CO', 'C3H4', 'CO2'
]
PICASO = make_picaso()
PRESS, ALT, TEMP, MIX, PARTICLES, FLUXES = make_interpolators()
PARAM_NAMES = ['log10PCO2','log10PO2','log10PCO','log10PH2','log10PCH4','log10Pcloud','offset']

def model(y, wavl, **kwargs):
    p = PICASO

    log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4,log10Pcloud,offset = y

    # Atmosphere
    x = (log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4)
    atm = make_picaso_atm(x, PICASO_SPECIES)

    # Clouds. log10Pcloud is the top of the cloud
    p.clouds_reset()
    log10Psurf = np.log10(atm['pressure'].to_numpy()[-1])
    dlog10Pcloud = log10Psurf - log10Pcloud
    if dlog10Pcloud < 0:
        dlog10Pcloud = 0.0

    # Compute spectrum
    _, rprs2 = p.rprs2(atm, wavl=wavl, log10Pcloudbottom=log10Psurf, dlog10Pcloud=dlog10Pcloud, **kwargs)

    # Add the offset
    rprs2 += offset

    return rprs2

def make_loglike(data_dict):
    def loglike(cube):
        wavl = data_dict['wavl']
        y = data_dict['rprs2']
        e = data_dict['err']
        resulty = model(cube, wavl)
        loglikelihood = -0.5*np.sum((y - resulty)**2/e**2)
        return loglikelihood
    return loglike

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

def PRIOR(cube):
    params = cube.copy()
    params[0] = quantile_to_uniform(cube[0], -9, 1) # log10CO2
    params[1] = quantile_to_uniform(cube[1], -7, 1) # log10O2
    params[2] = quantile_to_uniform(cube[2], -7, 1) # log10CO
    params[3] = quantile_to_uniform(cube[3], -6, 0) # log10H2
    params[4] = quantile_to_uniform(cube[4], -11, 1) # log10CH4
    params[5] = quantile_to_uniform(cube[5], -5, 0) # log10Pcloud (top)
    params[6] = quantile_to_uniform(cube[6], -1000e-6, 1000e-6) # offset
    return params 

def prism_10trans():
    p = PICASO
    R_planet = planets.TRAPPIST1e.radius
    R_star = planets.TRAPPIST1.radius
    transit_duration = planets.TRAPPIST1e.transit_duration*60*60
    kmag = planets.TRAPPIST1.kmag
    inst = 'NIRSpec Prism'
    starpath = 'inputs/TRAPPIST1e_stellar_flux_picaso.txt'
    wavl, _, err = p.run_pandexo(
        R_planet,
        R_star,
        transit_duration*2,
        transit_duration,
        kmag,
        inst,
        starpath,
        R=100,
        ntrans=10
    )

    # Krissansen-totton et al. (2018)
    # CO2, O2, CO, H2, CH4, clouds, offset
    x = [-1.3, -7, -4.5, -4, -2.3, 0, 0]
    rprs2 = model(x, wavl)

    data_dict = {
        'wavl': wavl,
        'rprs2': rprs2,
        'err': err
    }

    return data_dict

# export HDF5_USE_FILE_LOCKING="FALSE"
# mpiexec -n 4 python -m mpi4py retrievals.py

if __name__ == '__main__':
    log_dir = 'ultranest/prism_10trans/'
    outfile = log_dir+'prism_10trans.pkl'
    if not os.path.isdir(log_dir): 
        os.mkdir(log_dir)

    # DATA_DICT = prism_10trans()
    # with open(log_dir+'data.pkl','wb') as f:
    #     pickle.dump(DATA_DICT,f)

    with open(log_dir+'data.pkl','rb') as f:
        DATA_DICT = pickle.load(f)

    LOGLIKE = make_loglike(DATA_DICT)
    sampler = ultranest.ReactiveNestedSampler(
        PARAM_NAMES,
        LOGLIKE,
        PRIOR,
        log_dir=log_dir,
        resume=True
    )
    result = sampler.run(min_num_live_points=400)
    with open(outfile,'wb') as f:
        pickle.dump(result, f)


