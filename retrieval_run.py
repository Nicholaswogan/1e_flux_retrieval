import numpy as np
from scipy import constants as const
from scipy import optimize
import pandas as pd
import os
import h5py
import dill as pickle
from copy import deepcopy
from pymultinest.solve import solve

from photochem._clima import rebin_with_errors
from photochem.utils import stars

import gridutils
import photochem_grid
import planets
import utils

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

def make_interpolators(filename, pc):
    gridvals = photochem_grid.get_gridvals()
    g = gridutils.GridInterpolator(filename, gridvals)

    gas_names = pc.dat.species_names[pc.dat.np:-2]
    MIX = {}
    for i,sp in enumerate(gas_names):
        MIX[sp] = g.make_interpolator(sp,logspace=True)
    particle_names = pc.dat.species_names[:pc.dat.np]
    PARTICLES = {}
    for i,sp in enumerate(particle_names):
        PARTICLES[sp] = g.make_interpolator(sp,logspace=True)
    FLUXES = {}
    for i,sp in enumerate(gas_names):
        FLUXES[sp] = g.make_interpolator(sp+'_BOA_flux')

    PRESS = g.make_interpolator('P',logspace=True)
    ALT = g.make_interpolator('z')
    TEMP = g.make_interpolator('T')

    return PRESS, ALT, TEMP, MIX, PARTICLES, FLUXES

def make_atm(x, P_interp, T_interp, f_interp):
    atm = {}
    atm['pressure'] = P_interp(x)
    atm['temperature'] = T_interp(x)
    ftot = np.zeros(atm['pressure'].shape[0])
    for key in f_interp:
        atm[key] = f_interp[key](x)
        ftot += atm[key]
    for key in f_interp:
        atm[key] /= ftot
    atm['pressure'] *= ftot # this seems to be a good thing to do
    return atm

def make_picaso_atm(x, P_interp, T_interp, f_interp):
    atm = make_atm(x, P_interp, T_interp, f_interp)
    atm['pressure'] /= 1e6 # bars
    for key in atm:
        atm[key] = atm[key][::-1].copy()
    atm = pd.DataFrame(atm)
    return atm

def vdep(x, sp, P_interp, T_interp, f_interp, flux_interp):
    P = P_interp(x)
    T = T_interp(x)
    Px = f_interp[sp](x)[0]*P[0]
    k_boltz = const.k*1e7
    den = Px/(k_boltz*T[0])
    flux = (flux_interp[sp](x))
    return (-flux/den)

def make_picaso(filename_db):
    M_planet = planets.TRAPPIST1e.mass
    R_planet = planets.TRAPPIST1e.radius
    R_star = planets.TRAPPIST1.radius
    star_kwargs = {
        'filename': 'inputs/TRAPPIST1e_hazmat_picaso.txt',
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

# Globals
PICASO = make_picaso(os.path.join(os.environ['picaso_refdata'],'opacities/opacities_0.3_15_R15000.db'))
PRESS, ALT, TEMP, MIX, PARTICLES, FLUXES = make_interpolators('results/photochem_v1.h5', photochem_grid.PHOTOCHEMICAL_MODEL)
PRESS_M, ALT_M, TEMP_M, MIX_M, PARTICLES_M, FLUXES_M = make_interpolators('results/photochem_muscles_v1.h5', photochem_grid.PHOTOCHEMICAL_MODEL_MUSCLES)
ALL_MODEL_PARAMETERS = ['log10CO2','log10O2','log10CO','log10H2','log10CH4','log10Pcloud','offset']

def _model(y, wavl, p, P_interp, T_interp, f_interp, **kwargs):

    log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4,log10Pcloud,offset = y

    # Atmosphere
    x = np.array([log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4])
    atm = make_picaso_atm(x, P_interp, T_interp, f_interp)

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

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

def build_x(y, params):
    x = np.zeros(len(ALL_MODEL_PARAMETERS))
    j = 0
    for i,sp in enumerate(ALL_MODEL_PARAMETERS):
        if params[sp] is None:
            x[i] = y[j]
            j += 1
        else:
            x[i] = params[sp]
    return x

def make_model(params, p, P_interp, T_interp, f_interp):
    def model1(y, wavl, **kwargs):
        x = build_x(y, params)
        return _model(x, wavl, p, P_interp, T_interp, f_interp, **kwargs)
    return model1

def make_loglike_prior(params, data_dict, p, P_interp, z_interp, T_interp, f_interp, particle_interp, flux_interp, prior_ranges=None):
    # example:
    # params = {
    #     'log10CO2': None, # means it is fit for
    #     'log10O2': -17, # means it is fixed
    #     ...
    # }

    # Get the parameter names
    truth = data_dict['truth']
    truth_new = []
    param_names = []
    for i,sp in enumerate(ALL_MODEL_PARAMETERS):
        if params[sp] is None:
            param_names.append(sp)
            truth_new.append(truth[i])

    # This data_dict will have truth that can be input to the model.
    data_dict_copy = deepcopy(data_dict)
    data_dict_copy['truth'] = truth_new

    # Default prior ranges
    if prior_ranges is None:
        prior_ranges = {
            'log10CO2': (-9, 1),
            'log10O2': (-15, 1),
            'log10CO': (-11, 1),
            'log10H2': (-9, 0),
            'log10CH4': (-11, 1),
            'log10Pcloud': (-5, 0),
            'offset': (-1000e-6, 1000e-6)
        }

    model1 = make_model(params, p, P_interp, T_interp, f_interp)
    
    def loglike(cube):
        wavl = data_dict['wavl']
        y = data_dict['rprs2']
        e = data_dict['err']
        resulty = model1(cube, wavl)
        loglikelihood = -0.5*np.sum((y - resulty)**2/e**2)
        return loglikelihood
    
    def prior(y):
        res = np.empty_like(y)
        j = 0
        for i,sp in enumerate(ALL_MODEL_PARAMETERS):
            if params[sp] is None:
                res[j] = quantile_to_uniform(y[j], *prior_ranges[sp])
                j += 1
        return res

    out = {
        'loglike': loglike,
        'prior': prior,
        'param_names': param_names,
        'model': model1,
        'params': params,
        'data_dict': data_dict,
        'P_interp': P_interp,
        'z_interp': z_interp,
        'T_interp': T_interp,
        'f_interp': f_interp,
        'particle_interp': particle_interp,
        'flux_interp': flux_interp,
    }

    return out

def make_prism_data():
    "One transit NIRSpec PRISM TRAPPIST-1e"
    p = PICASO
    R_planet = planets.TRAPPIST1e.radius
    R_star = planets.TRAPPIST1.radius
    transit_duration = planets.TRAPPIST1e.transit_duration*60*60
    kmag = planets.TRAPPIST1.kmag
    inst = 'NIRSpec Prism'
    starpath = 'inputs/TRAPPIST1e_hazmat_picaso.txt'
    wavl, _, err = p.run_pandexo(
        R_planet,
        R_star,
        transit_duration*2,
        transit_duration,
        kmag,
        inst,
        starpath,
        ntrans=1,
        R=None
    )

    with h5py.File('data/TRAPPIST1e_prism.h5','w') as f:
        f.create_dataset('wavl', shape=wavl.shape, dtype=wavl.dtype)
        f['wavl'][:] = wavl

        f.create_dataset('err', shape=err.shape, dtype=err.dtype)
        f['err'][:] = err

def prism_data(R=None, ntrans=1):

    # 1 transit NIRSpec Prism
    with h5py.File('data/TRAPPIST1e_prism.h5','r') as f:
        wavl = f['wavl'][:]
        err = f['err'][:]

    err = err/np.sqrt(ntrans)

    if R is not None:
        wavl_n = stars.grid_at_resolution(np.min(wavl), np.max(wavl), R)
        _, err = rebin_with_errors(wavl.copy(), err.copy(), err.copy(), wavl_n.copy())
        wavl = wavl_n

    return wavl, err

def nominal_archean_objective(y, P_interp, T_interp, f_interp, flux_interp):
    x = np.append(-1.3,y)
    F_O2 = flux_interp['O2'](x)
    vdep_CO = vdep(x, 'CO', P_interp, T_interp, f_interp, flux_interp)
    vdep_H2 = vdep(x, 'H2', P_interp, T_interp, f_interp, flux_interp)
    F_CH4 = flux_interp['CH4'](x)

    res = np.array([
        F_O2 - 0.0,
        vdep_CO - 1.2e-4,
        vdep_H2 - 2.4e-4,
        F_CH4 - 1.122e11
    ])
    scaling = np.array([
        1e-11,
        1e4,
        1e4,
        1e-11
    ])
    return res*scaling

def compute_nominal_archean():
    guess = [-10.0, -4.4, -4.2, -3.6]
    args = (PRESS, TEMP, MIX, FLUXES) # Hazmat case
    sol = optimize.root(nominal_archean_objective, guess, args=args, method='lm')
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-10

    # - 5% CO2 is a sensible values, resulting in a reasonable surface T (~297 K).
    # - F_O2 = 0 molecules/cm^2/s
    # - F_CH4 = 1.122e11 molecules/cm^2/s = Modern biological flux
    # - vdep_CO = 1.2e-4 cm/s = Plausible Archean Earth value
    # - vdep_H2 = 2.4e-4 cm/s = Plausible Archean Earth value
    # CO2 + other gases + cloud top + offset
    # Choose cloud top of -0.7 (equal to 0.2 bar), because this is Earth's tropopause.

    x = [-1.3] + list(sol.x) + [-0.7, 0]
    return x

def make_data_dict_nominal_archean(ntrans, R=None):

    # Get error bars
    wavl, err = prism_data(ntrans=ntrans, R=R)

    # Get inputs for the nominal archean case
    x = compute_nominal_archean()
    model = make_model({a: None for a in ALL_MODEL_PARAMETERS}, PICASO, PRESS, TEMP, MIX) # we use HAZMAT grid
    rprs2 = model(x, wavl)

    data_dict = {
        'wavl': wavl,
        'rprs2': rprs2,
        'err': err,
        'truth': x
    }

    return data_dict

def make_cases():
    cases = {}

    # Nominal case. Archean Earth with 10 transits of Prism. Hazmat grid.
    data_dict = make_data_dict_nominal_archean(ntrans=10) # data made with Hazmat grid.
    params = {a: None for a in ALL_MODEL_PARAMETERS} # fit for all parameters
    cases['archean'] = make_loglike_prior(params, data_dict, PICASO, PRESS, ALT, TEMP, MIX, PARTICLES, FLUXES)

    # Archean Earth with 10 transits of Prism. Muscles grid, but the data is generated with Hazmat grid.
    params = {a: None for a in ALL_MODEL_PARAMETERS} # fit for all parameters
    cases['archean_muscles'] = make_loglike_prior(params, data_dict, PICASO, PRESS_M, ALT_M, TEMP_M, MIX_M, PARTICLES_M, FLUXES_M)

    # Nominal case, except we will assume that we know the partial pressures of O2, CO and H2. 
    # This gets at the question of how much not knowing these things matters.
    x = data_dict['truth']
    params = {ALL_MODEL_PARAMETERS[i]: x[i] for i in range(len(ALL_MODEL_PARAMETERS))}
    params['log10CO2'] = None
    params['log10CH4'] = None
    params['log10Pcloud'] = None
    params['offset'] = None
    cases['archean_constrained'] = make_loglike_prior(params, data_dict, PICASO, PRESS, ALT, TEMP, MIX, PARTICLES, FLUXES)

    # Make a couple more cases which consider all parameters, but with various number of transits
    params = {a: None for a in ALL_MODEL_PARAMETERS}
    for ntrans in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
        data_dict = make_data_dict_nominal_archean(ntrans=ntrans)
        key = 'archean_%i'%ntrans
        cases[key] = make_loglike_prior(params, data_dict, PICASO, PRESS, ALT, TEMP, MIX, PARTICLES, FLUXES)

    return cases

RETRIEVAL_CASES = make_cases()

if __name__ == '__main__':
    # mpiexec -n <number of processes> python retrieval_run.py

    models_to_run = [
        'archean_20','archean_30','archean_40','archean_50','archean_60',
        'archean_70','archean_80','archean_90','archean_100'
    ]
    for model_name in models_to_run:

        # Setup directories
        outputfiles_basename = f'pymultinest/{model_name}/{model_name}'
        try:
            os.mkdir(f'pymultinest/{model_name}')
        except FileExistsError:
            pass

        # Do nested sampling
        results = solve(
            LogLikelihood=RETRIEVAL_CASES[model_name]['loglike'], 
            Prior=RETRIEVAL_CASES[model_name]['prior'], 
            n_dims=len(RETRIEVAL_CASES[model_name]['param_names']), 
            outputfiles_basename=outputfiles_basename, 
            verbose=True,
            n_live_points=1000
        )
        # Save pickle
        pickle.dump(results, open(outputfiles_basename+'.pkl','wb'))
