
import warnings
warnings.filterwarnings('ignore')

import multiprocessing as mp
import pickle
from tqdm import tqdm
import os
import numpy as np
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import grid_utils
import utils
import climate_grid

# This stuff has to be top level but does not have to change

def worker(i, x, q):
    res = model(x)
    q.put((i,x,res,))

def main(gridvals, filename, ncores):

    inputs = grid_utils.get_inputs(gridvals) # get inputs

    # Exception if the file already exists
    inds = []
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            while True:
                try:
                    tmp = pickle.load(f)
                    ind = tmp[0]
                    x = tmp[1]
                    assert np.all(np.isclose(inputs[ind,:], x))
                    inds.append(ind)
                except EOFError:
                    break
        print('Number of calculations already completed: %i. Restarting calculation.'%(len(inds)))
    else:
        if not os.path.isfile(filename):
            with open(filename, 'wb') as f:
                pass

    manager = mp.Manager()
    q = manager.Queue()
    with mp.Pool(ncores+1) as pool:

        # Put listener to work first
        watcher = pool.apply_async(grid_utils.listener, (q,filename,))

        # Fire off workers
        jobs = []
        for i in range(inputs.shape[0]):
            if i in inds:
                continue
            x = inputs[i,:]
            job = pool.apply_async(worker, (i, x, q))
            jobs.append(job)

        # Collect results from the workers through the pool result queue
        for job in tqdm(jobs): 
            job.get()

        # Kill the listener
        q.put('kill')

def get_gridvals():
    log10PCO2 = np.arange(-5,1.01,1)
    log10PO2 = np.arange(-7,1.01,2)
    log10PCO = np.arange(-7,1.01,2)
    log10PH2 = np.arange(-6,0.01,2)
    log10PCH4 = np.arange(-7,1.01,1)
    gridvals = (log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4)
    return gridvals

def make_climate_interpolators():
    gridvals = climate_grid.get_gridvals()
    g = grid_utils.GridInterpolator('results/climate_fixed_v1.pkl',gridvals)
    PRESS = g.make_array_interpolator('P',logspace=True)
    TEMP = g.make_array_interpolator('T')
    MIX = {}
    species = ['H2O','CO2','N2','O2','CO','H2','CH4']
    for sp in species:
        MIX[sp] = g.make_array_interpolator(sp,logspace=True)
    return PRESS, TEMP, MIX

def make_photochemical_model():
    pc = utils.EvoAtmosphereRobust(
        'inputs/zahnle_HNOC.yaml',
        'inputs/settings.yaml',
        'inputs/TRAPPIST1e_stellar_flux.txt'
    )
    pc.set_particle_radii({
        'H2Oaer': 1.0e-2, # 100 microns
        'HCaer1': 1.0e-4, # =1 micron
        'HCaer2': 1.0e-4,
        'HCaer3': 1.0e-4,
    })
    pc.rdat.max_total_step = 10_000
    pc.rdat.verbose = False
    return pc

PRESS, TEMP, MIX = make_climate_interpolators()
PHOTOCHEMICAL_MODEL = make_photochemical_model()

def model(y):
    pc = PHOTOCHEMICAL_MODEL

    log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4 = y

    log10PN2 = 0.0
    log10Kzz = 5.0

    x1 = (log10PN2,log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4)
    ind = -4
    
    # P-T-mix profile from climate model
    P = PRESS(x1)[:ind]
    T = TEMP(x1)[:ind]
    mix = {}
    for key in MIX:
        mix[key] = MIX[key](x1)[:ind]

    # Constant Kzz
    Kzz = np.ones(P.shape[0])*10.0**log10Kzz

    # Surface boundary conditions
    Pi = x_to_press(x1)

    pc.initialize_to_PT_bcs(P, T, Kzz, mix, Pi)
    converged = pc.find_steady_state()

    result = make_result(y, pc, converged)

    return result

def make_result(y, pc, converged):
    P = pc.wrk.pressure
    z = pc.var.z
    T = pc.var.temperature
    sol = pc.mole_fraction_dict()
    surf, top = pc.gas_fluxes()
    for key in surf:
        surf[key] = surf[key].astype(np.float32)
        top[key] = top[key].astype(np.float32)

    result = {}
    result['converged'] = converged
    result['x'] = y.astype(np.float32)
    result['P'] = P.astype(np.float32)
    result['z'] = z.astype(np.float32)
    result['T'] = T.astype(np.float32)
    for i,sp in enumerate(pc.dat.species_names[:-2]):
        result[sp] = sol[sp].astype(np.float32)
    result['BOA flux'] = surf
    result['TOA flux'] = top
    result['max time'] = pc.rdat.max_time

    return result

def x_to_press(x):
    species = ['N2','CO2','O2','CO','H2','CH4']
    Pi = {
        'H2O': 270.0e6
    }
    for i,key in enumerate(species):
        Pi[key] = 1e6*10.0**x[i]
    return Pi

if __name__ == "__main__":
    filename = 'results/photochem_v1.1.pkl'
    ncores = 4
    gridvals = get_gridvals()
    main(gridvals, filename, ncores)

    


