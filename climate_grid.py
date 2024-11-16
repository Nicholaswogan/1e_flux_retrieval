import warnings
warnings.filterwarnings('ignore')

from clima import ClimaException
import pickle
import multiprocessing as mp
from tqdm import tqdm
import os
import numpy as np
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import grid_utils
import utils

# This stuff has to be top level but does not have to change

def worker(i, x, q):
    res = model(x)
    q.put((i,x,res,))

def main(gridvals, filename, ncores):

    inputs = grid_utils.get_inputs(gridvals) # get inputs

    # Exception if the file already exists
    if os.path.isfile(filename):
        raise Exception(filename+' already exists!')
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
            x = inputs[i,:]
            job = pool.apply_async(worker, (i, x, q))
            jobs.append(job)

        # Collect results from the workers through the pool result queue
        for job in tqdm(jobs): 
            job.get()

        # Kill the listener
        q.put('kill')

# Make the model function

# Initialize the climate model
CLIMATE_MODEL = utils.AdiabatClimateRobust(
    'inputs/species_climate.yaml',
    'inputs/settings_climate.yaml',
    'inputs/TRAPPIST1e_stellar_flux.txt'
)
CLIMATE_MODEL.verbose = False

def model(x):
    c = CLIMATE_MODEL

    P_i = x_to_Pi(x, c)

    # Compute climate
    converged = c.RCE_robust(P_i)

    result = make_result(x, c, converged)

    return result

def x_to_Pi(x, c):
    log10PN2,log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4 = x
    P_i = np.ones(len(c.species_names))*1e-10
    P_i[c.species_names.index('H2O')] = 270.0
    P_i[c.species_names.index('N2')] = 10.0**log10PN2
    P_i[c.species_names.index('CO2')] = 10.0**log10PCO2
    P_i[c.species_names.index('O2')] = 10.0**log10PO2
    P_i[c.species_names.index('CO')] = 10.0**log10PCO
    P_i[c.species_names.index('H2')] = 10.0**log10PH2
    P_i[c.species_names.index('CH4')] = 10.0**log10PCH4
    P_i *= 1.0e6 # convert to dynes/cm^2
    return P_i

def make_result(x, c, converged):
    # Save the P-z-T profile
    P = np.append(c.P_surf,c.P)
    z = np.append(0,c.z)
    T = np.append(c.T_surf,c.T)

    # Mixing ratios
    f_i = np.concatenate((np.array([c.f_i[0,:]]),c.f_i),axis=0)

    # Save results as 32 bit floats
    result = {}
    result['converged'] = converged
    result['x'] = x.astype(np.float32)
    result['P'] = P.astype(np.float32)
    result['z'] = z.astype(np.float32)
    result['T'] = T.astype(np.float32)
    for i,sp in enumerate(c.species_names):
        result[sp] = f_i[:,i].astype(np.float32)

    return result

def get_gridvals():
    log10PN2 = np.array([-1.0, 0.0])
    log10PCO2 = np.arange(-5,1.01,1)
    log10PO2 = np.arange(-7,1.01,2)
    log10PCO = np.arange(-7,1.01,2)
    log10PH2 = np.arange(-6,0.01,2)
    log10PCH4 = np.arange(-7,1.01,1)
    gridvals = (log10PN2,log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4)
    return gridvals

def fix_climate_grid(filename, outfile):

    c = CLIMATE_MODEL

    res = []
    i = 0
    with open(filename,'rb') as f:
        while True:
            try:
                tmp = pickle.load(f)
                res.append(tmp)
            except EOFError:
                break

    # Find failed runs 
    failed = []
    for i in range(len(res)):
        val = res[i]
        if not val[2]['converged']:
            failed.append(val)

    # Fix the failed runs
    fixed = []
    for i,val in enumerate(failed):
        print('Working on failed %i'%i)
        for perturb in [-1,1,-2,2,-3,3]:
            converged = False
            for j in range(len(res)):
                if res[j][0] == val[0]+perturb and res[j][2]['converged']:
                    # Run model that worked
                    model(res[j][1])

                    # Now run model that didn't work with initial conditions
                    # from nearby one that did work. 
                    P_i = x_to_Pi(val[1], c)
                    try:
                        converged = c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below)
                    except ClimaException:
                        converged = False
                    break
            if converged:
                break
            
        result = make_result(val[1], c, converged)
        fixed.append((val[0],val[1],result))
        print(converged)

    res_new = []
    for val in res:
        if val[2]['converged']:
            res_new.append(val)
        else:
            found = False
            for val1 in fixed:
                if val1[0] == val[0]:
                    found = True
                    replace = val1
            
            if not found:
                raise Exception()

            res_new.append(replace)

    with open(outfile, 'wb') as f:
        pass

    for val in res_new:
        with open(outfile, 'ab') as f:
            pickle.dump(val,f)

if __name__ == "__main__":
    filename = 'results/climate_v1.pkl' # Specify output filename
    ncores = 4 # Specify number of cores
    gridvals = get_gridvals()
    main(gridvals, filename, ncores)
    fix_climate_grid(filename, 'results/climate_fixed_v1.pkl')