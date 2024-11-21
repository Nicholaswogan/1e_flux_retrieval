import pickle
import numpy as np
from scipy import interpolate

class GridInterpolator():

    def __init__(self, filename, gridvals):

        # read the file
        res = []
        inds = []
        with open(filename,'rb') as f:
            while True:
                try:
                    tmp = pickle.load(f)
                    res.append(tmp[2])
                    inds.append(tmp[0])
                except EOFError:
                    break

        # Reorder results to match gridvals
        res_new = [0 for a in res]
        for i,r in enumerate(res):
            res_new[inds[i]] = res[i]
        
        self.gridvals = gridvals
        self.gridshape = tuple([len(a) for a in gridvals])
        self.results = res_new

    def make_array_interpolator(self, key, logspace=False):
        # Make interpolate for key
        interps = []
        for j in range(len(self.results[0][key])):
            val = np.empty(len(self.results))
            for i,r in enumerate(self.results):
                val[i] = r[key][j]
            if logspace:
                val = np.log10(np.maximum(val,2e-38))
            interp = interpolate.RegularGridInterpolator(self.gridvals, val.reshape(self.gridshape))
            interps.append(interp)

        def interp_arr(vals):
            out = np.empty([len(interps)])
            for i,interp in enumerate(interps):
                out[i] = interp(vals)
            if logspace:
                out = 10.0**out
            return out
        
        return interp_arr
    
    def make_value_interpolator(self, key1, key2=None, logspace=False):
        val = np.empty(len(self.results))

        if key2 is None:
            for i,r in enumerate(self.results):
                val[i] = r[key1]
        else:
            for i,r in enumerate(self.results):
                val[i] = r[key1][key2]
        if logspace:
            val = np.log10(np.maximum(val,2e-38))
        interp = interpolate.RegularGridInterpolator(self.gridvals, val.reshape(self.gridshape))

        def interp1(vals):
            out = interp(vals)[0]
            if logspace:
                out = 10.0**out
            return out
        
        return interp1
        
def listener(q, filename):

    while True:

        m = q.get()
        if m == 'kill':
            break

        # Write to it
        with open(filename, 'ab') as f:
            pickle.dump(m,f)

def get_inputs(gridvals):
    tmp = np.meshgrid(*gridvals, indexing='ij')
    inputs = np.empty((tmp[0].size,len(tmp)))
    for i,t in enumerate(tmp):
        inputs[:,i] = t.flatten()
    return inputs