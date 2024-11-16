import numpy as np
from clima import AdiabatClimate, ClimaException

class AdiabatClimateRobust(AdiabatClimate):

    def __init__(self, species_file, settings_file, flux_file, data_dir=None):

        super().__init__(
            species_file, 
            settings_file, 
            flux_file,
            data_dir
        )

        self.solve_for_T_trop = True # Enable solving for T_trop
        self.max_rc_iters = 30 # Lots of iterations
        self.P_top = 10.0 # 10 dynes/cm^2 top, or 1e-5 bars.
        self.convective_newton_step_size = 0.05
        self.RH = np.ones(len(self.species_names))*0.5 # 0.5 RH

    def surface_temperature_robust(self, P_i, T_guess_mid=None, T_perturbs=None):

        if T_guess_mid is None:
            T_guess_mid = self.rad.skin_temperature(0)*(1/2)**(-1/4) + 50
        
        if T_perturbs is None:
            T_perturbs = np.array([0.0, 20.0, -20.0, 30.0, 50.0, 80.0, 100.0, 200.0, 300.0, 400.0, 600.0])

        for i,T_perturb in enumerate(T_perturbs):
            T_surf_guess = T_guess_mid + T_perturb
            try:
                self.T_trop = self.rad.skin_temperature(0.0)*1.2
                self.surface_temperature(P_i, T_surf_guess)
                converged = True
                break
            except ClimaException as e:
                converged = False
        
        return converged
    
    def RCE_simple_guess(self, P_i):

        converged_simple = self.surface_temperature_robust(P_i)
        if not converged_simple:
            # If this fails, then we give up, returning no convergence
            return False
        
        # If simple climate model converged, then save the atmosphere
        T_surf_guess, T_guess, convecting_with_below_guess = self.T_surf, self.T, self.convecting_with_below
        
        try:
            converged = self.RCE(P_i, T_surf_guess, T_guess, convecting_with_below_guess)
        except ClimaException:
            converged = False

        return converged
        
    def RCE_isotherm_guess(self, P_i, T_guess_mid=None, T_perturbs=None):  

        if T_guess_mid is None:
            T_guess_mid = self.rad.skin_temperature(0)*(1/2)**(-1/4)
        
        if T_perturbs is None:
            T_perturbs = np.array([0.0, 10.0, -10.0, 20.0, 30.0, 50.0, 80.0])

        # First, we try a bunch of isothermal atmospheres.
        for i,T_perturb in enumerate(T_perturbs):
            T_surf_guess = T_guess_mid + T_perturb
            T_guess = np.ones(self.T.shape[0])*T_surf_guess
            try:
                converged = self.RCE(P_i, T_surf_guess, T_guess)
                if converged:
                    break
            except ClimaException:
                converged = False

        return converged
    
    def RCE_robust(self, P_i, T_guess_mid=None, T_perturbs=None):

        # First try guess based on simple climate model
        converged = self.RCE_simple_guess(P_i)
        if converged:
            return converged

        # Next try with isotherms
        converged = self.RCE_isotherm_guess(P_i, T_guess_mid, T_perturbs)

        return converged


