import numpy as np

class Star:
    radius : float # relative to the sun
    Teff : float # K
    metal : float # log10(M/H)
    kmag : float
    logg : float
    planets : dict # dictionary of planet objects

    def __init__(self, radius, Teff, metal, kmag, logg, planets):
        self.radius = radius
        self.Teff = Teff
        self.metal = metal
        self.kmag = kmag
        self.logg = logg
        self.planets = planets
        
class Planet:
    radius : float # in Earth radii
    mass : float # in Earth masses
    Teq : float # Equilibrium T in K
    transit_duration : float # in seconds
    a: float # semi-major axis in AU
    a: float # semi-major axis in AU
    stellar_flux: float # W/m^2
    
    def __init__(self, radius, mass, Teq, transit_duration, a, stellar_flux):
        self.radius = radius
        self.mass = mass
        self.Teq = Teq
        self.transit_duration = transit_duration
        self.a = a
        self.stellar_flux = stellar_flux

# Agol et al.

TRAPPIST1e = Planet(
    radius=0.920,
    mass=0.692,
    Teq=249.9,
    transit_duration=0.9385, # Exo.Mast
    a=2.925e-2,
    stellar_flux=0.646*1370
)

TRAPPIST1 = Star(
    radius=0.1192,
    Teff=2566,
    metal=np.nan,
    kmag=10.3, # Exo.Mast
    logg=5.2396,
    planets={'e':TRAPPIST1e}
)








