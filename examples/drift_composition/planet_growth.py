import numpy as np
from drift_composition.constants import k_boltzmann, m_hydrogen, G_Msun, Rau, Msun, yr, Mearth

def seed_mass(hr, flaring, gas_slope, dist):
    dist = dist*Rau
    vk = np.sqrt(mass_star*G_Msun/dist)
    pres_grad = 2*(flaring-1)+gas_slope
    eta = - 0.5* hr**2 * pres_grad
    m_min = (eta*vk)**3/G_Msun/vk*dist/np.sqrt(3)
    return m_min

def loc_disc (g_val, Rg, dist):
    cid = np.argmin(np.abs(Rg-dist))
    print(cid)
    loc_val = g_val[cid]
    #d_val = (g_val[cid+1]-g_val[cid-1])/(Rg[cid+1]-Rg[cid-1])
    #loc_val = g_val[cid] + d_val*(dist-Rg[cid])
    return loc_val

class Planet:
    def __init__(self, mass, mc, mg, f_comp, dist=10.0):
        self.mass = mass
        self.mc = mc
        self.mg = mg
        self.dist = dist 
        self.f_comp = f_comp

class PlanetEnv:
    def __init__(self, grid, alpha, mu, mass_star):
        self.alpha = alpha
        self.mu    = mu
        self.mass_star = mass_star
        self.grid  = grid

    def temp(self, T, dist):
        return loc_disc(T, self.grid.Rc, dist) 

    def sig_gas(self, disc, dist):    
        return loc_disc(disc.Sigma_gas, self.grid.Rc, dist) 

    def sig_dust(self, disc, dist):    
        return loc_disc(disc.Sigma_dust, self.grid.Rc, dist) 

    def mols(self, disc): 
        molc = disc.Molecules
        return molc

    def sig_mol(self, disc, dist):
        molc = disc.Molecules
        sigma_mol  = disc.Sigma_mol

        sig_mol_d = {}
        sig_mol_g = {}
        for mol in molc:
            s_mol_d = loc_disc(sigma_mol[mol.name][:,1], self.grid.Rc, dist)
            sig_mol_d[mol.name] = s_mol_d
            
            s_mol_g = loc_disc(sigma_mol[mol.name][:,0], self.grid.Rc, dist)
            sig_mol_g[mol.name] = s_mol_g
        return sig_mol_g, sig_mol_d

    def Stokes(self, disc, dist):
        st = disc.Stokes
        if np.isscalar(st):
            stokes = st
        else:
            stokes = loc_disc(st, self.grid.Re, dist)
        return stokes

    def vk(self, dist):
        return np.sqrt(self.mass_star*G_Msun/dist)

    def hr(self, T, dist):
        temp = loc_disc(T, self.grid.Rc, dist) 
        cs = np.sqrt(k_boltzmann/self.mu/m_hydrogen*temp)  
        return cs/(np.sqrt(self.mass_star*G_Msun/dist))

#def hr (temperature,dist, star_mass, mu):
#    vk = np.sqrt(mass_star*G_Msun/dist)
#    cs = np.sqrt(k_boltzmann/mu/m_hydrogen*temperature)
#    hr = 0.05#cs/vk
#    return hr

def pebble_accretion(planet, p_env, disc, T):
    mass_p = planet.mass
    dist   = planet.dist
    hr        = p_env.hr(T,dist)
    stokes    = p_env.Stokes(disc,dist)
    mass_star = p_env.mass_star
    alpha     = p_env.alpha
    pebble_density = p_env.sig_dust(disc,dist)
    
    r_hill = dist*(mass_p/mass_star/3.)**(1./3.)
    v_hill = r_hill * np.sqrt(mass_star*G_Msun / dist**3)
    h_peb = hr * dist * np.sqrt(alpha / stokes)
    dm_2d = 2.0 * (stokes / 0.1)**(2. / 3.) * r_hill * v_hill * pebble_density
    dm_3d = dm_2d * (r_hill * np.pi**0.5 / 2**1.5 / h_peb *(stokes/0.1)**(1./3.))
    crit_h = np.pi* (stokes/0.1)**(1./3.) * r_hill /2/np.sqrt(2*np.pi)
    if h_peb > crit_h: dm_peb = dm_2d
    else: dm_peb = dm_3d
    return dm_peb/Msun*yr

def gas_accretion(planet, p_env, disc, T, f=0.2, kap=0.05, rho_c=5.):
    mass_p = planet.mass
    mc     = planet.mc
    mg     = planet.mg
    dist   = planet.dist
    gas_density = p_env.sig_gas(disc,dist)
    temperature = p_env.temp(T,dist)
    hr          = p_env.hr(T,dist)

    r_hill = dist*(mass_p/p_env.mass_star/3.)**(1./3.)
    omg_k  = np.sqrt(p_env.mass_star*G_Msun/dist**3)
    if mc > mg:
        dm_gas = (0.00175/f/f/ kap * (rho_c/5.5)**(-1./6.) * np.sqrt(81/temperature)
            *(mc/(Mearth/Msun))**(11./3.) * (0.1*Mearth/Msun / mg) * Mearth/1e6)/Msun
    else:
        dm_low  = 0.83 * omg_k * gas_density * (hr*dist)**2 * (r_hill/hr/dist)**(4.5) /Msun*yr
        dm_high = 0.14 * omg_k * gas_density * (hr*dist)**2 /Msun*yr
        dm_gas = np.min((dm_low,dm_high))
    return dm_gas  

def mass_growth(planet, p_env, disc, T, dt):
    dist = planet.dist
    mol_comp = planet.f_comp

    #print(planet.mass, 20 * (p_env.hr(T, dist)/0.05)**3. * Mearth/Msun, p_env.hr(T, dist))
    if planet.mass > 20 * (p_env.hr(T, dist)/0.05)**3. * Mearth/Msun:
        dm_peb = 0
    else:
        dm_peb = pebble_accretion(planet, p_env, disc, T)
    dm_gas = gas_accretion(planet, p_env, disc, T)
    mc = planet.mc + dm_peb*dt
    mg = planet.mg + dm_gas*dt
    
    #print(dm_peb,dm_gas)
    sg = p_env.sig_gas(disc,dist)
    sd = p_env.sig_dust(disc,dist)

    molg, mold = p_env.sig_mol(disc,dist)
    mol_names = list(molg.keys())

    for mol in mol_names:
        dm_mol_g = dm_gas*(molg[mol]/sg)
        dm_mol_d = dm_peb*(mold[mol]/sd)
        mol_comp[mol][0] = mol_comp[mol][0] + dm_mol_g*dt
        mol_comp[mol][1] = mol_comp[mol][1] + dm_mol_d*dt
        #mass = planet.mass+dm*dt

    new_planet = Planet(mc+mg, mc, mg, mol_comp, planet.dist)
    return new_planet

