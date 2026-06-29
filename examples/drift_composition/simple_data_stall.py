from drift_composition.constants import Mearth, Rau, stefan_boltzmann, G_Msun, Msun, Lsun, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.simple_planet import Planet, PlanetEnv
from drift_composition.atoms import atoms_in_molecule, ELEMENT_MASS, load_protosolar_abundances, molecule_mass
from drift_composition.simple_reduction import Evolution, atom_mass, dust_to_gas, final_accretion, crit_mass
import drift_composition.simple_reduction as red
import drift_composition.simple_planet as simp

from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np


def store_data_range(planet_ini, DM, p_env, T, inp='test', f_plansis=np.logspace(-6,-1,10), radii = np.linspace(6.,9.,10), final_radius = 1e-3, si= True, stall=0.1):

    dt_ini = 500
    #dts = np.linspace(500,1000,10)
    Nt = 5000
    header = "#mini, mcini, mgini, rini, plans, rfin, mfin, mcfin, mgfin, mgH, mgO, mgC, mdH, mdO, mdC, m10, mg10, mc10, mgH10, mgO10, mgC10, mdH10, mdO10, mdC10, yr \n"
    f = open('data2/{}.txt'.format(inp), 'w')
    f.write(header)
    p_ini = planet_ini

    for fp in f_plansis:
        for rad in radii:
            fin_r =final_radius*(1+9*np.random.rand())
            p_ini.dist = rad*Rau
            planet_evo, nn = simp.std_evo_comp(p_ini, DM, p_env, T(p_env.grid.Rc),fp, dt_ini, Nt, final_radius=fin_r, stall=stall)
            planet_fin = planet_evo[-1]
            #print(planet_fin.dist/Rau , fin_r)
            if si:
                exc = ()
            else:
                exc = list(p_env.dust.keys())
            #print('Si',si)
            evo = Evolution(planet_evo, nn, exclude=exc)
            fin_mass, fin_mc, fin_mg, fin_comp, fin_atom = final_accretion(evo, crit_mass(evo, threshhold_mass_fraction=0.9))
            data = (str(planet_ini.mass), 
                    str(planet_ini.mc), 
                    str(planet_ini.mg), 
                    str(planet_ini.dist/Rau),
                    str(fp),
                    str(planet_fin.dist/Rau),
                    str(planet_fin.mass), 
                    str(planet_fin.mc), 
                    str(planet_fin.mg), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['H'][0]), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['O'][0]), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['C'][0]),
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['H'][1]), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['O'][1]), 
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['C'][1]),
                    str(fin_mass), 
                    str(fin_mc), 
                    str(fin_mg),
                    str(fin_atom['H'][0]),
                    str(fin_atom['O'][0]),
                    str(fin_atom['C'][0]),
                    str(fin_atom['H'][1]),
                    str(fin_atom['O'][1]),
                    str(fin_atom['C'][1]),
                    str(planet_fin.time),
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['N'][0]),
                    str(atom_mass(planet_fin.f_comp,exclude=exc)['N'][1]),
                    str(fin_atom['N'][0]),
                    str(fin_atom['N'][1]),
                    str(atom_mass(planet_fin.f_comp,exclude=list(p_env.dust.keys()))['O'][0]),
                    str(atom_mass(planet_fin.f_comp,exclude=list(p_env.dust.keys()))['O'][1]),
                    )
            #print(data)
            f.write('  '.join(data))
            f.write('\n')
            #print(planet_ini.dist/Rau ,planet_fin.dist/Rau)
    f.close()
    pass


def create_temperature_profile(grid, Lstar, Mdot_gas, alpha, 
                               kappa=1, gamma=10, phi=0.05, Mstar=1, Tmax=1300, mu=2.35):
    """Estimate an irridiated and viscously heated temperature profile given the
    accretion rate, viscous alpha and Rosseland-mean opacity (per unit gas) mass"""

    class T_Profile:
        def __init__(self, grid, tauZ, Lstar, Mdot, Mstar, gamma, phi, Tmax):
            
            # Interpolate the vertical optical depth:
            self._interp = PchipInterpolator(np.log(grid.Rc), np.log(tauZ))

            self._Lstar = Lstar
            self._Mstar = Mstar
            self._Mdot = Mdot

            self._gamma = gamma
            self._phi = phi

            self._Tmax = Tmax


        def tau(self, R):
            return np.exp(self._interp(np.log(R)))

        def Tirr4(self, R):
            return self._Lstar*Lsun / (4*np.pi*R*R * stefan_boltzmann)

        def Tint4(self, R):
            Omk2 = G_Msun * self._Mstar / R**3
            return (3/(8*np.pi)) * self._Mdot*Msun/yr * Omk2 / stefan_boltzmann
            
        def __call__(self, R):
            tau = self.tau(R)
            f_visc = (2/3 + tau) 

            fac = np.exp(-self._gamma*tau/self._phi)
            fac1 = -np.expm1(-self._gamma*tau/self._phi)

            f_irr = 0.5*self._phi + 3*self._phi**2/self._gamma * fac1 + 0.25*self._gamma*fac
            
            Teq4 = 0.75*self.Tint4(R)*f_visc + 0.75*self.Tirr4(R)*f_irr
            
            # Smooth the temperature limit as the model needs smooth pressure gradients
            return np.power(1/Teq4**2 + 1/self._Tmax**8, -0.125)


    # Use a passive disc as the initial temperature
    T1 = T_Profile(grid, np.full_like(grid.Rc, 1000), Lstar, 0, Mstar, gamma, phi, Tmax)

    # Iterate temperature to convergence.
    for iter in range(100):
        T0 = T1
        DM = DiscModel(grid, Mdot_gas, alpha, T0,  mu)
        tau = DM.Sigma_gas * kappa

        T1 = T_Profile(grid, tau, Lstar, Mdot_gas, Mstar, gamma, phi, Tmax)

        if np.all(np.abs(T1(grid.Rc) - T0(grid.Rc))/np.maximum(T1(grid.Rc), T0(grid.Rc)) < 1e-5):
            break
    else:
        raise ValueError("Temperature iteration failed to converge")

    return T1

def solar_org_comp(atom_abund=load_protosolar_abundances()):

# Oxygen / Nitrogen / Noble gases
    mol_abund = {
        'H2O' : atom_abund['O'] / 3,
        'CO'  : atom_abund['O'] / 6,
        'CO2' : atom_abund['O'] / 12,
    
        'N2'  : atom_abund['N'] * 0.45,
        'NH3' : atom_abund['N'] * 0.1,

        #'He' : atom_abund['He'], 
        'Ar' : atom_abund['Ar'], 
        'Kr' : 1.8e-9,
        'Xe' : 1.7e-10,
    }
    grain_abund = {
        'MgFeSiO4' : atom_abund['O'] / 12,
        'P'  : atom_abund['P'],
        'S'  : atom_abund['S'],
        'Na' : atom_abund['Na'],
        'K'  : atom_abund['K'],
    }
    gas_abund = {
        'H2' : atom_abund['H']/2,
        'He'  : atom_abund['He'],
    }
    dust = {n: g / np.sum(list(grain_abund.values())) for n,g in grain_abund.items()}
    gas  = {n: g / np.sum(list(gas_abund.values())) for n,g in gas_abund.items()}

# Count up the total carbon/oxygen abundance
    Ctot = 0
    for mol in mol_abund:
        atoms = atoms_in_molecule(mol)
        if 'C' in atoms:
            Ctot += mol_abund[mol] * atoms['C']

    for mol in grain_abund:
        atoms = atoms_in_molecule(mol)
        if 'C' in atoms:
            Ctot += grain_abund[mol] * atoms['C']

    # Put the rest into ethane / refractory carbon:
    C_org = atom_abund['C'] - Ctot 
    mol_abund['CH4'] = C_org * 0.25 / 1 
    mol_abund['C4H10'] = C_org * 0.75 / 4
    mol_abund['CH3OH'] = C_org * 0.0

    return mol_abund, atom_abund, dust, gas

def get_species_info(abund, atom_abund):
    """Load the molecular properties for the given abundances"""
    # Load the properties of the molecules
    species, _ = get_molecular_properties()
    s_map = { s.name : s for s in species}

    s_inc, abund_arr = [], []
    for mol in abund:
        s_inc.append(s_map[mol])
        abund_arr.append(abund[mol] / atom_abund['H'])

    return s_inc, np.array(abund_arr)

def set_env(mol_abund, atom_abund, St_alp=1.,Mdot_gas=1e-8, Md_Mg=0.1, radii = np.linspace(7.,9.,10), f_plansis= np.logspace(-6,-1,10), 
            gas={'H2':0.912,'He':0.087}, dust={'MgFeSiO4':3.235e-5}, init_m=5.0, mu=2.35, T0=150., m_star=1.4, beta_T=0.5):

    #Set up disc dynamics
    alp = 1e-4
    alpha = lambda R: alp
    grid = Grid(0.0005*Rau, 100*Rau, 512)
    T = lambda R: T0*(R/Rau)**(-beta_T)#create_temperature_profile(grid, L_star, Mdot_gas, alpha, mu=mu)
    T = np.where(T>900, 900*(Rau/R)*(T0/900)**(1/beta_T) ,T)

    DM = DiscModel(grid, Mdot_gas, alpha, T, mu)

    Mdot_dust = Mdot_gas * Md_Mg 
    Stokes = lambda R: St_alp * alp
    DM.compute_dust_surface_density(Mdot_dust, Stokes)
    print('alpha = {}, Stokes = {}, Mdot_gas = {}, Mdot_dust ={}, T[1au] = {}'.format(alp, alp*St_alp, Mdot_gas, Mdot_dust, T(grid.Rc)[np.argmin(abs(grid.Rc-Rau))]))

    #Set up chemistry

    species, abundances = get_species_info(mol_abund, atom_abund)
    #species, abundances = get_molecular_properties()
    DM.compute_chemistry(species, abundances )
    print([(spec.name) for spec in species], abundances)


    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    for g in gas:
        f_comp[g] = np.zeros(2)
    for d in dust:
        f_comp[d] = np.zeros(2)
    p_env = PlanetEnv(grid, alpha(grid.Rc), mu, m_star, gas, dust)

    #Set up iniital planet

    frac_gc = 0.01
    planet_ini = Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun, init_m*(frac_gc)*Mearth/Msun, f_comp, 8.5*Rau)

    return planet_ini, DM, p_env, T, f_plansis, radii
    #Set up sample data

def default_data():
    inp = 'default'
    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    print(gas)
    planet_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab, 
                                                         St_alp=1.,
                                                         Mdot_gas=1e-8,
                                                         Md_Mg=0.01, 
                                                         radii = np.linspace(12.,20.,2), 
                                                         f_plansis= np.logspace(-6,-1,3), 
                                                         gas=gas, 
                                                         dust=dust, 
                                                         init_m=5.0, 
                                                         T0=150.)
    store_data_range(planet_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii)
    pass

def data_sets(Mdots, Md_Mgs, St_alps, radiis, final_radius, T0, si, mdot_alp=(0.01,10), m_star=1.4, beta_T=0.5):
   
    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    for (mdot, radii) in zip(Mdots, radiis):
        for ma in mdot_alp:
            if si:
                inp = 'mdot_{}_st{}_{}K_Si'.format(mdot, ma, T0)
            else: 
                inp = 'mdot_{}_st{}_{}K_noSi'.format(mdot, ma, T0)
            print(inp)
            planet_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab, 
                                                         St_alp=ma,
                                                         Mdot_gas=mdot,
                                                         Md_Mg=0.01, 
                                                         radii = radii, 
                                                         f_plansis= np.logspace(-5,0,20), 
                                                         gas=gas, 
                                                         dust=dust, 
                                                         init_m=5.0, 
                                                         T0=T0,
                                                         m_star=m_star,
                                                         beta_T=beta_T
                                                         )
            store_data_range(planet_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=final_radius, si=si)
    for mdmg in Md_Mgs:
        for st_a in St_alps:
            if si:
                inp = 'dust2gas_{}_St2alp{}_{}K_Si'.format(mdmg, st_a, T0)
            else:
                inp = 'dust2gas_{}_St2alp{}_{}K_noSi'.format(mdmg, st_a, T0)
            print(inp)
            planet_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab, 
                                                         St_alp=st_a,
                                                         Mdot_gas=1e-8,
                                                         Md_Mg=mdmg, 
                                                         radii = radiis[1], 
                                                         f_plansis= np.logspace(-5,0,20), 
                                                         gas=gas, 
                                                         dust=dust, 
                                                         init_m=5.0, 
                                                         T0=T0,
                                                         m_star=m_star,
                                                         beta_T=beta_T)
            store_data_range(planet_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=final_radius, si=si)
    pass


def main():    
    #default_data()
    T0s = (100,125,150)
    m_as = (0.01, 10)
    si_sws = (True,)
    Mdots = (1e-7, 1e-8, 1e-9)
    Md_Mgs = (1e-2, 5e-2, 1e-1)
    St_alps = (1e-2, 1., 10.)
    radiis = [np.linspace(5.5, 20.5, 25), 
              np.linspace(5.5, 17.5, 25), 
              np.linspace(5.5, 12.5, 25)
             ]
    final_radius = 1e-2

    for si_sw in si_sws:
        for T0 in T0s:
            print('Si',si_sw)
            data_sets(Mdots, Md_Mgs, St_alps, radiis, final_radius, T0, si=si_sw, m_star=0.85, beta_T=0.5)
    pass

if '__main__'==__name__:
    main()
