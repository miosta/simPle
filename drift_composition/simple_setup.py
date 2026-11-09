from drift_composition.constants import Mearth, Rau, stefan_boltzmann, G_Msun, Msun, Lsun, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.simple_planet import Planet, PlanetEnv
from drift_composition.atoms import atoms_in_molecule, ELEMENT_MASS, load_protosolar_abundances, molecule_mass
from drift_composition.simple_reduction import Evolution, atom_mass, dust_to_gas, final_accretion, crit_mass, snapshot
import drift_composition.simple_reduction as red
import drift_composition.simple_planet as simp

from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np

def no_chem_comp():
    gas = {'H2':0.912,'He':0.087}
    dust = {'MgFeSiO4':3.235e-5}
    mol_abund = {}
    atom_abund = {}
    return mol_abund, atom_abund, dust, gas

def solar_frac_comp(frac_dir, atom_abund=load_protosolar_abundances()):
    # Oxygen / Nitrogen / Noble gases
    S_oxid = atom_abund['S'] * frac_dir['SO2'] * 2

    mol_abund = {
        'FeS' : atom_abund['S'] * frac_dir['FeS'],
        'H2S' : atom_abund['S'] * frac_dir['H2S'],
        'SO2' : atom_abund['S'] * frac_dir['SO2'],

        'H2O' : (atom_abund['O'] - S_oxid)/ frac_dir['H2O'],
        'CO'  : (atom_abund['O'] - S_oxid)/ frac_dir['CO'],
        'CO2' : (atom_abund['O'] - S_oxid)/ frac_dir['CO2'] / 2,
    
        'N2'  : atom_abund['N'] * frac_dir['N2'],
        'NH3' : atom_abund['N'] * frac_dir['NH3'],

        #'He' : atom_abund['He'], 
        'Ar' : atom_abund['Ar'], 
        'Kr' : 1.8e-9,
        'Xe' : 1.7e-10,
    }
    grain_abund = {
        'MgFeSiO4' : atom_abund['O'] / frac_dir['MgFeSiO4'],
        'P'  : atom_abund['P'],
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
    mol_abund['CH4'] = C_org * 1 * frac_dir['CH4']
    mol_abund['C2H6'] = C_org * 2 * frac_dir['C2H6']
    mol_abund['C4H10'] = C_org * 4 * frac_dir['C4H10']
    #mol_abund['CH3OH'] = C_org * 1 * frac_dir['CH3OH']

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

def set_env(abund, atom_ab, gas, dust, 
            St_alp, Mdot_gas, Md_Mg, T0, m_star, beta_T, alp,
            m_ini,r_ini, env2core,mu):

    #Set up disc dynamics
    alpha = lambda R: alp
    grid = Grid(0.0005*Rau, 100*Rau, 512)
    def T(R):
        '''Temperature profile with flat disc inside of 900K'''
        T = T0*(R/Rau)**(-beta_T)
        return np.where(T>900, 900*(Rau/R)*(T0/900)**(1/beta_T) ,T)

    DM = DiscModel(grid, Mdot_gas, alpha, T, mu)
    Mdot_dust = Mdot_gas * Md_Mg 
    Stokes = lambda R: St_alp * alp
    DM.compute_dust_surface_density(Mdot_dust, Stokes)
    
    print('alpha = {}, Stokes = {}, Mdot_gas = {}, Mdot_dust ={}, T[1au] = {}'.format(alp, alp*St_alp, Mdot_gas, Mdot_dust, T(grid.Rc)[np.argmin(abs(grid.Rc-Rau))]))

    #Set up chemistry
    species, abundances = get_species_info(abund, atom_ab)
    DM.compute_chemistry(species, abundances)
    print([(spec.name) for spec in species], abundances)

    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    for g in gas:
        f_comp[g] = np.zeros(2)
    for d in dust:
        f_comp[d] = np.zeros(2)
    p_env = PlanetEnv(grid, alpha(grid.Rc), mu, m_star, gas, dust)

    #Set up iniital planet
    init_m = m_ini*Mearth/Msun
    init_core = m_ini*Mearth/Msun*(1-env2core)
    init_env = m_ini*Mearth/Msun*(env2core)
    planet_ini = Planet(init_m, init_core, init_env, f_comp, r_ini*Rau)

    return planet_ini, DM, p_env, T

def individual_run(Mdot, Md_Mg, St_alp, T0, m_star, beta_T, alpha, frac_dir,
                    r_ini, m_ini, env2core,
                    f_plansi, stall, pl_mode, dt_ini,
                    Nt, final_radius, final_mass, final_time, exc=()):
    '''
    Sets up and runs an individual planet evolution

    Disc parameters: Mdot = mass accrtion rate, 
                     Md_Mg = dust-to-gas accretion,
                     St_alp = Stokes number/visc. alpha, 
                     T0 = temperature at 1au, 
                     m_star = Star mass in solar masses, 
                     beta_T = temperature profile exponent, 
                     alpha = viscous alpha
    Chemistry: frac_dir = directory of composition fractions, only Silicate+H2/He if empty
    Initial planet: r_ini = initial radial position , 
                    m_ini= initial mass in earth masses, 
                    env2core = fraction of initial envelope to core mass
    Planet evolution: f_plansi = planetesimal accretion proability 
                    stall = final migration velocity in viscous velocities, 
                    pl_mode = evolving planetesimal distribution if True, else prop. to disc dust profile,
                    dt_ini = first intigration step in yrs,
                    exc = list of species not added to the composition,
    Termination critetia: Nt = maximum number of integration steps, 
                    final_radius= last inner position in au, 
                    final_mass= final mass in solar masses, 
                    final_time= end of simulation in years,  
    '''
    if frac_dir == ():
        abund, atom_ab, dust, gas = no_chem_comp()
    else:
        abund, atom_ab, dust, gas = solar_frac_comp(frac_dir, atom_abund=load_protosolar_abundances()) #TODO:Add default dict
    p_ini, DM, p_env, T = set_env(abund,
                                  atom_ab,
                                  gas=gas, 
                                  dust=dust, 
                                  St_alp=St_alp,
                                  Mdot_gas=Mdot,
                                  Md_Mg=Md_Mg,
                                  T0=T0,
                                  m_star=m_star,
                                  beta_T=beta_T,
                                  alp=alpha,   
                                  m_ini = m_ini,
                                  r_ini = r_ini,
                                  env2core = env2core,
                                  mu=2.35)
    fin_r =final_radius*(1+9*np.random.rand())
    if pl_mode:
        planet_evo, nn = simp.full_evo_comp(p_ini, DM, p_env, T(p_env.grid.Rc),f_plansi, dt_ini, Nt, final_radius=fin_r, stall=stall, final_time=final_time, final_mass=final_mass) #TODO: Add planet scattering for singles
    else:
        planet_evo, nn = simp.full_evo_comp(p_ini, DM, p_env, T(p_env.grid.Rc),f_plansi, dt_ini, Nt, final_radius=fin_r, stall=stall, final_time=final_time, final_mass=final_mass)
    #evo = Evolution(planet_evo, nn, exclude=exc)
    return planet_evo, DM, p_env, T, nn
