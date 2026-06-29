import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from drift_composition.constants import Rau
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.atoms import (
    load_protosolar_abundances,
    atoms_in_molecule, molecule_mass
)
from drift_composition.constants import Mearth, Msun
from drift_composition.simple_planet import PlanetEnv, Planet, std_evo_comp
from drift_composition.simple_reduction import Evolution
import drift_composition.simple_graphs as sg

atom_abund = load_protosolar_abundances()

Mdot_gas = 1e-8          #Mass accretion rate of gas
dust_flux_factor = 1    #Dust enhancement due to radial drift
T0    = 200              #Temperature at 1AU, note: we are working with F star
dt = 1000       #initial timestep in yr
Nt = 2000       #max number of timesteps
f_plansi = 1e-2 #fraction of planitesimal/dust

frac_gc = 0.01  #initial gas/core fraction of the planet
init_m  = 5.0   #initial planet mass in M_earth
init_r  = 8.0  #initial planet distance in au
m_star  = 1.4   #solar_masses

Stokes = lambda R: 0.001  #Stokes number of the dust in the disc
alpha = lambda R: 1e-3   #viscous turbulent alpha coefficient in the disc
grid = Grid(0.005*Rau, 300*Rau, 512) #the radial logarithmic grid
T = lambda R: T0 * (R/Rau)**(-0.5)  #the temperature function

plot_specs =['H2O', 'CO', 'CO2', 'C2H6', 'C4H10']

def create_chemistry():

    atom_abund = load_protosolar_abundances()

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
        'MgFeSiO4' : atom_abund['O'] / 12 ,
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

    
    mean_mol_wgt = (atom_abund['H'] + 4*atom_abund['He']) /  (0.5*atom_abund['H'] + atom_abund['He'])

    return atom_abund, mol_abund, grain_abund, mean_mol_wgt#, dust, gas

def species_label(species):
    if species in ['C4H10']:
        return 'C_grain'
    else:
        return species

def get_species_info(abund):
    """Load the molecular properties for the given abundances"""
    # Load the properties of the molecules
    species, _ = get_molecular_properties()
    s_map = { s.name : s for s in species}

    s_inc, abund_arr = [], []
    for mol in abund:
        s_inc.append(s_map[mol])
        abund_arr.append(abund[mol] / atom_abund['H'])

    return s_inc, np.array(abund_arr)

def create_disc(grid, Mdot_gas, alpha, T, Mdot_dust, Stokes):

    atom_abund, mol_abund, grain_abund, mu = create_chemistry()
    DM = DiscModel(grid, Mdot_gas, alpha, T, mu)

    DM.compute_dust_surface_density(Mdot_dust, Stokes)

    species, abundances = get_species_info(mol_abund) #Use to initialize specific chemistry
    DM.compute_chemistry(species, abundances, d2g=Mdot_dust/Mdot_gas)
    mass_dir = dict(zip([spec.name for spec in species],[spec.mass_amu for spec in species]))
    print('Considered molecules with atomic masses in the system:', mass_dir)
    return DM, species, mass_dir, grain_abund

def create_env(grid, alpha, m_star):
    #calculating the local disc environment around the planet
    atom_abund, mol_abund, grain_abund, mu = create_chemistry()
    gas = {'H2':atom_abund['H']/2,'He':atom_abund['He']} #components that make up the gas disc
    dust = grain_abund                                   #components that make up the solids
    dust = {n: g / np.sum(list(grain_abund.values())) for n,g in grain_abund.items()}
    gas  = {n: g / np.sum(list(gas.values())) for n,g in gas.items()}
    p_env = PlanetEnv(grid, alpha(grid.Rc), mu, m_star, gas, dust) #create local disc properties
    return p_env

def init_planet(init_m, frac_gc, init_r, species, gas, dust):
    #initializing the planets chemical composition as zeros
    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    for g in gas:
        f_comp[g] = np.zeros(2)
    for d in dust:
        f_comp[d] = np.zeros(2)

    #intial planet object
    planet_ini = Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun, init_m*(frac_gc)*Mearth/Msun, f_comp, init_r*Rau)
    return planet_ini

def evolve_planet(planet_ini, DM, p_env, T, f_plansi, dt, Nt, final_radius):
    #evolving the planet in time, planet_evo is a list of Planet
    planet_evo, nn = std_evo_comp(planet_ini, DM, p_env, T(p_env.grid.Rc),f_plansi, dt, Nt, final_radius)
    e = Evolution(planet_evo, nn, exclude=()) #rearrange the data for easy plotting
    return planet_evo, e, nn

def model_run(grid, Mdot_gas, alpha, T, Mdot_dust, Stokes, f_plansi, dt, Nt, final_radius, init_r=10.):
    DM, species,mass_dir,grain_abund = create_disc(grid, Mdot_gas, alpha, T, Mdot_dust, Stokes)
    p_env                = create_env(grid, alpha, m_star)
    planet_in            = init_planet(init_m, frac_gc, init_r, species, p_env.gas, p_env.dust)
    p_evo, evo, nn       = evolve_planet(planet_in, DM, p_env, T, f_plansi, dt, Nt, final_radius)
    return DM, p_env, p_evo, evo, nn, grain_abund, mass_dir

def subplot_disc_molH(grid, DM, ax, c, specs):
    columns = DM.compute_molecular_column()

    idx = columns['CO'][:,0] > 0.5 * columns['CO'][:,1]

    for spec in specs:
        ax.loglog(grid.Rc/Rau, 0.5*columns[spec][:,0]/columns['H2'][:,0], 
                         label=f'{species_label(spec)} (gas)',  ls='--', c=c)
        ax.loglog(grid.Rc/Rau, 0.5*columns[spec][:,1]/columns['H2'][:,0], 
                 label=f'{species_label(spec)} (ice)',c=c)

    ax.set_ylabel('Abundance (X/H)')
    ax.legend(ncol=4)
    ax.set_ylim(1e-5, 0.1)
    pass


def subplot_disc_CO(grid, DM, grain_abund, ax, c):
    columns = DM.compute_elemental_column(dust=grain_abund)
    C2O = columns['C']/columns['O']

    columns =  DM.compute_elemental_column() 
    C2O_ice = columns['C']/columns['O']

    plt.figure()
    ax.semilogx(grid.Rc[:]/Rau, C2O[:,0], label='gas')
    ax.semilogx(grid.Rc[:]/Rau, C2O[:,1], label='ice+dust')
    ax.semilogx(grid.Rc/Rau, C2O_ice[:,1], label='ice only')

    ax.set_xlabel('Radius [AU]')
    ax.set_ylabel('C/O ratio')
    ax.set_ylim(-0.1, 2)
    ax.legend()
    pass

#def subplot_p_molH(e, ax, c, specs, mass_dir):
#    for spec in specs:
#        ax.plot(e.rs[1:], e.f_comps[spec][0][1:]/e.f_comps['H2'][0][1:]*(1/mass_dir[spec]), '-', alpha=0.5,lw=5, c=c, label='planet gas')

def main():
    fig, (ax1,ax2) = plt.subplots(2,1)

    DM, p_env, p_evo, evo, nn, grain_abund = model_run(grid, Mdot_gas, alpha, T, Mdot_gas*0.01, Stokes, f_plansi, dt, Nt, final_radius=0.005)
    subplot_disc_molH(grid, DM, ax1, 'k', plot_specs)
    subplot_disc_CO(grid, DM, grain_abund, ax2, 'k')

    plt.show()
    pass
    
