from drift_composition.constants import Mearth, Rau, stefan_boltzmann, G_Msun, Msun, Lsun, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.simple_planet import PlanetEnv
from drift_composition.simple_multis import Planet, Planetesimals
from drift_composition.atoms import atoms_in_molecule, ELEMENT_MASS, load_protosolar_abundances, molecule_mass
from drift_composition.simple_reduction import Evolution, atom_mass, dust_to_gas, final_accretion, crit_mass
import drift_composition.simple_reduction as red
import drift_composition.simple_multis as simp

from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np


def store_data_multis(p_inis, DM, p_env, T, inp='test', f_plansis=np.logspace(-3,-2.3,5), radii = np.linspace(7.,17.,10), final_radius = 1e-3, final_m= 2e-3, si= True, f_mig = 0.1):

    dt_ini = 500
    #dts = np.linspace(500,1000,10)
    Nt = 10000
    header = "#id, mini, mcini, mgini, rini, plans, rfin, mfin, mcfin, mgfin, mgH, mgO, mgC, mdH, mdO, mdC, m10, mg10, mc10, mgH10, mgO10, mgC10, mdH10, mdO10, mdC10, yr \n"
    f = open('data_multi/{}.txt'.format(inp), 'w')
    f.write(header)
    f.close()

    i__i = 0
    for fp in f_plansis:
        f = open('data_multi/{}.txt'.format(inp), 'a')
        for rad in radii:
            i__i += 10
            fin_r =final_radius*(1+9*np.random.rand())
            p_inis[0].dist = rad*Rau
            #p_inis[1].dist = rad*Rau*2**(2/3)

            pl = Planetesimals(p_env.grid, r=10, e0=1e-4, fSigma=fp)

            multi_evo, nn, _ = simp.multi_evo_comp(p_inis, DM, p_env, T(p_env.grid.Rc),pl, dt_ini, Nt, final_radius=fin_r, final_mass= final_m, final_time= 1e7, f_mig=f_mig)
            for ii, planet_evo in enumerate(multi_evo):
                planet_fin = planet_evo[-1]
            #print(planet_fin.dist/Rau , fin_r)
                if si:
                    exc = ()
                else:
                    exc = list(p_env.dust.keys())
                #print('Si',si)
                evo = Evolution(planet_evo, nn, exclude=exc)
                fin_mass, fin_mc, fin_mg, fin_comp, fin_atom = final_accretion(evo, crit_mass(evo, threshhold_mass_fraction=0.5))
                data = (str(ii+i__i),
                        str(p_inis[ii].mass), 
                        str(p_inis[ii].mc), 
                        str(p_inis[ii].mg), 
                        str(p_inis[ii].dist/Rau),
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
                        str(planet_fin.time)
                        )
                #print(data)
                f.write('  '.join(data))
                f.write('\n')
                #print(planet_ini.dist/Rau ,planet_fin.dist/Rau)
        f.close()
    pass

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

def set_env(mol_abund, atom_abund, St_alp=1.,Mdot_gas=1e-8, Md_Mg=0.1, radii = np.linspace(7.,9.,10), f_plansis= np.logspace(-6,-1,10), gas={'H2':0.912,'He':0.087}, dust={'MgFeSiO4':3.235e-5}, init_m=5.0, mu=2.35, T0=150., m_star=1.4):

    #Set up disc dynamics
    alp = 1e-3
    alpha = lambda R: alp
    grid = Grid(0.0005*Rau, 100*Rau, 512)
    T = lambda R: T0*(R/Rau)**(-0.5)#create_temperature_profile(grid, L_star, Mdot_gas, alpha, mu=mu)

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
    planets_ini = [Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun, init_m*(frac_gc)*Mearth/Msun, f_comp, 8.5*Rau),]
                   #Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun, init_m*(frac_gc)*Mearth/Msun, f_comp, 8.5*Rau*2**(2/3))]

    return planets_ini, DM, p_env, T, f_plansis, radii
    #Set up sample data

def default_data():
    inp = 'default_plansi_300_15i'
    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    print(gas)
    planets_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab, 
                                                         St_alp=1.,
                                                         Mdot_gas=1e-8,
                                                         Md_Mg=0.01, 
                                                         radii = np.linspace(4.,7.,20), 
                                                         f_plansis= np.logspace(-4,-2,10), 
                                                         gas=gas, 
                                                         dust=dust, 
                                                         init_m=3.0, 
                                                         T0=150.)
    store_data_multis(planets_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=1e-2, final_m= 1.5e-3, si=True, f_mig=0.1)
    pass

def data_sets(Mdots, Md_Mgs, St_alps, radiis, final_radius, T0, si, fmig, mdot_alp=(0.01,10)):
   
    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    for (mdot, radii) in zip(Mdots, radiis):
        for ma in mdot_alp:
            if si:
                inp = 'mdot_{}_st{}_{}K_Si'.format(mdot, ma, T0)
            else: 
                inp = 'mdot_{}_st{}_{}K_noSi'.format(mdot, ma, T0)
            print(inp)
            planets_ini, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab, 
                                                         St_alp=ma,
                                                         Mdot_gas=mdot,
                                                         Md_Mg=0.01, 
                                                         radii = radii, 
                                                         f_plansis= np.logspace(-3,-2.3,5), 
                                                         gas=gas, 
                                                         dust=dust, 
                                                         init_m=1.0, 
                                                         T0=T0)
            store_data_multis(planets_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=final_radius, si=si, f_mig=fmig)
    for mdmg in Md_Mgs:
        for st_a in St_alps:
            if si:
                inp = 'dust2gas_{}_St2alp{}_{}K_Si'.format(mdmg, st_a, T0)
            else:
                inp = 'dust2gas_{}_St2alp{}_{}K_noSi'.format(mdmg, st_a, T0)
            print(inp)
            planets_inis, DM, p_env, T, f_plansis, radii = set_env(abund,
                                                         atom_ab, 
                                                         St_alp=st_a,
                                                         Mdot_gas=1e-8,
                                                         Md_Mg=mdmg, 
                                                         radii = radiis[1], 
                                                         f_plansis= np.logspace(-3,-2.3,5), 
                                                         gas=gas, 
                                                         dust=dust, 
                                                         init_m=1.0, 
                                                         T0=T0)
            store_data_multis(planets_ini, DM, p_env, T, inp = inp, f_plansis=f_plansis, radii = radii, final_radius=final_radius, si=si, f_mig=fmig)
    pass


def main():    
    default_data()
    T0s = (125,150,200)
    m_as = (0.01, 10)
    si_sws = (True,)
    Mdots = (1e-7, 1e-8)
    Md_Mgs = (1e-2,)
    St_alps = (1e-2, 1.)
    radiis = [np.linspace(10.0, 20.5, 10), 
              np.linspace(7.5, 17.5, 10), 
              np.linspace(5.5, 12.5, 10)
             ]
    final_radius = 1e-2
    fmig = (0.12, 0.11, 0.1, 0.09, 0.08, 0.07)

    #for si_sw in si_sws:
    #    for T0 in T0s:
    #        for fm in fmig:
    #            print('Si',si_sw)
    #            data_sets(Mdots, Md_Mgs, St_alps, radiis, final_radius, T0, fmig=fm, si=si_sw)
    pass

if '__main__'==__name__:
    main()
