from drift_composition.constants import Mearth, Msun, Rau, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.simple_planet import Planet, PlanetEnv
from drift_composition.atoms import atoms_in_molecule, ELEMENT_MASS, load_protosolar_abundances
from drift_composition.simple_reduction import Evolution, atom_mass, dust_to_gas
import drift_composition.simple_reduction as red
import drift_composition.simple_planet as dumb
import matplotlib.pyplot as plt
import numpy as np

def plot_planet_comp(e, comp='CO', title=''):

    fig, ax = plt.subplots()
    ax.plot(e.time, e.masses*Msun/Mearth, 'k-', label='total')
    ax.plot(e.time, e.mcs*Msun/Mearth, 'g:', label='core')
    ax.plot(e.time, e.mgs*Msun/Mearth, 'b-.', label='gas')
    ax.plot(e.time, e.f_comps[comp][0]*Msun/Mearth, 'c:', label='{} gas'.format(comp))
    ax.plot(e.time, e.f_comps[comp][1]*Msun/Mearth, 'c--', label='{} dust'.format(comp))
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("mass [M_Earth]")
    ax.legend()
    #ax.savefig('frag_masses.png')
    ax.text(0, 1, title)
    plt.show()
    pass

def plot_planet(e):

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.plot(e.time, e.masses*Msun/Mearth, 'k-', label='mass total')
    plt.plot(e.time, e.mgs*Msun/Mearth, 'k--', label='gas total')
    plt.plot(e.time, e.mcs*Msun/Mearth, 'k:', label='dust total')
    for name, c in zip(list(e.f_comps.keys()),colors):
        plt.plot(e.time, e.f_comps[name][0]*Msun/Mearth, '--', c=c, label=name)
        plt.plot(e.time, e.f_comps[name][1]*Msun/Mearth, ':', c=c)
    plt.ylim(1e-5,1e3)
    plt.ylabel(r'mass [$M_{\oplus}$]')
    plt.xlabel('time [yr]')

    plt.legend()
    plt.yscale('log')
    plt.show()

    plt.plot(e.rs, e.masses*Msun/Mearth, 'k-', label='mass total')
    plt.plot(e.rs, e.mgs*Msun/Mearth, 'k--', label='gas total')
    plt.plot(e.rs, e.mcs*Msun/Mearth, 'k:', label='dust total')
    for name, c in zip(list(e.f_comps.keys()),colors):
        plt.plot(e.rs, e.f_comps[name][0]*Msun/Mearth, '--', c=c, label=name)
        plt.plot(e.rs, e.f_comps[name][1]*Msun/Mearth, ':', c=c)
    plt.ylim(1e-5,1e3)
    plt.xlabel('radius [au]')
    plt.ylabel(r'mass [$M_{\oplus}$]')
    plt.legend()
    plt.yscale('log')
    plt.show()
    pass

def lecture_plot():
    Mdot_gas = 1e-8
    Mdot_dust = 1e-9
    Stokes = lambda R: 0.01

    T = lambda R: 150*(R/Rau)**-0.5
    alpha = lambda R: 1e-3

    species, abundances = get_molecular_properties()
    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    f_comp['H2'] = np.zeros(2)
    f_comp['Si'] = np.zeros(2)
    grid = Grid(0.1*Rau, 300*Rau, 512)

    p_env = PlanetEnv(grid, alpha(grid.Rc), 2.35, 1.0)

    dt = 1000
    Nt = 450
    f_plansi = 1e-1
    frac_gc = 0.1
    init_m  = 0.50

    ms2=np.zeros(Nt)
    mcs2=np.zeros(Nt)
    mgs2=np.zeros(Nt)
    rrs2=np.zeros(Nt)

    for r,mm in zip((15,30,40),(0.1,0.1,0.1,0.1,0.1,0.1)):
        DM = DiscModel(grid, Mdot_gas, alpha, T)
        DM.compute_dust_surface_density(Mdot_dust, Stokes)
        p_env = PlanetEnv(grid, 1e-3, 2.35, 1.0)
        planet = Planet(mm*init_m*Mearth/Msun, mm*init_m*(1-frac_gc)*Mearth/Msun, mm*init_m*(frac_gc)*Mearth/Msun, f_comp,r*Rau)
        m,mc,mg,mco_g,mco_d,rr = std_mig(planet, DM, p_env, T(grid.Rc),f_plansi, dt, Nt)
        ms2=np.vstack((ms2,m))
        mcs2=np.vstack((mcs2,mc))
        mgs2=np.vstack((mgs2,mg))
        rrs2=np.vstack((rrs2,rr))
    
        multi_plan(ms2[1:],mcs2[1:],mgs2[1:],rrs2[1:],Nt,dt,(15,30,40))
    pass

def plot_CtoO(e,atm1,atm2,solar = 1.):
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharex='col', sharey='row')

    atoms = (atm1, atm2)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for (x,a1,a2) in zip((e.time,e.rs),(ax1,ax2),(ax3,ax4)):
        a1.plot(x, e.masses*Msun/Mearth, 'k-', label='mass total')
        a1.plot(x, e.mgs*Msun/Mearth, 'k--', label='gas total')
        a1.plot(x, e.mcs*Msun/Mearth, 'k:', label='dust total')
        for atom,c in zip(atoms,colors):
            a1.plot(x, e.f_atms[atom][0]*Msun/Mearth, '--', c=c, label=atom)
            a1.plot(x, e.f_atms[atom][1]*Msun/Mearth, ':', c=c)

        a2.plot(x, (e.f_atms[atm1][0]/e.f_atms[atm2][0])*(ELEMENT_MASS[atm2]/ELEMENT_MASS[atm1])/solar, label='gas only')
        a2.plot(x, ((e.f_atms[atm1][0]+e.f_atms[atm1][1])/(e.f_atms[atm2][0]+e.f_atms[atm2][1])*(ELEMENT_MASS[atm2]/ELEMENT_MASS[atm1]))/solar, label='gas+dust')

    ax3.set_xlabel('time [yr]')
    ax4.set_xlabel('distance [au]')
    ax1.set_ylabel(r'mass [$M_{\oplus}$]')
    ax3.set_ylabel(r'{}/{}'.format(atm1,atm2))

    ax1.set_ylim(1e-3,1e3)
    ax2.set_ylim(1e-3,1e3)
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    ax1.legend()
    ax3.legend()
    plt.show()
    pass

def plot_CtoO_cut(e,atm1,atm2,cut_index,solar = 1.):
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharex='col', sharey='row')

    atoms = (atm1, atm2)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    mgss, mcss, comps, f_atms = dust_to_gas(e, cut_index)

    for (x,a1,a2) in zip((e.time,e.rs),(ax1,ax2),(ax3,ax4)):
        a1.plot(x, e.masses*Msun/Mearth, 'k-', label='mass total')
        a1.plot(x, mgss*Msun/Mearth, 'k--', label='gas total')
        a1.plot(x, e.mcs*Msun/Mearth, 'k:', alpha=0.5)
        a1.plot(x, mcss*Msun/Mearth, 'k:', label='dust total')
        for atom,c in zip(atoms,colors):
            a1.plot(x, f_atms[atom][0]*Msun/Mearth, '--', c=c, label=atom)
            a1.plot(x, f_atms[atom][1]*Msun/Mearth, ':', c=c)
            a1.plot(x, e.f_atms[atom][0]*Msun/Mearth, '--', c=c, alpha=0.5)
            a1.plot(x, e.f_atms[atom][1]*Msun/Mearth, ':', c=c, alpha=0.5)

        a2.plot(x[1:], (e.f_atms[atm1][0][1:]/e.f_atms[atm2][0][1:])*(ELEMENT_MASS[atm2]/ELEMENT_MASS[atm1])/solar, label='gas only')
        a2.plot(x[1:], (f_atms[atm1][0][1:]/f_atms[atm2][0][1:])*(ELEMENT_MASS[atm2]/ELEMENT_MASS[atm1])/solar, label='enriched_gas')
        a2.plot(x, ((f_atms[atm1][0]+f_atms[atm1][1])/(f_atms[atm2][0]+f_atms[atm2][1])*(ELEMENT_MASS[atm2]/ELEMENT_MASS[atm1]))/solar, label='gas+dust')

    ax3.set_xlabel('time [yr]')
    ax4.set_xlabel('distance [au]')
    ax1.set_ylabel(r'mass [$M_{\oplus}$]')
    ax3.set_ylabel(r'{}/{}'.format(atm1,atm2))

    ax1.set_ylim(1e-3,1e3)
    ax2.set_ylim(1e-3,1e3)
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    ax1.legend()
    ax3.legend()
    plt.show()
    pass

def plot_atoms(e):

    ms3 = e.masses

    #print(atom_mass(planet_evo[0].f_comp))

    atoms = list(ELEMENT_MASS.keys())

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.plot(e.rs, ms3*Msun/Mearth, 'k-', label='mass total')
    plt.plot(e.rs, e.mgs*Msun/Mearth, 'k--', label='gas total')
    plt.plot(e.rs, e.mcs*Msun/Mearth, 'k:', label='dust total')
    plt.plot(e.rs, e.f_atms['Si'][0]*Msun/Mearth, '--', c='b', label='Si')
    plt.plot(e.rs, e.f_atms['Si'][1]*Msun/Mearth, ':', c='b')
    for atom,c in zip(atoms,colors):
        plt.plot(e.rs, e.f_atms[atom][0]*Msun/Mearth, '--', c=c, label=atom)
        plt.plot(e.rs, e.f_atms[atom][1]*Msun/Mearth, ':', c=c)


    plt.xlabel('Radius [au]')
    plt.ylabel(r'mass [$M_{\oplus}$]')

    plt.ylim(1e-3,1e3)
    plt.yscale('log')
    plt.legend()
    plt.show()
    pass

def main():
    Mdot_gas = 1e-8
    Mdot_dust = 1e-9
    Stokes = lambda R: 0.001

    T = lambda R: 350*(R/Rau)**-0.5
    alpha = lambda R: 1e-3
    grid = Grid(0.1*Rau, 300*Rau, 512)

    DM = DiscModel(grid, Mdot_gas, alpha, T)
    DM.compute_dust_surface_density(Mdot_dust, Stokes)
    species, abundances = get_molecular_properties()
    DM.compute_chemistry(species, abundances )

    from drift_composition.simple_data import get_species_info, solar_org_comp
    abund, atom_ab, dust, gas = solar_org_comp(atom_abund=load_protosolar_abundances())
    species, abundances = get_species_info(abund, atom_ab)
    #species, abundances = get_molecular_properties()
    print([(spec.name) for spec in species], abundances)

    gas={'H2':0.912,'He':0.087} 
    dust={'MgFeSiO4':3.235e-5}
    p_env = PlanetEnv(grid, alpha(grid.Rc), 2.35, 1.4, gas, dust)

    SOLAR_OH = 0.0005242
    SOLAR_CO = 326./477.
    SOLAR_Z  = 0.0134

    dt = 5000
    Nt = 2000
    f_plansi = 1e-2

    frac_gc = 0.01
    init_m  = 5.0
    f_comp = dict(zip([spec.name for spec in species],np.zeros((len(species),2))))
    for g in gas:
        f_comp[g] = np.zeros(2)
    for d in dust:
        f_comp[d] = np.zeros(2)

    planet_ini = Planet(init_m*Mearth/Msun, init_m*(1-frac_gc)*Mearth/Msun, init_m*(frac_gc)*Mearth/Msun, f_comp, 4.5*Rau)

    planet_evo, nn = dumb.std_evo_comp(planet_ini, DM, p_env, T(grid.Rc),f_plansi, dt, Nt)
    evolution = Evolution(planet_evo, nn)
    plot_planet(evolution)
    plot_atoms(evolution)    
    evolution = Evolution(planet_evo, nn, exclude=dust)
    #red.store_data_range(planet_ini, DM, p_env, T)
    plot_CtoO_cut(evolution,'O','H', red.crit_mass(evolution), solar = SOLAR_OH)
    plot_CtoO_cut(evolution, 'C','O', red.crit_mass(evolution), solar= SOLAR_CO)
    pass

if '__main__'==__name__:
    main()
