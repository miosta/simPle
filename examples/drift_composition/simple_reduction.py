from drift_composition.constants import Mearth, Msun, Rau, yr
from drift_composition.grid import Grid
from drift_composition.disc import DiscModel
from drift_composition.molecule import get_molecular_properties
from drift_composition.simple_planet import Planet, PlanetEnv
from drift_composition.atoms import atoms_in_molecule, ELEMENT_MASS
import drift_composition.simple_planet as dumb
import matplotlib.pyplot as plt
import numpy as np

class Evolution:
    def __init__(self, planet_evo, Nt, exclude=()):
        self.planet_evo = planet_evo
        self.Nt = Nt
        self.exclude = exclude
        self.masses = np.array([p.mass for p in planet_evo])
        self.mgs    = np.array([p.mg for p in planet_evo])
        self.mcs    = np.array([p.mc for p in planet_evo])
        self.rs     = np.array([p.dist for p in planet_evo])/Rau
        self.time   = np.array([p.time for p in planet_evo])
    @property
    def f_comps(self):
        names = list(self.planet_evo[0].f_comp.keys())
        comps = dict(zip(names,np.zeros((len(names),2,len(range(self.Nt))))))
        for name in names:
            comps[name][0] = np.array([p.f_comp[name][0] for p in self.planet_evo])
            comps[name][1] = np.array([p.f_comp[name][1] for p in self.planet_evo])
        return comps
    @property
    def f_atms(self):
        at_evo = np.array([atom_mass(p.f_comp,self.exclude) for p in self.planet_evo])
        atoms = list(ELEMENT_MASS.keys())
        abun = dict(zip(atoms,np.zeros((len(atoms),2,len(range(self.Nt))))))
        for atom in atoms:
            abun[atom][0] = np.array([am[atom][0] for am in at_evo])
            abun[atom][1] = np.array([am[atom][1] for am in at_evo])
        return abun        

def atom_mass(f_comp, exclude=()):
    f_atom = dict(zip(list(ELEMENT_MASS.keys()),np.zeros((len(list(ELEMENT_MASS.keys())),2))))
    all_names = list(f_comp.keys())
    mol_names = [nam for nam in all_names if nam not in exclude]
    for mol_n in mol_names:
        n_atoms = atoms_in_molecule(mol_n)
        mass_mol = np.sum([n_atoms[nam]*ELEMENT_MASS[nam] for nam in list(n_atoms.keys())])
        for nam in list(n_atoms.keys()):
            f_atom[nam][0] += f_comp[mol_n][0]/mass_mol*ELEMENT_MASS[nam] * n_atoms[nam]
            f_atom[nam][1] += f_comp[mol_n][1]/mass_mol*ELEMENT_MASS[nam] * n_atoms[nam]
    return f_atom

def dust_to_gas(evo, cut_index):
        core_mass = evo.mcs[cut_index]
        new_mcs = np.where(evo.mcs>core_mass,core_mass,evo.mcs)
        new_mgs = evo.mgs + evo.mcs - new_mcs
        all_names = list(evo.f_comps.keys())
        names = [na for na in all_names if na not in evo.exclude]
        print(names)
        new_comps = evo.f_comps
        for name in names:
            solid = evo.f_comps[name][1][cut_index]
            new_comps[name][1][cut_index+1:] = solid
            new_comps[name][0] += evo.f_comps[name][1] - new_comps[name][1]
        new_atoms = evo.f_atms
        atoms = list(evo.f_atms.keys())
        for at in atoms:
            solid = evo.f_atms[at][1][cut_index]       
            new_atoms[at][1][cut_index+1:] = solid
            new_atoms[at][0] += evo.f_atms[at][1] - new_atoms[at][1]
        return new_mgs, new_mcs, new_comps, new_atoms

def final_accretion(evo, cut_index):
        fin_mass = evo.masses[-1] - evo.masses[cut_index]
        fin_mc = evo.mcs[-1] - evo.mcs[cut_index]
        fin_mg = evo.mgs[-1] - evo.mgs[cut_index]
        names = list(evo.f_comps.keys())
        fin_comp = dict(zip(names,np.zeros((len(names),2))))
        for name in names:
            fin_comp[name][1] = evo.f_comps[name][1][-1] - evo.f_comps[name][1][cut_index]
            fin_comp[name][0] = evo.f_comps[name][0][-1] - evo.f_comps[name][0][cut_index]
        for name in evo.exclude:
            fin_comp[name][1] = 0.0
            fin_comp[name][0] = 0.0
        atoms = list(evo.f_atms.keys())
        fin_atom = atom_mass(fin_comp)
        return fin_mass, fin_mc, fin_mg, fin_comp, fin_atom

def run_away(evo):
    return np.argmax(evo.mgs>evo.mcs)

def final_comp(evo):
    return np.argmax(evo.time>evo.time[-1]*0.9)

def crit_mass(evo, threshhold_mass_fraction=0.9):
    return np.argmax(evo.masses/evo.masses[-1]>threshhold_mass_fraction)

def store_test_data(planet_ini, DM, p_env, T, inp='test'):
    f_plansis = np.logspace(-5,0,5)
    radii = np.linspace(7.,9.,5)
    dts = np.linspace(425,625,10)
    Nt = 1000
    header = "#mini, mcini, mgini, rini, plans, rfin, mfin, mcfin, mgfin, mgH, mgO, mgC, mdH, mdO, mdC, m10, mg10, mc10, mgH10, mgO10, mgC10, mdH10, mdO10, mdC10, yr \n"
    f = open('{}.txt'.format(inp), 'w')
    f.write(header)

    for fp in f_plansis:
        for (rad, dt) in zip(radii,dts):
            planet_ini.dist = rad*Rau
            planet_evo, nn = dumb.std_evo_comp(planet_ini, DM, p_env, T(p_env.grid.Rc),fp, dt, Nt)
            planet_fin = planet_evo[-1]
            evo = Evolution(planet_evo, dt, exclude=list(p_env.dust.keys()))
            fin_mass, fin_mc, fin_mg, fin_comp, fin_atom = final_accretion(evo, crit_mass(evo))
            data = (str(planet_ini.mass), 
                    str(planet_ini.mc), 
                    str(planet_ini.mg), 
                    str(planet_ini.dist/Rau),
                    str(fp),
                    str(planet_fin.dist/Rau),
                    str(planet_fin.mass), 
                    str(planet_fin.mc), 
                    str(planet_fin.mg), 
                    str(atom_mass(planet_fin.f_comp,exclude=p_env.dust)['H'][0]), 
                    str(atom_mass(planet_fin.f_comp,exclude=p_env.dust)['O'][0]), 
                    str(atom_mass(planet_fin.f_comp,exclude=p_env.dust)['C'][0]),
                    str(atom_mass(planet_fin.f_comp,exclude=p_env.dust)['H'][1]), 
                    str(atom_mass(planet_fin.f_comp,exclude=p_env.dust)['O'][1]), 
                    str(atom_mass(planet_fin.f_comp,exclude=p_env.dust)['C'][1]),
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
            f.write('  '.join(data))
            f.write('\n')
    f.close()
    pass
