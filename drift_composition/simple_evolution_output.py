from drift_composition.simple_setup import individual_run
from drift_composition.simple_reduction import Evolution
from drift_composition.constants import Rau

import itertools

import numpy as np
import pandas as pd


def data_nochem_evolution(
    run_index,
    Mdot, Md_Mg, St_alp, T0, m_star, beta_T, alpha,
    r_ini, m_ini, env2core,
    f_plansi, stall, pl_mode, dt_ini,
    Nt, final_radius, final_mass, final_time,
):
    """
    saving the no composition planet evolution

    Disc parameters: Mdot = mass accrtion rate,
                     Md_Mg = dust-to-gas accretion,
                     St_alp = Stokes number/visc. alpha,
                     T0 = temperature at 1au,
                     m_star = Star mass in solar masses,
                     beta_T = temperature profile exponent,
                     alpha = viscous alpha
    Initial planet: r_ini = initial radial position ,
                    m_ini= initial mass in earth masses,
                    env2core = fraction of initial envelope to core mass
    Planet evolution: f_plansi = planetesimal fraction compared relative to
                                 the dust mass
                    stall = final migration velocity in viscous velocities,
                    pl_mode = evolving planetesimal distribution if True, else
                              prop. to disc dust profile,
                    dt_ini = first intigration step in yrs,
    Termination critetia: Nt = maximum number of integration steps,
                    final_radius= last inner position in au,
                    final_mass= final mass in solar masses,
                    final_time= end of simulation in years,
    """

    p_evo, DM, p_env, T, nn = individual_run(
        Mdot, Md_Mg, St_alp,T0, m_star, beta_T, alpha,
        (),
        r_ini, m_ini, env2core,
        f_plansi, stall, pl_mode, dt_ini, Nt,
        final_radius, final_mass, final_time,
    )
    evo = Evolution(p_evo, nn)

    evo_dir = {
        "time": evo.time,
        "r_p": evo.rs,
        "m_core": evo.mcs,
        "m_env": evo.mgs,
        "temp_p": T(evo.rs),
    }

    param_dir = {
        "run_id": run_index,
        "Mdot_gas": Mdot,
        "gas2dust": Md_Mg,
        "Stokes": St_alp * alpha,
        "alpha": alpha,
        "T0": T0,
        "m_star": m_star,
        "beta_T": beta_T,
        "f_plansi": f_plansi,
        "stall": stall,
    }

    evo_id = {"run_id": run_index}

    result = pd.DataFrame(evo_dir)
    result = result.assign(**evo_id)

    # move run_id column to the left:
    result.insert(0, "run_id", result.pop("run_id"))

    return result, param_dir

def seq_output_grid(params_file):
    Nt = 7000
    dt_ini = 100
    params = np.loadtxt(params_file)

    run_index = itertools.count()
    params_row = []
    for param in params:
        run_id = next(run_index)
        df_evo, param_dir = data_nochem_evolution(
                run_id,
                Mdot    = param[0],
                Md_Mg   = param[1],
                St_alp  = param[2],
                T0      = param[3],
                m_star  = param[4],
                beta_T  = param[5],
                alpha   = param[6],
                r_ini   = param[7],
                m_ini   = param[8],
                env2core = param[9],
                f_plansi = param[10],
                stall   = param[11],
                pl_mode = False,
                dt_ini  = dt_ini,
                Nt      = Nt,
                final_radius = param[12],
                final_mass = param[13],
                final_time = param[14],
            )
        df_evo.to_hdf("evo_test.h5", key=f'run{run_id}')
        params_row.append(param_dir)
    df_param = pd.DataFrame(params_row)
    df_param.to_hdf("param_test.h5", key=f'run{run_id}')


def test_grid():
    St_alp = 1
    Mdot = 1e-8
    Md_Mg = 1e-2
    m_ini = 5.0
    env2core = 1e-3
    m_star = 1.0
    beta_T = 0.5
    alpha = 1e-3
    final_radius = 1e-2
    final_mass = 2e-3
    final_time = 1e7
    Nt = 1000
    dt_ini = 100
    pl_mode = False
    f_plansi = 1e-3
    r_inis = np.linspace(1, 10, 2)
    stall = 0.1

    run_index = itertools.count()
    for T0 in (100, 120):
        for r_ini in r_inis:
            run_id = next(run_index)
            df = data_nochem_evolution(
                run_id,
                Mdot,
                Md_Mg,
                St_alp,
                T0,
                m_star,
                beta_T,
                alpha,
                r_ini,
                m_ini,
                env2core,
                f_plansi,
                stall,
                pl_mode,
                dt_ini,
                Nt,
                final_radius,
                final_mass,
                final_time,
            )

            df.to_hdf("test.h5", key=f'run{run_id}')


seq_output_grid("drift_composition/test_parameters")
dat = pd.read_hdf("param_test.h5")
print(dat)
