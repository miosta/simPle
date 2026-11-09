import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import matplotlib as mpl

# sobol sequence is a way to sample the parameter space in a way that is uniform and efficient
# lets say you have 2 model parameters theta = [theta1, theta2]

# theta1 varies from 1e-4 to 1e0, so we actually want to sample log10(theta1)
# theta2 varies from 0 to 10, and let's say it's ok for this to be sampled linearly

def print_input(Mdot, Md_Mg, St_alp, T0, m_star, beta_T, alpha, r_ini, m_ini, env2core, f_plansi, stall, final_radius, final_mass, final_time):
    '''
    
    '''

    header="# Mdot 0, Md_Mg 1, St_alp 2, T0 3, m_star 4, beta_T 5, alpha 6, r_ini 7, m_ini 8, env2core 9, f_plansi 10, stall 11, final_radius 12, final_mass 13, final_time 14"

    n=0
    scale_dir =  {}
    var_dir =  {}
    var_list = []
    l_bounds = []
    u_bounds = []
    for (var,name) in zip((Mdot, Md_Mg, St_alp, T0, m_star, beta_T, alpha, r_ini, m_ini, 
                            env2core, f_plansi, stall, final_radius, final_mass, final_time),
                          ("Mdot", "Md_Mg", "St_alp","T0","m_star","beta_T","alpha","r_ini","m_ini", 
                            "env2core","f_plansi","stall","final_radius","final_mass","final_time")):
        if np.isscalar(var):
            scale_dir[name] = var
        else:
            var_list.append(name)
            if var[2]<1:
                l_bounds.append(var[0])
                u_bounds.append(var[1])
            elif var[2]>0:
                l_bounds.append(np.log10(var[0]))
                u_bounds.append(np.log10(var[1]))
            var_min = var[0]
            var_max = var[1]
            var_dir[name] = var[2]
            n +=1
        #else:
        #    print('Error: input parameter either scalar or tuple') 

    # sobol sequence
    M = 12 # 2^M samples
    D = n  # 2 parameters

    sampler = qmc.Sobol(d=D, scramble=True) # generate sobol sequence
    samples = sampler.random_base2(m=M) # generate M samples

    samples = qmc.scale(samples, l_bounds, u_bounds) # scale the samples to the bounds
    # ^^ use the above sample to generate your models

    f = open('input_list.txt', 'a')
    f.write(header)
    output_dir = scale_dir | var_dir
    out_dir_new = scale_dir | var_dir
    for sample in samples:
        for name, sam in zip(var_list,sample):
            if output_dir[name]>0:
                out_dir_new.update({name : 10**sam})
            elif output_dir[name]<1:
                out_dir_new.update({name : sam})
        line = [str(out_dir_new[n]) for n in ("Mdot", "Md_Mg", "St_alp","T0","m_star","beta_T","alpha","r_ini","m_ini", 
                            "env2core","f_plansi","stall","final_radius","final_mass","final_time")]
        print(line)
        f.write('  '.join(line))
        f.write('\n')
    f.close()
    plt.scatter(samples[:,0], samples[:,2], s=2, c=samples[:,1],cmap='plasma')
    plt.xlabel(r'$\log_{10}(\theta_1)$')
    plt.ylabel(r'$\theta_2$')
    plt.tight_layout()
    plt.show()

print_input(1e-8, 1e-2, (0.01,10,1), 100, 1, 0.5, 1e-3, (5,10,0), 4, 0.01, (1e-5,1e-2,1), 0.1, 0.01, 3e-3, 1e7)
