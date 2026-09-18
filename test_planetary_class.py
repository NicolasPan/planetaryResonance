
from planetary_class import planetary_resonance

#%%

# Instantiate resonance object (runs Fortran compilation & execution automatically)
# res = planetary_resonance(star_mass = 1.0,
#                           a_inn = 1.0,
#                           e_inn = 0.2,
#                           e_out = 0.3,
#                           i_inn = 0.0,
#                           i_out = 0.0,
#                           nodo_inn = 0.0,
#                           nodo_out = 0.0,
#                           varpi_inn = 0.0,
#                           varpi_out = 120.0,
#                           mass_inn = 0.001,
#                           mass_out = 0.001,
#                           k2 = 3,
#                           k1 = 2)

#%%

# res.equilibrium_points()


#%%

# res.periods()

#%%

# res.width()

#%%

# import matplotlib.pyplot as plt

# df = res.sigmas()

# plt.figure(figsize=(7, 4), dpi=300)
# plt.plot(df['sigma_deg'], df['R_minus_Rmin'], color='navy', lw=1.5)
# plt.xlabel(r'$\sigma\ [^\circ]$', fontsize=12)
# plt.ylabel(r'$R - R_{\mathrm{min}}$', fontsize=12)
# plt.title('Disturbing Function Potential', fontsize=13)
# plt.xlim(0, 360)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

#%%

# res.plot_hamiltonian(num_levels=25)

#%%

# import numpy as np

# varpis = np.linspace(0, 360, 37)

# for varpi in varpis:
#     res = planetary_resonance(star_mass = 1.0,
#                               a_inn = 1.0,
#                               e_inn = 0.2,
#                               e_out = 0.3,
#                               i_inn = 0.0,
#                               i_out = 0.0,
#                               nodo_inn = 0.0,
#                               nodo_out = 0.0,
#                               varpi_inn = 0.0,
#                               varpi_out = varpi,
#                               mass_inn = 0.001,
#                               mass_out = 0.001,
#                               k2 = 3,
#                               k1 = 2)
    
#     res.plot_hamiltonian(num_levels=25)

#%%

import numpy as np
import matplotlib.pyplot as plt

varpis = np.linspace(0, 360, 37)

for varpi in varpis:
    # 1. Initialize resonance model
    res = planetary_resonance(
        star_mass=1.0,
        a_inn=1.0,
        e_inn=0.2,
        e_out=0.3,
        i_inn=0.0,
        i_out=0.0,
        nodo_inn=0.0,
        nodo_out=0.0,
        varpi_inn=0.0,
        varpi_out=varpi,
        mass_inn=0.001,
        mass_out=0.001,
        k2=2,
        k1=1
    )

    # 2. Extract plot data
    data = res.plot_hamiltonian(num_levels=25, plot=False)
    SIGMA, A, H = data['SIGMA'], data['A'], data['H']

    # 3. Create figure
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.contour(SIGMA, A, H, levels=25, cmap='Blues')
    
    # 4. Extract stable equilibrium points & calculate delta_sigma
    stable_sigmas = res.stable_eq()
    
    delta_sig_str = ""
    if len(stable_sigmas) == 2:
        delta_sig = abs(stable_sigmas[1] - stable_sigmas[0])
        delta_sig_str = rf', $\Delta\sigma={delta_sig:.1f}^\circ$'
    
    if stable_sigmas:
        a_out = getattr(res, 'a_out', res.a_inn * (res.k2 / res.k1) ** (2 / 3))
        ax.scatter(
            stable_sigmas, 
            [a_out] * len(stable_sigmas), 
            marker='x', 
            color='red', 
            s=90, 
            linewidths=2, 
            zorder=5, 
            label=r'Stable Eq. ($a_{\mathrm{out}}$)'
        )
        ax.legend(loc='upper right')

    # Formatting
    ax.set_xlabel(r'$\sigma\ [\mathrm{deg}]$', fontsize=14)
    ax.set_ylabel(r'$a\ [\mathrm{au}]$', fontsize=14)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    
    delta_varpi = abs(res.varpi_inn - res.varpi_out)
    
    # Title conditionally includes Delta Sigma if two stable points exist
    ax.set_title(
        rf'Resonance {res.k2}:{res.k1} '
        rf'($e_\mathrm{{in}}={res.e_inn}$, $e_\mathrm{{out}}={res.e_out}$, '
        rf'$\Delta\varpi={delta_varpi:.1f}^\circ${delta_sig_str})'
    )
    
    plt.tight_layout()
    plt.show()
