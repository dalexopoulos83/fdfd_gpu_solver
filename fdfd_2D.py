#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:14:29 2022

@author: thkam, dimitrisalexopoulos
"""

import numpy as np
import matplotlib.pyplot as plt
from fdfd_2D_solver import yee_grid, graded_edges, calc_dist_e


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.serif": ["Tahoma"],
})

case = 3

# Run calc_orth_vectors/calc_eavg on the GPU (PyCUDA); False = original CPU path
use_gpu = True

# Non-uniform (graded) mesh, denser around the waveguiding region; False = uniform grid
nonuniform = True
grid_boost = 6.0   # point density near the waveguiding region is ~(1 + grid_boost)x the baseline

dPML_cells = 5     # PML depth in cells of the equivalent uniform grid

# What to run
run_optimization = False   # Bayesian optimization of the HPW geometry (case 3 only, needs scikit-optimize)
run_convergence = True     # neff vs N for all four averaging schemes (otherwise only tensor averaging runs)
run_mode_plots = True      # structure + mode field plots

# Bayesian optimization settings (same objective and bounds as fdfd_optimization_benchmarking)
opt_N = 100                # grid size used for every evaluation
opt_n_calls = 60           # total objective evaluations
opt_n_initial = 15         # random evaluations before the Gaussian-process model takes over
opt_seed = 0
opt_bounds = {
    'w_wg': (0.30, 0.70),
    'h_si': (0.20, 0.35),
    'h_spacer': (0.02, 0.10),
    'h_metal': (0.05, 0.20),
    'n_target': (1.50, 3.00),
}
metals = {  # complex refractive index at 1.55 um
    'Silver (Ag)': 0.1453+11.3587j,
    'Gold (Au)': 0.559+9.81j,
    'Copper (Cu)': 0.606+8.26j,
    'Aluminum (Al)': 1.44+16.0j,
}
n2_si = 4.5e-18            # Si nonlinear index (m^2/W)
rho_si = 1.27              # Si nonlinear tensor coefficient

# Mode plot settings
N_plot = 150
ploting_mode = 0

if case == 1:
    #Case 1
    # This parameters are for step index fiber
    n1 = 1.45
    n2 = 0
    n3 = 0
    n_rect = 1.00
    Lx = 15.0
    Ly = 15.0
    outer_radius = 3
    midle_radius = 0
    inner_radius = 0
    di = 0
    target_neff = 1.438604
    l = 1.50
    ntarget = n1
    NN = np.arange(100, 120, 10)

elif case == 2:
    #Case 2
    # This are for Microstractured Optical Fiber
    n1 = 1.45
    n2 = 1.42
    n3 = 1.0
    n_rect = 1.42
    Lx = 20.0
    Ly = 20.0
    outer_radius = 2
    midle_radius = 0
    inner_radius = 2
    di = 5
    target_neff = 1.4353607
    l = 1.50
    ntarget = n1
    NN = np.arange(100, 120, 10)

elif case == 3:
    #Case 3
    # Rectangular hybrid plasmonic waveguide (HPW) on SOI:
    # Si core / SiO2 spacer / metal cap on a SiO2 substrate, in air.
    # Winning geometry of fdfd_optimization_benchmarking (Fiber_Opt_20260207_173459).
    n_si = 3.47
    n_sio2 = 1.444
    n_air = 1.0
    metal = 'Silver (Ag)'
    Lx = 2.0
    Ly = 2.0
    hpw = {'w_wg': 0.4982, 'h_si': 0.3469, 'h_spacer': 0.0920, 'h_metal': 0.2000}
    target_neff = 2.604379
    l = 1.55
    ntarget = 2.7198
    NN = np.arange(100, 120, 10)


nmodes = 2
C0 = 3e8
M0 = 1.257e-06
omega = 2 * np.pi * C0 / l


# =============================================================================
# Geometry
# =============================================================================

def build_fibers():
    fiber_dict = [
        { 'type' : 'rectangle',
          'x1' : -np.inf,
          'x2' : +np.inf,
          'y1' : -np.inf,
          'y2' : +np.inf,
          'e_value_inside' : n_rect ** 2.0,
        },
        { 'type' : 'disk',
          'x0' : 0.0,
          'y0' : 0.0,
          'radius' : outer_radius,
          'e_value_inside' : n1 ** 2.0,
      },
      { 'type' : 'midle_disk',
        'x0' : 0.0,
        'y0' : 0.0,
        'midle_radius' : midle_radius,
        'e_value_inside' : n2 ** 2.0,
      },
      { 'type' : 'inner_disk',
        'theta' : [15, 75, 135, 195, 255, 315],
        'di' : di,  # This is Λ
        'inner_radius' : inner_radius,
        'e_value_inside' : n3 ** 2.0,
      }
    ]

    # Waveguiding region: the core, |x|, |y| <= core_radius (case 2: inside the ring of holes)
    core_radius = outer_radius if case == 1 else di - inner_radius
    grading = dict(x_targets = [-core_radius, 0.0, core_radius],
                   y_targets = [-core_radius, 0.0, core_radius],
                   x_width = core_radius / 3.0, y_width = core_radius / 3.0)
    return fiber_dict, grading


def build_hpw(g, metal):
    fiber_dict = [
        {'type': 'rectangle', 'x1': -np.inf, 'x2': +np.inf, 'y1': -np.inf, 'y2': +np.inf,
         'e_value_inside': n_air ** 2},
        {'type': 'rectangle', 'x1': -np.inf, 'x2': +np.inf, 'y1': -np.inf, 'y2': 0.0,
         'e_value_inside': n_sio2 ** 2},
        {'type': 'multilayer_rect', 'x0': 0.0, 'y0': 0.0, 'width': g['w_wg'], 'layers': [
            {'height': g['h_si'], 'e_value_inside': n_si ** 2},
            {'height': g['h_spacer'], 'e_value_inside': n_sio2 ** 2},
            {'height': g['h_metal'], 'e_value_inside': metals[metal] ** 2},
        ]},
    ]

    # Waveguiding region: the Si/SiO2/metal stack, finest around the spacer gap
    # where the hybrid plasmonic mode is concentrated
    y_si_top = g['h_si']
    y_gap_top = g['h_si'] + g['h_spacer']
    y_metal_top = y_gap_top + g['h_metal']
    grading = dict(x_targets = [-g['w_wg'] / 2, 0.0, g['w_wg'] / 2],
                   y_targets = [0.0, y_si_top, (y_si_top + y_gap_top) / 2, y_gap_top, y_metal_top],
                   x_width = g['w_wg'] / 4, y_width = g['h_spacer'] / 2)
    return fiber_dict, grading


def build_geometry():
    if case == 3:
        return build_hpw(hpw, metal)
    return build_fibers()


def make_solver(N, fiber_dict, grading, averaging = 'tensor', nt = None, verbose = False):
    Nx = N
    Ny = N
    Dx = Lx / Nx
    Dy = Ly / Ny
    xmin = -Nx / 2 * Dx
    ymin = -Ny / 2 * Dy

    if nonuniform:
        # Same span as the uniform grid's primary points, same point budget
        x_edges = graded_edges(Nx, xmin, xmin + (Nx - 1) * Dx, targets = grading['x_targets'],
                               width = grading['x_width'], boost = grid_boost)
        y_edges = graded_edges(Ny, ymin, ymin + (Ny - 1) * Dy, targets = grading['y_targets'],
                               width = grading['y_width'], boost = grid_boost)
        # On a non-uniform grid dPML is a physical length, not a cell count
        dPML = dPML_cells * Dx
        if verbose:
            print('N = ', N, 'min cell = %.4g, max cell = %.4g (uniform = %.4g)'
                  % (np.diff(y_edges).min(), np.diff(y_edges).max(), Dy))
    else:
        x_edges = y_edges = None
        dPML = dPML_cells

    return yee_grid(Nx, Ny, Dx, Dy, fiber_dict, xmin = xmin, ymin = ymin, omega = omega,
                    averaging = averaging, nmodes = nmodes,
                    ntarget = ntarget if nt is None else nt, dPML = dPML,
                    use_gpu = use_gpu, x_edges = x_edges, y_edges = y_edges)


# =============================================================================
# HPW figures of merit (ported from fdfd_optimization_benchmarking, integrated
# over the actual physical cell areas so they also hold on a graded grid)
# =============================================================================

def field_grid(Y):
    """Physical coordinates and cell areas (um^2) of the Nx x Ny field arrays."""
    x = Y.xe_phys[0::2]
    y = Y.ye_phys[0::2]
    X, Yg = np.meshgrid(x, y, indexing = 'ij')
    dA = np.outer(np.gradient(x), np.gradient(y))
    return X, Yg, dA


def hpw_masks(Y, g):
    X, Yg, dA = field_grid(Y)
    inside = (X >= -g['w_wg'] / 2) & (X <= g['w_wg'] / 2)
    core = inside & (Yg >= 0) & (Yg < g['h_si'])
    gap = inside & (Yg >= g['h_si']) & (Yg <= g['h_si'] + g['h_spacer'])
    return core, gap, dA


def select_hpw_mode(Y, g):
    """Picks the hybrid plasmonic mode: the one most confined in the spacer gap,
    preferring modes whose peak intensity sits in the gap rather than the Si core."""
    core, gap, dA = hpw_masks(Y, g)
    best = (-1.0, 0, False, 0.0)
    for m in range(Y.nmodes):
        I = np.abs(Y.norm_e_calc[m]) ** 2
        P = np.sum(I * dA)
        if P == 0 or np.isnan(P):
            continue
        conf = np.sum((I * dA)[gap]) / P
        I_gap, I_core = I[gap].max(), I[core].max()
        is_valid = I_gap > I_core
        score = conf + (10.0 if is_valid else 0.0)
        if score > best[0]:
            best = (score, m, is_valid, I_gap / I_core if I_core > 1e-12 else 100.0)
    return best[1], best[2], best[3]


def gamma_vectorial(Y, m, g):
    """Vectorial nonlinear coefficient gamma (1/(W m)); only Si is nonlinear."""
    core, gap, dA = hpw_masks(Y, g)
    dA_m = dA * 1e-12
    k0 = 2 * np.pi / (l * 1e-6)
    eps0_div_mu0 = 8.8541878128e-12 / (4 * np.pi * 1e-7)

    Ex, Ey, Ez = Y.ex_calc[m], Y.ey_calc[m], Y.ez_calc[m]
    Hx, Hy = Y.hx[m], Y.hy[m]

    denominator = np.abs(np.sum((Ex * np.conj(Hy) - Ey * np.conj(Hx)) * dA_m)) ** 2
    if denominator < 1e-60:
        return 0.0

    E_sq = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
    E_dot_E = Ex ** 2 + Ey ** 2 + Ez ** 2
    sum_Ej4 = np.abs(Ex) ** 4 + np.abs(Ey) ** 4 + np.abs(Ez) ** 4
    integrand = n_si ** 2 * n2_si * (rho_si * (2 * E_sq ** 2 + np.abs(E_dot_E) ** 2) / 3.0
                                     + (1 - rho_si) * sum_Ej4)
    numerator = np.sum((integrand * dA_m)[core])
    return k0 * eps0_div_mu0 * numerator / denominator


def hpw_metrics(Y, g):
    m, is_valid, ratio = select_hpw_mode(Y, g)
    neff = Y.neff_q[m]
    L_prop = l / (4 * np.pi * abs(np.imag(neff))) if abs(np.imag(neff)) > 1e-12 else 0.0  # um
    gamma = gamma_vectorial(Y, m, g)
    return dict(mode = m, neff = neff, valid = is_valid, ratio = ratio, L_prop = L_prop,
                gamma = gamma, FOM = gamma * L_prop * 1e-6)


def hpw_cost(metrics):
    # Maximize propagation length and gamma (log scale, equal weights)
    cost = -(0.5 * np.log10(metrics['L_prop'] + 1e-2) + 0.5 * np.log10(metrics['gamma'] + 1e-2))
    if not metrics['valid']:
        cost += 2000.0 + 500.0 * max(0.0, 1.0 - metrics['ratio'])
    return 1e6 if np.isnan(cost) else cost


def print_hpw_metrics(metrics):
    print('  mode %d: neff = %.6f%+.3ej, valid = %s' % (metrics['mode'], metrics['neff'].real,
                                                     metrics['neff'].imag, metrics['valid']))
    print('  L_prop = %.2f um, gamma = %.2f 1/(W m), FOM = %.4f 1/W'
          % (metrics['L_prop'], metrics['gamma'], metrics['FOM']))


# =============================================================================
# Bayesian optimization
# =============================================================================

if run_optimization and case != 3:
    print('Bayesian optimization is only defined for case 3 (HPW), skipping')

elif run_optimization:
    from skopt import gp_minimize
    from skopt.space import Real, Categorical

    space = ([Real(*b, name = k) for k, b in opt_bounds.items()]
             + [Categorical(list(metals), name = 'metal')])
    history = []

    def objective(x):
        params = dict(zip([d.name for d in space], x))
        g = {k: params[k] for k in hpw}
        try:
            Y = make_solver(opt_N, *build_hpw(g, params['metal']), nt = params['n_target'])
            Y.solve()
            cost = hpw_cost(hpw_metrics(Y, g))
        except Exception:
            cost = 1e6
        history.append(cost)
        print('eval %3d: cost = %.4f (best %.4f)' % (len(history), cost, min(history)))
        return cost

    print('Bayesian optimization: %d evaluations at N = %d' % (opt_n_calls, opt_N))
    res = gp_minimize(objective, space, n_calls = opt_n_calls, n_initial_points = opt_n_initial,
                      acq_func = 'EI', random_state = opt_seed)

    best = dict(zip([d.name for d in space], res.x))
    hpw = {k: best[k] for k in hpw}
    ntarget = best['n_target']
    metal = best['metal']
    print('Best cost = %.4f' % res.fun)
    for k, v in best.items():
        print('  %-9s: %s' % (k, v if isinstance(v, str) else '%.4f' % v))

    plt.figure()
    plt.plot(np.arange(1, len(history) + 1), np.minimum.accumulate(history), 'r-')
    plt.xlabel('evaluation')
    plt.ylabel('best cost')
    plt.title('Bayesian optimization convergence')
    plt.tight_layout()


fiber_dict, grading = build_geometry()


# =============================================================================
# Convergence test
# =============================================================================

if run_convergence:
    # Tensor averaging (ours) against the other averaging schemes
    schemes = [('tensor', 'tensor averaging', 'r-*'),
               ('straight', r'$<\epsilon>^{-1}$', 'm--'),
               ('inverse', r'$<\epsilon^{-1}>$', 'b--'),
               ('none', 'no averaging', 'c')]
    neff_conv = {s: np.zeros([NN.size, nmodes], dtype = complex) for s, _, _ in schemes}

    for i, N in enumerate(NN):
        for s, _, _ in schemes:
            Yc = make_solver(N, fiber_dict, grading, averaging = s, verbose = (s == 'tensor'))
            Yc.solve()
            neff_conv[s][i, :] = Yc.neff_q
            print('N = ', N, 'neff %s = ' % s, neff_conv[s][i,:])

    print('reference neff = ', target_neff)

    fig, axes = plt.subplots(1, nmodes, figsize = (5.5 * nmodes, 4.5), squeeze = False)
    for m, ax in enumerate(axes[0]):
        for s, label, style in schemes:
            ax.plot(NN, np.real(neff_conv[s][:,m]), style, label = label)
        ax.axhline(target_neff, color = 'k', linestyle = ':', label = 'reference')
        ax.set_xlabel('N')
        ax.set_ylabel(r'Re($n_{eff}$)')
        ax.set_title('Mode %d convergence (%s grid)' % (m, 'graded' if nonuniform else 'uniform'))
    axes[0, -1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    fig.tight_layout()


# =============================================================================
# Structure and mode plots
# =============================================================================

def draw_outline(ax, Y, color = 'w'):
    """Material boundaries, from the permittivity sampled on a fine uniform grid."""
    xf = np.linspace(Y.xmin_phys, Y.xmax_phys, 600)
    yf = np.linspace(Y.ymin_phys, Y.ymax_phys, 600)
    YYf, XXf = np.meshgrid(yf, xf)
    ef = np.real(calc_dist_e(Y.calldicts, XXf, YYf))
    values = np.unique(np.round(ef, 6))
    if values.size > 1:
        ax.contour(XXf, YYf, ef, levels = (values[:-1] + values[1:]) / 2,
                   colors = color, linewidths = 0.8, linestyles = 'solid')


if run_mode_plots:
    Y = make_solver(N_plot, fiber_dict, grading, verbose = True)
    Y.solve()
    X, Yg, dA = field_grid(Y)
    grid_name = 'graded' if nonuniform else 'uniform'

    if case == 3:
        metrics = hpw_metrics(Y, hpw)
        print('Design (N = %d, %s grid):' % (N_plot, grid_name))
        print_hpw_metrics(metrics)

    # Structure: refractive index with material boundaries and the mesh lines,
    # full domain and zoomed on the waveguiding region
    zoom_x = (min(grading['x_targets']), max(grading['x_targets']))
    zoom_y = (min(grading['y_targets']), max(grading['y_targets']))
    pad = 0.5 * max(zoom_x[1] - zoom_x[0], zoom_y[1] - zoom_y[0])
    fig, axes = plt.subplots(1, 2, figsize = (12, 5.5))
    for ax, zoom in zip(axes, (False, True)):
        pc = ax.pcolormesh(Y.xxe_phys, Y.yye_phys, np.real(np.sqrt(Y.eavg)), shading = 'nearest',
                           cmap = 'viridis')
        fig.colorbar(pc, ax = ax, label = 'Re(n)')
        for xv in X[:, 0]:
            ax.axvline(xv, color = 'w', linewidth = 0.3, alpha = 0.5)
        for yv in Yg[0, :]:
            ax.axhline(yv, color = 'w', linewidth = 0.3, alpha = 0.5)
        draw_outline(ax, Y, color = 'r')
        if zoom:
            ax.set_xlim(zoom_x[0] - pad, zoom_x[1] + pad)
            ax.set_ylim(zoom_y[0] - pad, zoom_y[1] + pad)
        ax.set_aspect('equal')
        ax.set_xlabel(r'x ($\mu m$)')
        ax.set_ylabel(r'y ($\mu m$)')
        ax.set_title('Structure, %s mesh (N = %d)%s' % (grid_name, N_plot, ', zoom' if zoom else ''))
    fig.tight_layout()

    # |E| of every computed mode
    fig, axes = plt.subplots(1, nmodes, figsize = (5 * nmodes, 4.5), squeeze = False)
    for m, ax in enumerate(axes[0]):
        pc = ax.pcolormesh(X, Yg, np.abs(Y.norm_e_calc[m]), shading = 'nearest', cmap = 'inferno')
        fig.colorbar(pc, ax = ax)
        draw_outline(ax, Y)
        ax.set_aspect('equal')
        ax.set_xlabel(r'x ($\mu m$)')
        ax.set_ylabel(r'y ($\mu m$)')
        ax.set_title(r'|E|, mode %d: $n_{eff}$ = %.4f%+.2ej' % (m, Y.neff_q[m].real, Y.neff_q[m].imag))
    fig.tight_layout()

    # Field components of the selected mode
    pm = metrics['mode'] if case == 3 else ploting_mode
    components = [('Ex', Y.ex_calc), ('Ey', Y.ey_calc), ('Ez', Y.ez_calc),
                  ('Hx', Y.hx), ('Hy', Y.hy), ('Hz', Y.hz)]
    fig, axes = plt.subplots(2, 3, figsize = (14, 8))
    for (name, data), ax in zip(components, axes.ravel()):
        pc = ax.pcolormesh(X, Yg, np.abs(data[pm]), shading = 'nearest', cmap = 'inferno')
        fig.colorbar(pc, ax = ax)
        draw_outline(ax, Y)
        ax.set_aspect('equal')
        ax.set_title('|%s|' % name)
    fig.suptitle(r'Mode %d field components, $n_{eff}$ = %.4f' % (pm, Y.neff_q[pm].real))
    fig.tight_layout()

    # |E| cut through the waveguide centre (x = 0)
    ix = np.argmin(np.abs(X[:, 0]))
    fig, ax = plt.subplots()
    ax.plot(Yg[ix, :], np.abs(Y.norm_e_calc[pm, ix, :]), 'r.-', markersize = 3)
    if case == 3:
        levels = np.cumsum([0.0, hpw['h_si'], hpw['h_spacer'], hpw['h_metal']])
        for y_lev in levels:
            ax.axvline(y_lev, color = 'k', linestyle = '--', alpha = 0.5)
        ax.set_xlim(-0.5, levels[-1] + 0.5)
    ax.set_xlabel(r'y ($\mu m$)')
    ax.set_ylabel('|E|')
    ax.set_title('Mode %d profile at x = 0' % pm)
    fig.tight_layout()

plt.show()
