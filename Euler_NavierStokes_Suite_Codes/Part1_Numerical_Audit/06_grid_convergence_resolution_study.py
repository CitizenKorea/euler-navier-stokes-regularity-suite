#!/usr/bin/env python3
"""
File: 06_grid_convergence_resolution_study.py
Description: Stage 2-Final: Rigorous Multi-Grid Resolution Convergence Study (Fixed & Robust)
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Computing Device for Grid Convergence: {device}")

# ==============================================================================
# 1. Optimal Blowup Parameter Vector
# ==============================================================================
param_path = 'optimal_blowup_ansatz_params.pt'
if os.path.exists(param_path):
    print(f"[+] Loading verified parameter checkpoint: {param_path}")
    p_opt = torch.load(param_path, weights_only=True).to(device)
else:
    p_opt = torch.tensor([1.652, 1.643, -1.527, 0.276, 1.518, 0.970], dtype=torch.float64, device=device)

# ==============================================================================
# 2. Resolution-Adaptive Simulator Engine
# ==============================================================================
def run_resolution_case(N, cutoff_w=35000.0, max_t=0.22):
    Nx, Ny = N, N
    Lx, Ly = 1.0, 1.0
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x = torch.linspace(0, Lx, Nx, dtype=torch.float64, device=device)
    y = torch.linspace(0, Ly, Ny, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Continuous ansatz formulation
    amp  = torch.exp(p_opt[0]) * 100.0
    cx   = torch.exp(p_opt[1]) * 12.0
    cy   = torch.exp(p_opt[2]) * 15.0
    x0   = torch.sigmoid(p_opt[3]) * 0.15
    y0   = torch.sigmoid(p_opt[4]) * 0.25
    skew = torch.tanh(p_opt[5]) * 10.0

    r_sq = cx * (X - x0)**2 + cy * (Y - y0)**2 + skew * (X - x0) * (Y - y0)
    theta = amp * (1.0 - X**2) * torch.exp(-torch.clamp(r_sq, min=0.0, max=50.0)) * torch.sin(np.pi * Y)
    w = torch.zeros_like(theta)

    # Resolution-scaled Poisson iterations
    n_iters = 35 if N <= 64 else (50 if N <= 128 else 90)

    def solve_stream(w_field):
        psi = torch.zeros_like(w_field)
        for _ in range(n_iters):
            psi_new = torch.zeros_like(psi)
            psi_new[1:-1, 1:-1] = 0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1] + 
                                          psi[1:-1, 2:] + psi[1:-1, :-2] + dx * dy * w_field[1:-1, 1:-1])
            psi_new[0, :] = 0.0
            psi_new[-1, :] = 0.0
            psi_new[:, 0] = psi_new[:, 1]
            psi_new[:, -1] = 0.0
            psi = psi_new
        return psi

    def get_velocity(psi):
        u = torch.zeros_like(psi)
        v = torch.zeros_like(psi)
        u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dy)
        v[1:-1, :] = +(psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
        u[0, :] = 0.0
        v[0, :] = (psi[1, :] - psi[0, :]) / dx
        return u, v

    def upwind_advection(field, u, v):
        u_p = torch.clamp(u, min=0.0)
        u_m = torch.clamp(u, max=0.0)
        v_p = torch.clamp(v, min=0.0)
        v_m = torch.clamp(v, max=0.0)

        df_dx_b = (field[1:, :] - field[:-1, :]) / dx
        df_dx_f = torch.cat([df_dx_b, df_dx_b[-1:]], dim=0)
        df_dx_b = torch.cat([df_dx_b[:1], df_dx_b], dim=0)

        df_dy_b = (field[:, 1:] - field[:, :-1]) / dy
        df_dy_f = torch.cat([df_dy_b, df_dy_b[:, -1:]], dim=1)
        df_dy_b = torch.cat([df_dy_b[:, :1], df_dy_b], dim=1)

        return u_p * df_dx_b + u_m * df_dx_f + v_p * df_dy_b + v_m * df_dy_f

    def measure_core_fwhm(w_field):
        w_abs = torch.abs(w_field)
        max_val = torch.max(w_abs).item()
        if max_val < 50.0:
            return 1.0, 1.0 / dx

        idx = torch.argmax(w_abs)
        ix, iy = idx // Ny, idx % Ny

        slice_x = w_abs[:, iy].cpu().numpy()
        half_max = max_val * 0.5
        above_half = np.where(slice_x >= half_max)[0]
        if len(above_half) > 1:
            fwhm_x = (above_half[-1] - above_half[0] + 1) * dx
            n_pts = float(len(above_half))
        else:
            fwhm_x = dx
            n_pts = 1.0
        return fwhm_x, n_pts

    t = 0.0
    t_records, w_records, bkm_records, n_core_records = [], [], [], []
    bkm_sum = 0.0

    # Scale max steps proportionally with N to ensure equal physical time reach
    max_steps = int(450 * (N / 64))
    t_start = time.time()

    for step in range(max_steps):
        peak_w = torch.max(torch.abs(w)).item()
        fwhm_x, n_core = measure_core_fwhm(w)

        # Synchronized record keeping
        t_records.append(t)
        w_records.append(peak_w)
        bkm_records.append(bkm_sum)
        n_core_records.append(n_core)

        if peak_w >= cutoff_w or t >= max_t:
            break

        psi = solve_stream(w)
        u, v = get_velocity(psi)
        max_u = max(torch.max(torch.abs(u)).item(), torch.max(torch.abs(v)).item(), 1e-4)
        dt = min(0.35 * min(dx, dy) / max_u, 0.001)

        bkm_sum += peak_w * dt

        d_theta_dx = torch.zeros_like(theta)
        d_theta_dx[1:-1, :] = (theta[2:, :] - theta[:-2, :]) / (2.0 * dx)
        d_theta_dx[0, :] = (theta[1, :] - theta[0, :]) / dx

        # RK2 Integrator
        dw1 = -upwind_advection(w, u, v) + d_theta_dx
        dth1 = -upwind_advection(theta, u, v)
        w_mid = w + 0.5 * dt * dw1
        theta_mid = theta + 0.5 * dt * dth1

        psi_mid = solve_stream(w_mid)
        u_mid, v_mid = get_velocity(psi_mid)

        d_theta_dx_mid = torch.zeros_like(theta_mid)
        d_theta_dx_mid[1:-1, :] = (theta_mid[2:, :] - theta_mid[:-2, :]) / (2.0 * dx)
        d_theta_dx_mid[0, :] = (theta_mid[1, :] - theta_mid[0, :]) / dx

        dw2 = -upwind_advection(w_mid, u_mid, v_mid) + d_theta_dx_mid
        dth2 = -upwind_advection(theta_mid, u_mid, v_mid)

        w = w + dt * dw2
        theta = theta + dt * dth2
        w[0, :] = 0.0
        t += dt

    elapsed = time.time() - t_start
    return (np.array(t_records), np.array(w_records), np.array(bkm_records), 
            np.array(n_core_records), elapsed, w.cpu().numpy(), dx)

# ==============================================================================
# 3. Multi-Grid Execution & Scaling Law Extraction
# ==============================================================================
GRID_RESOLUTIONS = [64, 128, 256]
COLORS = {64: '#1f77b4', 128: '#ff7f0e', 256: '#d62728'}
grid_results = {}
scaling_metrics = {}

print("\n" + "="*70)
print("[*] Launching Multi-Grid Convergence Study: N = [64, 128, 256]")
print("="*70)

for N in GRID_RESOLUTIONS:
    t_hist, w_hist, bkm_hist, n_core_hist, el_sec, w_final, dx_val = run_resolution_case(N)
    grid_results[N] = (t_hist, w_hist, bkm_hist, n_core_hist, w_final, dx_val)

    # Perform Power-law fit: ||w|| ~ C / (T* - t)^gamma
    mask = w_hist > 1000.0
    t_blow = t_hist[mask]
    w_blow = w_hist[mask]

    best_r2, best_T, best_gamma = -1e9, 0.0, 0.0
    if len(t_blow) > 5:
        t_cands = np.linspace(t_blow[-1] + 1e-5, t_blow[-1] + 0.015, 500)
        for T_cand in t_cands:
            dt_vals = T_cand - t_blow
            if np.any(dt_vals <= 0):
                continue
            log_dt = np.log(dt_vals)
            log_w = np.log(w_blow)
            slope, intercept = np.polyfit(log_dt, log_w, 1)
            pred = slope * log_dt + intercept
            ss_res = np.sum((log_w - pred)**2)
            ss_tot = np.sum((log_w - np.mean(log_w))**2)
            r2 = 1.0 - (ss_res / ss_tot)
            if r2 > best_r2:
                best_r2 = r2
                best_T = T_cand
                best_gamma = -slope

    scaling_metrics[N] = (best_T, best_gamma, best_r2, max(w_hist))
    print(f"  [+] Grid {N:3d}x{N:3d} (dx={dx_val:.5f}) | Max ||w||: {max(w_hist):8.1f} | "
          f"T*: {best_T:.5f}s | gamma: {best_gamma:.3f} (R^2={best_r2:.4f}) | Time: {el_sec:.2f}s")

print("="*70)
print("[*] Grid Convergence Summary:")
print("    N      dx        T*(s)     gamma      R^2       Max ||w||")
for N in GRID_RESOLUTIONS:
    T_s, gam, r2, pk = scaling_metrics[N]
    dx_v = grid_results[N][5]
    print(f"   {N:3d}   {dx_v:.5f}   {T_s:.5f}   {gam:.4f}    {r2:.5f}   {pk:8.1f}")
print("="*70 + "\n")

# ==============================================================================
# 4. Publication-Grade Diagnostic Visualization (2x2 Panel)
# ==============================================================================
fig = plt.figure(figsize=(14, 10))

# Panel (a): Peak Vorticity Growth across Grid Resolutions
ax1 = fig.add_subplot(2, 2, 1)
for N in GRID_RESOLUTIONS:
    t_h, w_h, _, _, _, _ = grid_results[N]
    ax1.plot(t_h, w_h, color=COLORS[N], lw=2.2, label=f'N = {N} (dx = {1.0/(N-1):.4f})')
ax1.set_title(r'(a) Vorticity Runaway $\|\omega(t)\|_\infty$ vs Grid Resolution', fontsize=11, fontweight='bold')
ax1.set_xlabel('Time t (seconds)', fontsize=10)
ax1.set_ylabel(r'Maximum Vorticity $\|\omega\|_\infty$', fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc='upper left', fontsize=9.5)

# Panel (b): Beale-Kato-Majda (BKM) Integral Growth
ax2 = fig.add_subplot(2, 2, 2)
for N in GRID_RESOLUTIONS:
    t_h, _, bkm_h, _, _, _ = grid_results[N]
    ax2.plot(t_h, bkm_h, color=COLORS[N], lw=2.2, label=f'N = {N}')
ax2.set_title(r'(b) Cumulative BKM Integral $\int_0^t \|\omega(s)\|_\infty ds$', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time t (seconds)', fontsize=10)
ax2.set_ylabel('BKM Accumulation', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=9.5)

# Panel (c): Core Resolution Metric
ax3 = fig.add_subplot(2, 2, 3)
for N in GRID_RESOLUTIONS:
    t_h, _, _, n_c_h, _, _ = grid_results[N]
    ax3.plot(t_h, n_c_h, color=COLORS[N], lw=2.0, label=f'N = {N}')
ax3.axhline(y=4.0, color='gray', linestyle='--', lw=1.8, label='Resolution Margin (N_pts = 4)')
ax3.set_title(r'(c) Vortex Core Resolution $N_{\mathrm{core}} = \mathrm{FWHM}_x / \Delta x$', fontsize=11, fontweight='bold')
ax3.set_xlabel('Time t (seconds)', fontsize=10)
ax3.set_ylabel('Grid Points across Core', fontsize=10)
ax3.set_ylim([0, 30])
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right', fontsize=9.0)

# Panel (d): Scaling Exponent & Extrapolated Continuum Limit
ax4 = fig.add_subplot(2, 2, 4)
inv_N = [1.0 / N for N in GRID_RESOLUTIONS]
gamma_vals = [scaling_metrics[N][1] for N in GRID_RESOLUTIONS]

poly = np.polyfit(inv_N, gamma_vals, 1)
h_dense = np.linspace(0.0, max(inv_N) * 1.1, 100)
gamma_extrap = np.polyval(poly, h_dense)
gamma_inf = poly[1]

ax4.scatter(inv_N, gamma_vals, color='red', s=60, zorder=5, label='Simulation Scaling Exponent')
ax4.plot(h_dense, gamma_extrap, 'k--', lw=2.0, 
         label=rf'Continuum Extrapolation ($h \to 0$): $\gamma_\infty \approx {gamma_inf:.3f}$')
ax4.axhline(y=1.0, color='blue', linestyle=':', lw=1.8, label=r'BKM Blowup Threshold ($\gamma = 1.0$)')
ax4.set_title(r'(d) Continuum Limit of Scaling Exponent $\gamma(h \to 0)$', fontsize=11, fontweight='bold')
ax4.set_xlabel(r'Grid Spacing $h = 1/N$', fontsize=10)
ax4.set_ylabel(r'Scaling Exponent $\gamma$', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='lower left', fontsize=9.0)

out_png = 'grid_convergence_resolution_study.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"[+] Grid convergence benchmark exported successfully: {out_png}")
plt.show()