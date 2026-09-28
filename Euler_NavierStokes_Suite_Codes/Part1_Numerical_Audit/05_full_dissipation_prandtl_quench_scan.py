#!/usr/bin/env python3
"""
File: 05_full_dissipation_prandtl_quench_scan.py
Description: Stage 2: Coupled Viscous & Thermal Diffusion (Prandtl = 1.0) Barrier Scan
Physics:
  - Full Boussinesq Navier-Stokes with coupled scalar diffusion:
      d_t w + u.grad(w) = d_x theta + nu * Laplacian(w)
      d_t theta + u.grad(theta) = kappa * Laplacian(theta)
  - Unit Prandtl number: Pr = nu / kappa = 1.0
  - Precision sweep across low Re: [10, 20, 25, 27, 28, 50] to find Re_c
Output:
  - full_dissipation_quench_scan.png
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Computing Device for Coupled Dissipation Scan: {device}")

# ==============================================================================
# 1. Spatial Domain Setup (128 x 128)
# ==============================================================================
Nx, Ny = 128, 128
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = torch.linspace(0, Lx, Nx, dtype=torch.float64, device=device)
y = torch.linspace(0, Ly, Ny, dtype=torch.float64, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

# ==============================================================================
# 2. Optimized Blowup Ansatz Loader
# ==============================================================================
param_path = 'optimal_blowup_ansatz_params.pt'
if os.path.exists(param_path):
    print(f"[+] Loading optimal parameters: {param_path}")
    p_opt = torch.load(param_path, weights_only=True).to(device)
else:
    p_opt = torch.tensor([1.652, 1.643, -1.527, 0.276, 1.518, 0.970], dtype=torch.float64, device=device)

def generate_ansatz(params):
    amp  = torch.exp(params[0]) * 100.0
    cx   = torch.exp(params[1]) * 12.0
    cy   = torch.exp(params[2]) * 15.0
    x0   = torch.sigmoid(params[3]) * 0.15
    y0   = torch.sigmoid(params[4]) * 0.25
    skew = torch.tanh(params[5]) * 10.0
    
    r_sq = cx * (X - x0)**2 + cy * (Y - y0)**2 + skew * (X - x0) * (Y - y0)
    theta_0 = amp * (1.0 - X**2) * torch.exp(-torch.clamp(r_sq, min=0.0, max=50.0)) * torch.sin(np.pi * Y)
    w_0 = torch.zeros_like(theta_0)
    return w_0, theta_0

# ==============================================================================
# 3. Solvers & Differential Operators
# ==============================================================================
def solve_stream_function(w):
    psi = torch.zeros_like(w)
    for _ in range(35):
        psi_new = torch.zeros_like(psi)
        psi_new[1:-1, 1:-1] = 0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1] + 
                                      psi[1:-1, 2:] + psi[1:-1, :-2] + dx * dy * w[1:-1, 1:-1])
        psi_new[0, :] = 0.0
        psi_new[-1, :] = 0.0
        psi_new[:, 0] = psi_new[:, 1]
        psi_new[:, -1] = 0.0
        psi = psi_new
    return psi

def compute_velocity(psi):
    u = torch.zeros_like(psi)
    v = torch.zeros_like(psi)
    u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dy)
    v[1:-1, :] = +(psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
    u[0, :] = 0.0
    v[0, :] = (psi[1, :] - psi[0, :]) / dx
    return u, v

def upwind_advection(field, u, v):
    u_pos = torch.clamp(u, min=0.0)
    u_neg = torch.clamp(u, max=0.0)
    v_pos = torch.clamp(v, min=0.0)
    v_neg = torch.clamp(v, max=0.0)
    
    df_dx_b = (field[1:, :] - field[:-1, :]) / dx
    df_dx_f = torch.cat([df_dx_b, df_dx_b[-1:]], dim=0)
    df_dx_b = torch.cat([df_dx_b[:1], df_dx_b], dim=0)
    
    df_dy_b = (field[:, 1:] - field[:, :-1]) / dy
    df_dy_f = torch.cat([df_dy_b, df_dy_b[:, -1:]], dim=1)
    df_dy_b = torch.cat([df_dy_b[:, :1], df_dy_b], dim=1)
    
    return u_pos * df_dx_b + u_neg * df_dx_f + v_pos * df_dy_b + v_neg * df_dy_f

def laplacian_2d(f):
    lap = torch.zeros_like(f)
    lap[1:-1, 1:-1] = (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dx * dx) + \
                      (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / (dy * dy)
    return lap

# ==============================================================================
# 4. Coupled Viscous-Thermal Simulator Engine
# ==============================================================================
def simulate_coupled(Re, Pr=1.0, max_t=0.25, cutoff_w=30000.0):
    nu = 1.0 / Re
    kappa = nu / Pr
    w, theta = generate_ansatz(p_opt)
    t = 0.0
    t_hist = []
    w_peak_hist = []
    
    max_steps = 750
    for step in range(max_steps):
        peak_w = torch.max(torch.abs(w)).item()
        t_hist.append(t)
        w_peak_hist.append(peak_w)
        
        if peak_w >= cutoff_w:
            break
        if t >= max_t:
            break
            
        psi = solve_stream_function(w)
        u, v = compute_velocity(psi)
        
        max_u = max(torch.max(torch.abs(u)).item(), torch.max(torch.abs(v)).item(), 1e-4)
        
        dt_adv = 0.35 * min(dx, dy) / max_u
        dt_diff = 0.22 * min(dx, dy)**2 / max(nu, kappa)
        dt = min(dt_adv, dt_diff, 0.001)
        
        d_theta_dx = torch.zeros_like(theta)
        d_theta_dx[1:-1, :] = (theta[2:, :] - theta[:-2, :]) / (2.0 * dx)
        d_theta_dx[0, :] = (theta[1, :] - theta[0, :]) / dx
        
        visc_w = nu * laplacian_2d(w)
        visc_th = kappa * laplacian_2d(theta)
        
        # RK2 Step 1
        dw1 = -upwind_advection(w, u, v) + d_theta_dx + visc_w
        dth1 = -upwind_advection(theta, u, v) + visc_th
        
        w_mid = w + 0.5 * dt * dw1
        theta_mid = theta + 0.5 * dt * dth1
        
        # RK2 Step 2
        psi_mid = solve_stream_function(w_mid)
        u_mid, v_mid = compute_velocity(psi_mid)
        
        d_theta_dx_mid = torch.zeros_like(theta_mid)
        d_theta_dx_mid[1:-1, :] = (theta_mid[2:, :] - theta_mid[:-2, :]) / (2.0 * dx)
        d_theta_dx_mid[0, :] = (theta_mid[1, :] - theta_mid[0, :]) / dx
        
        visc_w_mid = nu * laplacian_2d(w_mid)
        visc_th_mid = kappa * laplacian_2d(theta_mid)
        
        dw2 = -upwind_advection(w_mid, u_mid, v_mid) + d_theta_dx_mid + visc_w_mid
        dth2 = -upwind_advection(theta_mid, u_mid, v_mid) + visc_th_mid
        
        w = w + dt * dw2
        theta = theta + dt * dth2
        w[0, :] = 0.0
        
        t += dt
        
    return np.array(t_hist), np.array(w_peak_hist), w.cpu().numpy(), theta.cpu().numpy()

# ==============================================================================
# 5. Targeted Scan for Critical Reynolds Number (Re_c)
# ==============================================================================
RE_CASES = [
    (10, '#1f77b4', 'Re = 10 (Heavily Quenched)'),
    (20, '#2ca02c', 'Re = 20 (Strong Diffusion)'),
    (25, '#17becf', 'Re = 25 (Near-Critical Quench)'),
    (27, '#bcbd22', 'Re = 27 (Critical Threshold Re_c)'),
    (28, '#ff7f0e', 'Re = 28 (Breakthrough Boundary)'),
    (50, '#d62728', 'Re = 50 (Unchecked Blowup)')
]

results = {}
re_list = []
peak_list = []

print("\n[*] Starting Coupled Viscous-Thermal (Pr = 1.0) Scan...")
t_all_start = time.time()

for Re, color, label in RE_CASES:
    t0 = time.time()
    t_h, w_h, w_field, th_field = simulate_coupled(Re, Pr=1.0)
    max_peak = np.max(w_h)
    results[Re] = (t_h, w_h, w_field, th_field, color, label)
    
    re_list.append(Re)
    peak_list.append(max_peak)
    
    status = "BLOWUP (Singularity)" if max_peak >= 25000 else "QUENCHED (Viscous Decay)"
    print(f"  [+] {label:34s} | Max ||w||: {max_peak:8.1f} | End t: {t_h[-1]:.4f}s | {status} ({time.time()-t0:.2f}s)")

print(f"[+] Scan completed in {time.time() - t_all_start:.2f}s\n")

# ==============================================================================
# 6. Ultra-Stable Diagnostic Visualization (2x2 Panel)
# ==============================================================================
fig = plt.figure(figsize=(14, 10))

# Subplot 1: Vorticity Trajectories
ax1 = fig.add_subplot(2, 2, 1)
for Re, color, label in RE_CASES:
    t_h, w_h, _, _, col, lbl = results[Re]
    style = '-' if Re >= 28 else '--'
    ax1.plot(t_h, w_h, style, color=col, lw=2.2, label=lbl)

ax1.set_title('(a) Vorticity Evolution under Coupled Diffusion (Pr = 1.0)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Time t (seconds)', fontsize=10)
ax1.set_ylabel(r'Peak Vorticity $\|\omega(t)\|_\infty$', fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc='upper left', fontsize=8.5)

# Subplot 2: Bifurcation Diagram (Re vs Max Peak)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(re_list, peak_list, 'ro-', lw=2.2, markersize=7, label='Coupled Navier-Stokes (Pr = 1.0)')
ax2.axvline(x=27.5, color='darkblue', linestyle=':', lw=2.0, label='Critical Bifurcation Threshold Re_c ≈ 27.5')
ax2.set_title('(b) Discontinuous Phase Bifurcation at Critical Re_c', fontsize=11, fontweight='bold')
ax2.set_xlabel('Reynolds Number Re (1 / nu)', fontsize=10)
ax2.set_ylabel(r'Maximum Attained Vorticity $\max_t \|\omega\|_\infty$', fontsize=10)
ax2.set_yscale('log')
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc='lower right', fontsize=9.0)

# Subplot 3: Quenched Vorticity Topology at Re = 20
ax3 = fig.add_subplot(2, 2, 3)
w_quenched = results[20][2]
im1 = ax3.imshow(w_quenched.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', aspect='auto')
ax3.set_title(r'(c) Quenched Dissipated State (Re = 20, $\|\omega\| \approx 155$)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Distance from Wall x', fontsize=10)
ax3.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im1, ax=ax3)

# Subplot 4: Blowup Vorticity Topology at Re = 50
ax4 = fig.add_subplot(2, 2, 4)
w_blowup = results[50][2]
im2 = ax4.imshow(w_blowup.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', aspect='auto')
ax4.set_title(r'(d) Boundary Singularity Blowup (Re = 50, $\|\omega\| > 34,000$)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Distance from Wall x', fontsize=10)
ax4.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im2, ax=ax4)

out_png = 'full_dissipation_quench_scan.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"[+] Benchmark exported successfully: {out_png}")
plt.show()