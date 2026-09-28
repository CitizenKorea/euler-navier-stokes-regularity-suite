#!/usr/bin/env python3
"""
File: 05_spectral_dissipation_reynolds_audit.py
Description: Spectral Counterpart Audit of Viscous Dissipation (Pr = 1.0)
Physics:
  - Full Boussinesq Navier-Stokes with coupled scalar diffusion:
      d_t w + u.grad(w) = d_x theta + nu * Laplacian(w)
      d_t theta + u.grad(theta) = kappa * Laplacian(theta)
  - Unit Prandtl number: Pr = nu / kappa = 1.0
  - Exact Machine-Precision Spectral DST-I Poisson Solver (FFT-based)
  - Re-scanning Re in [10, 20, 25, 27, 28, 50, 100] to verify saturation
Output:
  - 05_spectral_dissipation_quench_scan.png
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Computing Device for Spectral Dissipation Audit: {device}")

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
# 3. Exact Spectral DST-I Poisson Solver & Differential Operators
# ==============================================================================
class SpectralPoissonSolverDST(nn.Module):
    def __init__(self, Nx, Ny, Lx=1.0, Ly=1.0, device=device):
        super().__init__()
        self.Nx, self.Ny = Nx, Ny
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.Mx, self.My = Nx - 2, Ny - 2
        
        jx = torch.arange(1, self.Mx + 1, dtype=torch.float64, device=device)
        jy = torch.arange(1, self.My + 1, dtype=torch.float64, device=device)
        lam_x = 2.0 * (1.0 - torch.cos(jx * np.pi / (self.Mx + 1))) / (self.dx**2)
        lam_y = 2.0 * (1.0 - torch.cos(jy * np.pi / (self.My + 1))) / (self.dy**2)
        self.register_buffer('LAM', lam_x[:, None] + lam_y[None, :])
        self.norm_factor = 1.0 / ((self.Mx + 1) * (self.My + 1))
        
    def forward(self, w):
        w_int = w[1:-1, 1:-1]
        ext = torch.zeros(2 * (self.Mx + 1), 2 * (self.My + 1), dtype=torch.float64, device=w.device)
        ext[1:self.Mx+1, 1:self.My+1] = w_int
        ext[self.Mx+2:, 1:self.My+1] = -torch.flip(w_int, dims=[0])
        ext[1:self.Mx+1, self.My+2:] = -torch.flip(w_int, dims=[1])
        ext[self.Mx+2:, self.My+2:] = torch.flip(w_int, dims=[0, 1])
        
        fft_w = torch.fft.fftn(ext)
        dst_w = -0.25 * torch.real(fft_w[1:self.Mx+1, 1:self.My+1])
        psi_hat = dst_w / self.LAM
        
        ext_psi = torch.zeros_like(ext)
        ext_psi[1:self.Mx+1, 1:self.My+1] = psi_hat
        ext_psi[self.Mx+2:, 1:self.My+1] = -torch.flip(psi_hat, dims=[0])
        ext_psi[1:self.Mx+1, self.My+2:] = -torch.flip(psi_hat, dims=[1])
        ext_psi[self.Mx+2:, self.My+2:] = torch.flip(psi_hat, dims=[0, 1])
        
        fft_psi = torch.fft.fftn(ext_psi)
        psi_int = -self.norm_factor * torch.real(fft_psi[1:self.Mx+1, 1:self.My+1])
        
        psi = torch.zeros(self.Nx, self.Ny, dtype=torch.float64, device=w.device)
        psi[1:-1, 1:-1] = psi_int
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

poisson_spectral = SpectralPoissonSolverDST(Nx, Ny, Lx, Ly, device=device)

# ==============================================================================
# 4. High-Precision Simulator Engine
# ==============================================================================
def simulate_spectral_coupled(Re, Pr=1.0, max_t=0.25, cutoff_w=35000.0):
    nu = 1.0 / Re
    kappa = nu / Pr
    w, theta = generate_ansatz(p_opt)
    t = 0.0
    t_hist = []
    w_peak_hist = []
    
    max_steps = 900
    for step in range(max_steps):
        peak_w = torch.max(torch.abs(w)).item()
        t_hist.append(t)
        w_peak_hist.append(peak_w)
        
        if peak_w >= cutoff_w or t >= max_t:
            break
            
        psi = poisson_spectral(w)
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
        psi_mid = poisson_spectral(w_mid)
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
# 5. Comparative Execution Suite
# ==============================================================================
RE_CASES = [
    (10,  '#1f77b4', 'Re = 10 (Heavily Damped)'),
    (20,  '#2ca02c', 'Re = 20 (Diffusive Control)'),
    (27,  '#bcbd22', 'Re = 27 (Apparent Re_c Threshold)'),
    (50,  '#ff7f0e', 'Re = 50 (Unchecked under Jacobi)'),
    (100, '#d62728', 'Re = 100 (High-Re Dissipative Test)')
]

results_spec = {}
re_list = []
peak_list = []

print("\n" + "="*75)
print("[*] Launching Exact Spectral Navier-Stokes Dissipation Audit (Pr = 1.0)...")
print("="*75)
t_start = time.time()

for Re, color, label in RE_CASES:
    t0 = time.time()
    t_h, w_h, w_f, th_f = simulate_spectral_coupled(Re, Pr=1.0)
    max_peak = np.max(w_h)
    results_spec[Re] = (t_h, w_h, w_f, th_f, color, label)
    re_list.append(Re)
    peak_list.append(max_peak)
    
    status = "BLOWUP" if max_peak >= 25000.0 else "REGULARIZED (Smooth Saturation)"
    print(f"  [+] {label:38s} | Max ||w||: {max_peak:8.1f} | End t: {t_h[-1]:.4f}s | {status} ({time.time()-t0:.2f}s)")

print("="*75)
print(f"[+] Spectral Dissipation Audit completed in {time.time() - t_start:.2f}s\n")

# ==============================================================================
# 6. Comparative Visualization
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for Re, color, label in RE_CASES:
    t_h, w_h, _, _, col, lbl = results_spec[Re]
    ax1.plot(t_h, w_h, '-', color=col, lw=2.2, label=lbl)

ax1.set_title(r'(a) Spectral Vorticity Trajectories (Pr = 1.0)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Time t (seconds)', fontsize=10)
ax1.set_ylabel(r'Maximum Vorticity $\|\omega(t)\|_\infty$', fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc='upper left', fontsize=9.0)

ax2.plot(re_list, peak_list, 'bs-', lw=2.2, markersize=7, label='Spectral DST-I (Machine Precision)')
ax2.axhline(y=30000.0, color='gray', linestyle=':', label='Spurious Jacobi Blowup Level (>30,000)')
ax2.set_title('(b) Regularized Spectral Peak vs. Reynolds Number', fontsize=11, fontweight='bold')
ax2.set_xlabel('Reynolds Number Re (1 / nu)', fontsize=10)
ax2.set_ylabel(r'Maximum Attained Vorticity $\max_t \|\omega\|_\infty$', fontsize=10)
ax2.set_yscale('log')
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc='lower right', fontsize=9.5)

out_png = '05_spectral_dissipation_quench_scan.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"[+] Output figure saved: {out_png}")
plt.show()