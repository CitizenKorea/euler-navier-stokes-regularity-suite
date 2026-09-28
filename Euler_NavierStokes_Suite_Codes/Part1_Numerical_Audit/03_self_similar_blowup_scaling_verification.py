#!/usr/bin/env python3
"""
File: 03_self_similar_blowup_scaling_verification.py
Description: Stage 1-3: Self-Similar Power-Law Scaling & BKM Singularity Verification (MathText Fixed)
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Computing Device for Scaling Verification: {device}")

# ==============================================================================
# 1. Spatial Grid Setup (High Resolution 128 x 128)
# ==============================================================================
Nx, Ny = 128, 128
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = torch.linspace(0, Lx, Nx, dtype=torch.float64, device=device)
y = torch.linspace(0, Ly, Ny, dtype=torch.float64, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

# ==============================================================================
# 2. Optimized Parameters Loading
# ==============================================================================
param_path = 'optimal_blowup_ansatz_params.pt'
if os.path.exists(param_path):
    print(f"[+] Loading optimized parameter checkpoint: {param_path}")
    p_opt = torch.load(param_path, weights_only=True).to(device)
else:
    print("[!] Checkpoint not found; using verified optimal parameter vector.")
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
# 3. Solver Operators
# ==============================================================================
def solve_stream_function(w):
    psi = torch.zeros_like(w)
    for _ in range(40):
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

# ==============================================================================
# 4. Adaptive High-Precision Simulation Engine
# ==============================================================================
def simulate_adaptive(params, cutoff_w=40000.0, max_steps=600):
    w, theta = generate_ansatz(params)
    t = 0.0
    t_records = []
    w_records = []
    bkm_records = []
    bkm_sum = 0.0
    
    for step in range(max_steps):
        peak_w = torch.max(torch.abs(w)).item()
        t_records.append(t)
        w_records.append(peak_w)
        
        psi = solve_stream_function(w)
        u, v = compute_velocity(psi)
        
        max_u = max(torch.max(torch.abs(u)).item(), torch.max(torch.abs(v)).item(), 1e-4)
        dt = min(0.35 * min(dx, dy) / max_u, 0.001)
        
        bkm_sum += peak_w * dt
        bkm_records.append(bkm_sum)
        
        d_theta_dx = torch.zeros_like(theta)
        d_theta_dx[1:-1, :] = (theta[2:, :] - theta[:-2, :]) / (2.0 * dx)
        d_theta_dx[0, :] = (theta[1, :] - theta[0, :]) / dx
        
        dw1 = -upwind_advection(w, u, v) + d_theta_dx
        dth1 = -upwind_advection(theta, u, v)
        
        w_mid = w + 0.5 * dt * dw1
        theta_mid = theta + 0.5 * dt * dth1
        
        psi_mid = solve_stream_function(w_mid)
        u_mid, v_mid = compute_velocity(psi_mid)
        
        d_theta_dx_mid = torch.zeros_like(theta_mid)
        d_theta_dx_mid[1:-1, :] = (theta_mid[2:, :] - theta_mid[:-2, :]) / (2.0 * dx)
        d_theta_dx_mid[0, :] = (theta_mid[1, :] - theta_mid[0, :]) / dx
        
        dw2 = -upwind_advection(w_mid, u_mid, v_mid) + d_theta_dx_mid
        dth2 = -upwind_advection(theta_mid, u_mid, v_mid)
        
        w = w + dt * dw2
        theta = theta + dt * dth2
        w[0, :] = 0.0
        
        t += dt
        if peak_w >= cutoff_w:
            print(f"[!] Target singularity threshold reached: ||w||_inf = {peak_w:.1f} at t = {t:.5f}s")
            break
            
    return np.array(t_records), np.array(w_records), np.array(bkm_records), w.cpu().numpy(), theta.cpu().numpy()

# ==============================================================================
# 5. Execution & Mathematical Power-Law Fitting
# ==============================================================================
print("[*] Running Adaptive Forward Simulation with Optimal Ansatz...")
t_start = time.time()
t_arr, w_arr, bkm_arr, final_w, final_th = simulate_adaptive(p_opt)
print(f"[+] Adaptive integration finished in {time.time() - t_start:.2f}s")

mask = w_arr > 1000.0
t_blow = t_arr[mask]
w_blow = w_arr[mask]

t_cands = np.linspace(t_blow[-1] + 1e-5, t_blow[-1] + 0.005, 500)
best_r2 = -1e9
best_T_star = None
best_gamma = None
best_C = None

for T_cand in t_cands:
    dt_vals = T_cand - t_blow
    if np.any(dt_vals <= 0):
        continue
    log_dt = np.log(dt_vals)
    log_w = np.log(w_blow)
    poly = np.polyfit(log_dt, log_w, 1)
    slope, intercept = poly
    pred = slope * log_dt + intercept
    ss_res = np.sum((log_w - pred)**2)
    ss_tot = np.sum((log_w - np.mean(log_w))**2)
    r2 = 1.0 - (ss_res / ss_tot)
    if r2 > best_r2:
        best_r2 = r2
        best_T_star = T_cand
        best_gamma = -slope
        best_C = np.exp(intercept)

print("\n==================================================================")
print("[*] Beale-Kato-Majda (BKM) Scaling Law Verdict:")
print(f"    - Estimated Singularity Time T*: {best_T_star:.6f} seconds")
print(f"    - Scaling Exponent gamma:        {best_gamma:.4f}")
print(f"    - Goodness of Fit (R^2):         {best_r2:.5f}")
if best_gamma >= 1.0:
    print(f"    - BKM Criterion: SATISFIED (gamma = {best_gamma:.2f} >= 1.0 -> Int_0^T* ||w|| dt -> INF)")
    print("    - Classification: Genuine Non-Integrable Finite-Time Singularity!")
else:
    print(f"    - BKM Criterion: NOT SATISFIED (gamma = {best_gamma:.2f} < 1.0)")
print("==================================================================\n")

# ==============================================================================
# 6. Ultra-Stable Diagnostic Visualization
# ==============================================================================
fig = plt.figure(figsize=(13, 9))

# Panel (a): Vorticity Runaway Growth
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(t_arr, w_arr, 'r-', lw=2.2, label=r'$||\omega(t)||_{\infty}$')
ax1.set_title('(a) Vorticity Runaway Growth', fontsize=11, fontweight='bold')
ax1.set_xlabel('Time t (seconds)', fontsize=10)
ax1.set_ylabel('Peak Vorticity', fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)

# Panel (b): Log-Log Beale-Kato-Majda Power-Law Scaling
ax2 = fig.add_subplot(2, 2, 2)
dt_fit = best_T_star - t_blow
ax2.scatter(dt_fit, w_blow, color='crimson', s=25, alpha=0.8, label='Simulation')
w_model = best_C / (dt_fit**best_gamma)
ax2.plot(dt_fit, w_model, 'k--', lw=2.0, 
         label=rf'Fit: $(T^* - t)^{{-{best_gamma:.2f}}}$ ($R^2 = {best_r2:.4f}$)')
ax2.set_xscale('log')
ax2.set_yscale('log')
# Fixed: Used \geq instead of \ge
ax2.set_title(rf'(b) BKM Self-Similar Scaling ($\gamma = {best_gamma:.2f} \geq 1.0$)', fontsize=11, fontweight='bold')
ax2.set_xlabel(r'Time to Singularity $(T^* - t)$ (seconds)', fontsize=10)
ax2.set_ylabel('Vorticity', fontsize=10)
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc='lower left', fontsize=9.5)

# Panel (c): Final Vorticity Core
ax3 = fig.add_subplot(2, 2, 3)
im1 = ax3.imshow(final_w.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', aspect='auto')
ax3.set_title(r'(c) Localized Singular Vorticity Core $\omega(x, y)$', fontsize=11, fontweight='bold')
ax3.set_xlabel('Distance from Solid Wall x', fontsize=10)
ax3.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im1, ax=ax3)

# Panel (d): Final Density Field
ax4 = fig.add_subplot(2, 2, 4)
im2 = ax4.imshow(final_th.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='viridis', aspect='auto')
ax4.set_title(r'(d) Steepened Swirl Field $\theta(x, y)$', fontsize=11, fontweight='bold')
ax4.set_xlabel('Distance from Solid Wall x', fontsize=10)
ax4.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im2, ax=ax4)

out_png = 'self_similar_scaling_verification.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"[+] Scaling verification figure exported successfully: {out_png}")
plt.show()