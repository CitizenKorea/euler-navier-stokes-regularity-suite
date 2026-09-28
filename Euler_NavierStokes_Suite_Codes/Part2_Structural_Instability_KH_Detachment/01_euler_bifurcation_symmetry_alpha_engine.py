#!/usr/bin/env python3
"""
========================================================================================
EULER RESEARCH PART II: STRUCTURAL INSTABILITY & REGULARITY BIFURCATION SUITE
Module: 01_euler_bifurcation_symmetry_alpha_engine.py
Mission:
  1. Audit Chen-Hou C^{1, alpha} corner torque persistence (alpha in [0.1, 1.0])
  2. Audit Symmetry-Breaking Phase Transition (epsilon-perturbation triggering K-H roll-up)
========================================================================================
"""

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Euler Bifurcation Engine running on: {device}")

# ==============================================================================
# 1. Machine-Precision Spectral DST-I Poisson Solver (O(10^-15) Residual)
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

# ==============================================================================
# 2. 2nd-Order TVD Advection Operator with Minmod Limiter
# ==============================================================================
def minmod(a, b):
    return torch.sign(a) * torch.clamp(torch.min(torch.abs(a), torch.abs(b)), min=0.0) * (torch.sign(a) == torch.sign(b)).double()

def tvd_advection_2d(f, u, v, dx, dy):
    df_dx = (f[1:, :] - f[:-1, :]) / dx
    slope_x = minmod(df_dx[:-1, :], df_dx[1:, :])
    slope_x = torch.cat([df_dx[:1, :], slope_x, df_dx[-1:, :]], dim=0)
    f_L = f[:-1, :] + 0.5 * dx * slope_x[:-1, :]
    f_R = f[1:, :] - 0.5 * dx * slope_x[1:, :]
    u_face = 0.5 * (u[:-1, :] + u[1:, :])
    flux_x = torch.where(u_face >= 0, u_face * f_L, u_face * f_R)
    adv_x = torch.zeros_like(f)
    adv_x[1:-1, :] = (flux_x[1:, :] - flux_x[:-1, :]) / dx
    
    df_dy = (f[:, 1:] - f[:, :-1]) / dy
    slope_y = minmod(df_dy[:, :-1], df_dy[:, 1:])
    slope_y = torch.cat([df_dy[:, :1], slope_y, df_dy[:, -1:]], dim=1)
    f_B = f[:, :-1] + 0.5 * dy * slope_y[:, :-1]
    f_T = f[:, 1:] - 0.5 * dy * slope_y[:, 1:]
    v_face = 0.5 * (v[:, :-1] + v[:, 1:])
    flux_y = torch.where(v_face >= 0, v_face * f_B, v_face * f_T)
    adv_y = torch.zeros_like(f)
    adv_y[:, 1:-1] = (flux_y[:, 1:] - flux_y[:, :-1]) / dy
    
    return adv_x + adv_y

def compute_velocity(psi, dx, dy):
    u = torch.zeros_like(psi)
    v = torch.zeros_like(psi)
    u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dy)
    v[1:-1, :] = +(psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
    u[0, :] = 0.0
    v[0, :] = (psi[1, :] - psi[0, :]) / dx
    return u, v

# ==============================================================================
# 3. Two-Parameter Generalized Ansatz: (alpha: Regularity, epsilon: Asymmetry)
# ==============================================================================
def generate_bifurcation_ansatz(X, Y, alpha=0.1, eps=0.0):
    """
    alpha: Regularity exponent at the corner (y -> 0).
           alpha = 1.0 -> Smooth sin(pi*y) behavior (Torque vanishes at corner)
           alpha = 0.1 -> Chen-Hou C^{1, alpha} singular torque persistence
    eps:   Symmetry-breaking perturbation amplitude (0.0 = strict symmetry)
    """
    Amp = 350.0
    cx, cy = 120.0, 25.0
    beta = 1.75
    
    denom = torch.clamp(1.0 + cx * (X**2) + cy * (Y**2), min=1.0)
    core_envelope = Amp * X * (1.0 - X**2) * (1.0 / torch.pow(denom, beta))
    
    # Corner regularity modulation
    y_reg = torch.pow(torch.clamp(Y, min=1e-7), alpha) * (1.0 - Y)
    
    # Symmetry-breaking perturbation
    asym_mode = 1.0 + eps * torch.cos(0.5 * np.pi * X) * torch.sin(np.pi * Y)
    
    theta = core_envelope * y_reg * asym_mode
    w = torch.zeros_like(theta)
    return w, theta

# ==============================================================================
# 4. Simulation Execution Routine
# ==============================================================================
def run_case(N, alpha, eps, label, max_t=0.25, cutoff_w=50000.0):
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / (N - 1), Ly / (N - 1)
    
    x = torch.linspace(0, Lx, N, dtype=torch.float64, device=device)
    y = torch.linspace(0, Ly, N, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    poisson = SpectralPoissonSolverDST(N, N, Lx, Ly, device=device)
    w, theta = generate_bifurcation_ansatz(X, Y, alpha=alpha, eps=eps)
    
    t = 0.0
    t_hist, w_hist, x_core_hist = [], [], []
    max_steps = 2500
    t0 = time.time()
    
    for step in range(max_steps):
        peak_w = torch.max(torch.abs(w)).item()
        t_hist.append(t)
        w_hist.append(peak_w)
        
        # Track physical core distance from wall
        idx_max = torch.argmax(torch.abs(w))
        ix_c = (idx_max // N).item()
        x_core_hist.append(ix_c * dx)
        
        if peak_w >= cutoff_w or t >= max_t:
            break
            
        psi = poisson(w)
        u, v = compute_velocity(psi, dx, dy)
        max_v = max(torch.max(torch.abs(u)).item(), torch.max(torch.abs(v)).item(), 1e-4)
        dt = min(0.30 * min(dx, dy) / max_v, 0.0006)
        
        d_theta_dx = torch.zeros_like(theta)
        d_theta_dx[1:-1, :] = (theta[2:, :] - theta[:-2, :]) / (2.0 * dx)
        d_theta_dx[0, :] = (theta[1, :] - theta[0, :]) / dx
        
        # RK2 Integrator
        dw1 = -tvd_advection_2d(w, u, v, dx, dy) + d_theta_dx
        dth1 = -tvd_advection_2d(theta, u, v, dx, dy)
        w_mid = w + 0.5 * dt * dw1
        theta_mid = theta + 0.5 * dt * dth1
        
        psi_mid = poisson(w_mid)
        u_mid, v_mid = compute_velocity(psi_mid, dx, dy)
        d_theta_dx_mid = torch.zeros_like(theta_mid)
        d_theta_dx_mid[1:-1, :] = (theta_mid[2:, :] - theta_mid[:-2, :]) / (2.0 * dx)
        d_theta_dx_mid[0, :] = (theta_mid[1, :] - theta_mid[0, :]) / dx
        
        dw2 = -tvd_advection_2d(w_mid, u_mid, v_mid, dx, dy) + d_theta_dx_mid
        dth2 = -tvd_advection_2d(theta_mid, u_mid, v_mid, dx, dy)
        
        w = w + dt * dw2
        theta = theta + dt * dth2
        w[0, :] = 0.0
        t += dt
        
    el = time.time() - t0
    final_pk = max(w_hist)
    print(f"  [+] {label:<35s} | Max ||w||: {final_pk:8.1f} | Final t: {t_hist[-1]:.4f}s | Wall x_c: {x_core_hist[-1]:.4f} ({el:.1f}s)")
    return np.array(t_hist), np.array(w_hist), np.array(x_core_hist), w.cpu().numpy(), theta.cpu().numpy()

# ==============================================================================
# 5. Execute 4-Case Phase Transition Audit (N = 256)
# ==============================================================================
N_RES = 256
CASES = [
    # (alpha, eps, color, label)
    (1.0, 0.00, '#1f77b4', 'Case 1: Smooth C^inf (alpha=1.0, eps=0)'),
    (0.1, 0.00, '#d62728', 'Case 2: Chen-Hou C^{1,a} (alpha=0.1, eps=0)'),
    (0.1, 0.05, '#ff7f0e', 'Case 3: Weak Asym (alpha=0.1, eps=0.05)'),
    (0.1, 0.20, '#2ca02c', 'Case 4: Strong Asym (alpha=0.1, eps=0.20)')
]

results = {}
print("\n" + "="*80)
print(f"  EULER SINGULARITY PHASE TRANSITION AUDIT (N = {N_RES} on {device})")
print("="*80)

for alpha, eps, col, lbl in CASES:
    t_h, w_h, x_c, w_f, th_f = run_case(N_RES, alpha, eps, lbl)
    results[lbl] = (t_h, w_h, x_c, w_f, th_f, col)

print("="*80 + "\n")

# ==============================================================================
# 6. Diagnostic Visualization (2x2 Panel)
# ==============================================================================
fig = plt.figure(figsize=(15, 10))

# Panel (a): Peak Vorticity Growth
ax1 = fig.add_subplot(2, 2, 1)
for _, _, col, lbl in CASES:
    t_h, w_h, _, _, _, _ = results[lbl]
    ax1.plot(t_h, w_h, color=col, lw=2.2, label=lbl.split(':')[0] + f' (Max: {max(w_h):.0f})')
ax1.set_title(r'(a) Vorticity Dynamics $\|\omega(t)\|_\infty$: $C^{1,\alpha}$ vs Asymmetry', fontsize=11, fontweight='bold')
ax1.set_xlabel('Time t (seconds)', fontsize=10)
ax1.set_ylabel(r'Maximum Vorticity $\|\omega\|_\infty$', fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc='upper left', fontsize=9.0)

# Panel (b): Wall-Normal Core Confinement vs Detachment
ax2 = fig.add_subplot(2, 2, 2)
for _, _, col, lbl in CASES:
    t_h, _, x_c, _, _, _ = results[lbl]
    ax2.plot(t_h, x_c, color=col, lw=2.0, label=lbl.split(':')[0])
ax2.axhline(y=0.0, color='black', lw=1.2, ls='-')
ax2.set_title(r'(b) Vortex Core Distance from Solid Wall $x_{\mathrm{core}}(t)$', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time t (seconds)', fontsize=10)
ax2.set_ylabel('Distance x', fontsize=10)
ax2.set_ylim([-0.01, 0.15])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9.0)

# Panel (c): Terminal Vorticity Topology (Case 2: Chen-Hou Enforced Blowup Candidate)
ax3 = fig.add_subplot(2, 2, 3)
w_case2 = results[CASES[1][3]][3]
im1 = ax3.imshow(w_case2.T, origin='lower', extent=[0, 1, 0, 1], cmap='inferno', aspect='auto')
ax3.set_title(r'(c) Case 2: Corner Pinning at $\alpha = 0.1, \epsilon = 0.0$', fontsize=11, fontweight='bold')
ax3.set_xlabel('Distance from Solid Wall x', fontsize=10)
ax3.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im1, ax=ax3)

# Panel (d): Terminal Vorticity Topology (Case 3: Asymmetry Roll-up & Detachment)
ax4 = fig.add_subplot(2, 2, 4)
w_case3 = results[CASES[2][3]][3]
im2 = ax4.imshow(w_case3.T, origin='lower', extent=[0, 1, 0, 1], cmap='inferno', aspect='auto')
ax4.set_title(r'(d) Case 3: K-H Roll-up & Wall Detachment at $\epsilon = 0.05$', fontsize=11, fontweight='bold')
ax4.set_xlabel('Distance from Solid Wall x', fontsize=10)
ax4.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im2, ax=ax4)

plt.tight_layout()
out_png = 'euler_bifurcation_symmetry_alpha_verdict.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"[+] Output benchmark figure saved: {out_png}")
plt.show()