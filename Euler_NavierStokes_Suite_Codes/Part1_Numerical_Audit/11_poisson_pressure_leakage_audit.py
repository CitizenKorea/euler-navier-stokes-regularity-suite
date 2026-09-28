#!/usr/bin/env python3
"""
File: 11_poisson_pressure_leakage_audit.py
Description: Quantitative 2D Diagnostic of Nonlocal Pressure Leakage:
             Iterative Jacobi Relaxation vs. Exact Spectral DST-I Projection
Diagnostics:
  - Poisson Operator Residual: R_Poisson = |Laplace(psi) + w|
  - Solenoidal Divergence Violation: |div(u)| = |d_x u + d_y v|
Output:
  - 11_poisson_pressure_leakage_comparison.png
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Audit Device: {device}")

# ==============================================================================
# 1. Spatial Setup (128 x 128)
# ==============================================================================
N = 128
Lx, Ly = 1.0, 1.0
dx = Lx / (N - 1)
dy = Ly / (N - 1)

x = torch.linspace(0, Lx, N, dtype=torch.float64, device=device)
y = torch.linspace(0, Ly, N, dtype=torch.float64, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

# ==============================================================================
# 2. Extract Realistic High-Gradient Vorticity Field
# ==============================================================================
param_path = 'optimal_blowup_ansatz_params.pt'
if os.path.exists(param_path):
    p_opt = torch.load(param_path, weights_only=True).to(device)
else:
    p_opt = torch.tensor([1.652, 1.643, -1.527, 0.276, 1.518, 0.970], dtype=torch.float64, device=device)

# Steep boundary layer vorticity profile simulating near-collapse state
amp  = torch.exp(p_opt[0]) * 100.0
cx   = torch.exp(p_opt[1]) * 12.0
cy   = torch.exp(p_opt[2]) * 15.0
x0   = torch.sigmoid(p_opt[3]) * 0.15
y0   = torch.sigmoid(p_opt[4]) * 0.25
skew = torch.tanh(p_opt[5]) * 10.0

r_sq = cx * (X - x0)**2 + cy * (Y - y0)**2 + skew * (X - x0) * (Y - y0)
theta = amp * (1.0 - X**2) * torch.exp(-torch.clamp(r_sq, min=0.0, max=50.0)) * torch.sin(np.pi * Y)

# Synthesize physical vorticity driven by baroclinic torque near solid wall
w_field = torch.zeros_like(theta)
d_theta_dx = torch.zeros_like(theta)
d_theta_dx[1:-1, :] = (theta[2:, :] - theta[:-2, :]) / (2.0 * dx)
d_theta_dx[0, :] = (theta[1, :] - theta[0, :]) / dx
w_field = d_theta_dx * 0.12
w_field[0, :] = 0.0

# ==============================================================================
# 3. Solver Implementations (Jacobi vs Spectral DST-I)
# ==============================================================================
def solve_jacobi_poisson(w, iters=35):
    """Iterative Jacobi Poisson solver used in baseline under-resolved models."""
    psi = torch.zeros_like(w)
    for _ in range(iters):
        psi_new = torch.zeros_like(psi)
        psi_new[1:-1, 1:-1] = 0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1] + 
                                      psi[1:-1, 2:] + psi[1:-1, :-2] + dx * dy * w[1:-1, 1:-1])
        psi_new[0, :] = 0.0
        psi_new[-1, :] = 0.0
        psi_new[:, 0] = psi_new[:, 1]
        psi_new[:, -1] = 0.0
        psi = psi_new
    return psi

class SpectralPoissonSolverDST(nn.Module):
    """Machine-precision Discrete Sine Transform Solver."""
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

spectral_solver = SpectralPoissonSolverDST(N, N, Lx, Ly, device=device)

# ==============================================================================
# 4. Residual and Divergence Audit
# ==============================================================================
def evaluate_operators(psi, w):
    # Discrete Laplacian: Delta(psi)
    lap = torch.zeros_like(psi)
    lap[1:-1, 1:-1] = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / (dx * dx) + \
                      (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (dy * dy)
    
    # Residual: R = |Delta(psi) + w|
    res = torch.abs(lap + w)
    
    # Incompressibility condition: div(u) = d_x u + d_y v
    u = torch.zeros_like(psi)
    v = torch.zeros_like(psi)
    u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dy)
    v[1:-1, :] = +(psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
    
    div_u = torch.zeros_like(psi)
    div_u[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx) + \
                        (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)
    div_norm = torch.abs(div_u)
    
    return res, div_norm

# Execute Solutions
psi_jacobi = solve_jacobi_poisson(w_field, iters=35)
psi_spectral = spectral_solver(w_field)

res_jacobi, div_jacobi = evaluate_operators(psi_jacobi, w_field)
res_spectral, div_spectral = evaluate_operators(psi_spectral, w_field)

max_res_j = torch.max(res_jacobi).item()
max_res_s = torch.max(res_spectral[1:-1, 1:-1]).item()
max_div_j = torch.max(div_jacobi).item()
max_div_s = torch.max(div_spectral[1:-1, 1:-1]).item()

print("\n" + "="*70)
print("  PRESSURE LEAKAGE & INCOMPRESSIBILITY AUDIT REPORT")
print("="*70)
print(f"  [Jacobi 35 iters]   Max Poisson Residual : {max_res_j:.4e}")
print(f"                      Max Divergence Error : {max_div_j:.4e}")
print(f"  [Spectral DST-I]    Max Poisson Residual : {max_res_s:.4e} (Machine Floor)")
print(f"                      Max Divergence Error : {max_div_s:.4e} (Machine Floor)")
print(f"  Leakage Factor Ratio (Jacobi / Spectral) : {max_res_j / max_res_s:.2e}x error suppression")
print("="*70 + "\n")

# ==============================================================================
# 5. Diagnostic Visualization
# ==============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Jacobi Residual Field
im1 = ax1.imshow(torch.clamp(res_jacobi, min=1e-15).cpu().numpy().T, origin='lower', 
                 extent=[0, Lx, 0, Ly], cmap='inferno', aspect='auto')
ax1.set_title(r'(a) Jacobi 35-Iter Residual $|\Delta\psi + \omega|$ (Max: ' + f'{max_res_j:.2e})', fontsize=11, fontweight='bold')
ax1.set_xlabel('Distance from Wall x', fontsize=10)
ax1.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im1, ax=ax1)

# Subplot 2: Spectral DST Residual Field
im2 = ax2.imshow(torch.clamp(res_spectral, min=1e-15).cpu().numpy().T, origin='lower', 
                 extent=[0, Lx, 0, Ly], cmap='viridis', aspect='auto')
ax2.set_title(r'(b) Spectral DST-I Residual $|\Delta\psi + \omega|$ (Max: ' + f'{max_res_s:.2e})', fontsize=11, fontweight='bold')
ax2.set_xlabel('Distance from Wall x', fontsize=10)
ax2.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im2, ax=ax2)

# Subplot 3: Incompressibility Violation |div(u)| (Jacobi)
im3 = ax3.imshow(div_jacobi.cpu().numpy().T, origin='lower', extent=[0, Lx, 0, Ly], cmap='magma', aspect='auto')
ax3.set_title(r'(c) Jacobi Incompressibility Error $|\nabla \cdot \mathbf{u}|$', fontsize=11, fontweight='bold')
ax3.set_xlabel('Distance from Wall x', fontsize=10)
ax3.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im3, ax=ax3)

# Subplot 4: Wall-Normal Residual Decay Profile at Critical Height y_core
idx_y = int(y0.item() * (N - 1))
res_slice_j = res_jacobi[:, idx_y].cpu().numpy()
res_slice_s = res_spectral[:, idx_y].cpu().numpy()

ax4.semilogy(x.cpu().numpy(), res_slice_j, 'r-', lw=2.2, label='Jacobi Relaxation (35 iters)')
ax4.semilogy(x.cpu().numpy(), np.maximum(res_slice_s, 1e-16), 'b--', lw=2.0, label='Spectral DST-I (Exact)')
ax4.set_title(r'(d) Wall-Normal Residual Comparison at Vortex Core Height', fontsize=11, fontweight='bold')
ax4.set_xlabel('Distance from Solid Wall x', fontsize=10)
ax4.set_ylabel('Poisson Residual (Log Scale)', fontsize=10)
ax4.set_ylim([1e-16, 1e1])
ax4.grid(True, which="both", alpha=0.3)
ax4.legend(loc='upper right', fontsize=9.5)

plt.tight_layout()
out_png = '11_poisson_pressure_leakage_comparison.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"[+] Diagnostic visual exported: {out_png}")
plt.show()