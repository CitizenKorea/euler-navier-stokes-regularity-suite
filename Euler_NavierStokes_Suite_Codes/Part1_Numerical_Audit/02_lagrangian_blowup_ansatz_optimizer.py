#!/usr/bin/env python3
"""
File: 02_lagrangian_blowup_ansatz_optimizer.py
Description: Stage 1-2: Differentiable Lagrangian Inverse Design for Blowup Ansatz
Method:
  - Parameterized continuous initial profile theta_0(x, y; p)
  - End-to-end differentiable Runge-Kutta Boussinesq integrator
  - Adam optimizer backpropagating through time to maximize peak vorticity & BKM accumulation
Output:
  - lagrangian_blowup_optimization.png
  - optimal_blowup_ansatz_params.pt
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Computing Device for Inverse Design: {device}")

# ==============================================================================
# 1. Spatial Domain & Grid Setup (Optimized for Fast Autograd)
# ==============================================================================
Nx, Ny = 80, 80
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = torch.linspace(0, Lx, Nx, dtype=torch.float64, device=device)
y = torch.linspace(0, Ly, Ny, dtype=torch.float64, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

# ==============================================================================
# 2. Fully Differentiable Differential Operators
# ==============================================================================
def solve_stream_function(w):
    """Differentiable Poisson solver via Jacobi iterations with rigid wall at x=0."""
    psi = torch.zeros_like(w)
    for _ in range(25):
        psi_new = torch.zeros_like(psi)
        psi_new[1:-1, 1:-1] = 0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1] + 
                                      psi[1:-1, 2:] + psi[1:-1, :-2] + dx * dy * w[1:-1, 1:-1])
        psi_new[0, :] = 0.0          # Rigid boundary condition at solid wall x = 0
        psi_new[-1, :] = 0.0
        psi_new[:, 0] = psi_new[:, 1]# Symmetry Neumann at y = 0
        psi_new[:, -1] = 0.0
        psi = psi_new
    return psi

def compute_velocity(psi):
    """Computes u = -d_y(psi), v = +d_x(psi)."""
    u = torch.zeros_like(psi)
    v = torch.zeros_like(psi)
    u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dy)
    v[1:-1, :] = +(psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
    u[0, :] = 0.0
    v[0, :] = (psi[1, :] - psi[0, :]) / dx
    return u, v

def upwind_advection(field, u, v):
    """Differentiable upwind gradient advection: (u * d_x + v * d_y) field."""
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
# 3. Continuous Parameterized Blowup Profile Generator
# ==============================================================================
def generate_parameterized_ansatz(params):
    """
    params: [amp, cx, cy, x0, y0, skew]
    Generates smooth, boundary-compatible initial density field theta_0(x, y).
    """
    amp  = torch.exp(params[0]) * 100.0  # Positive amplitude
    cx   = torch.exp(params[1]) * 12.0   # Horizontal sharpness
    cy   = torch.exp(params[2]) * 15.0   # Vertical sharpness
    x0   = torch.sigmoid(params[3]) * 0.15  # Near solid boundary
    y0   = torch.sigmoid(params[4]) * 0.25  # Near corner stagnation
    skew = torch.tanh(params[5]) * 10.0  # Hyperbolic tilting
    
    r_sq = cx * (X - x0)**2 + cy * (Y - y0)**2 + skew * (X - x0) * (Y - y0)
    theta_0 = amp * (1.0 - X**2) * torch.exp(-torch.clamp(r_sq, min=0.0, max=50.0)) * torch.sin(np.pi * Y)
    w_0 = torch.zeros_like(theta_0)
    return w_0, theta_0

# ==============================================================================
# 4. Differentiable Forward Rollout & Loss Evaluation
# ==============================================================================
T_HORIZON = 0.15
DT = 0.002
N_STEPS = int(T_HORIZON / DT)

def rollout_and_evaluate(params):
    w, theta = generate_parameterized_ansatz(params)
    
    peak_w_list = []
    bkm_integral = 0.0
    
    for step in range(N_STEPS):
        peak_w = torch.norm(w, p=float('inf'))
        peak_w_list.append(peak_w)
        bkm_integral = bkm_integral + peak_w * DT
        
        # RK2 Integrator Step 1
        psi = solve_stream_function(w)
        u, v = compute_velocity(psi)
        
        d_theta_dx = torch.zeros_like(theta)
        d_theta_dx[1:-1, :] = (theta[2:, :] - theta[:-2, :]) / (2.0 * dx)
        d_theta_dx[0, :] = (theta[1, :] - theta[0, :]) / dx
        
        dw1 = -upwind_advection(w, u, v) + d_theta_dx
        dth1 = -upwind_advection(theta, u, v)
        
        w_mid = w + 0.5 * DT * dw1
        theta_mid = theta + 0.5 * DT * dth1
        
        # RK2 Integrator Step 2
        psi_mid = solve_stream_function(w_mid)
        u_mid, v_mid = compute_velocity(psi_mid)
        
        d_theta_dx_mid = torch.zeros_like(theta_mid)
        d_theta_dx_mid[1:-1, :] = (theta_mid[2:, :] - theta_mid[:-2, :]) / (2.0 * dx)
        d_theta_dx_mid[0, :] = (theta_mid[1, :] - theta_mid[0, :]) / dx
        
        dw2 = -upwind_advection(w_mid, u_mid, v_mid) + d_theta_dx_mid
        dth2 = -upwind_advection(theta_mid, u_mid, v_mid)
        
        w = w + DT * dw2
        theta = theta + DT * dth2
        w = w * (1.0 - (X == 0).double()) # Boundary pin
        
    final_peak_w = torch.norm(w, p=float('inf'))
    
    # Lagrangian Loss: Maximize log(final_peak_w) and BKM integral accumulation
    loss = -torch.log(final_peak_w + 1.0) - 0.05 * bkm_integral + 0.01 * torch.sum(params**2)
    return loss, final_peak_w.item(), bkm_integral.item(), w.detach(), theta.detach()

# ==============================================================================
# 5. Inverse Design Optimization Loop
# ==============================================================================
if __name__ == "__main__":
    print(f"[*] Starting Lagrangian Inverse Design for Blowup Profile (Horizon T = {T_HORIZON}s)...")
    
    # Initial parameters p0 (Baseline configuration)
    params = torch.zeros(6, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=0.08)
    
    # Baseline benchmark
    with torch.no_grad():
        _, base_peak, base_bkm, w_base, th_base = rollout_and_evaluate(params)
    print(f"[+] Baseline Canonical State -> Peak ||w||_inf = {base_peak:8.2f} | BKM = {base_bkm:8.4f}")
    
    history_peak = [base_peak]
    history_loss = []
    
    t_start = time.time()
    EPOCHS = 20
    
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss, peak_val, bkm_val, _, _ = rollout_and_evaluate(params)
        loss.backward()
        optimizer.step()
        
        history_peak.append(peak_val)
        history_loss.append(loss.item())
        
        print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {loss.item():8.4f} | Peak ||w||_inf: {peak_val:8.2f} | BKM: {bkm_val:8.4f}")
        
    print(f"[+] Optimization completed in {time.time() - t_start:.2f}s")
    
    # Optimal rollout for comparison
    with torch.no_grad():
        _, opt_peak, opt_bkm, w_opt, th_opt = rollout_and_evaluate(params)
    
    amplification = opt_peak / max(base_peak, 1e-6)
    print(f"\n==================================================================")
    print(f"[*] Inverse Design Verdict:")
    print(f"    - Baseline Peak Vorticity: {base_peak:.2f}")
    print(f"    - Optimized Peak Vorticity: {opt_peak:.2f} ({amplification:.2f}x Amplification)")
    print(f"    - Optimal Parameter Vector p*: {params.detach().cpu().numpy().round(3)}")
    print(f"==================================================================")
    
    # Save optimal parameter checkpoint
    torch.save(params.detach().cpu(), 'optimal_blowup_ansatz_params.pt')
    
    # ==============================================================================
    # 6. Diagnostic Visualization (Baseline vs. Optimal Inverse-Designed Ansatz)
    # ==============================================================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    
    # Panel (a): Optimization Convergence
    ax1.plot(range(len(history_peak)), history_peak, 'ro-', lw=2.0)
    ax1.set_title(r'(a) Peak Vorticity $\|\omega(T^*)\|_\infty$ Growth via Autograd', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Optimization Epoch', fontsize=10)
    ax1.set_ylabel(r'$\|\omega(T^*)\|_\infty$', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel (b): Loss Function Trajectory
    ax2.plot(range(1, len(history_loss) + 1), history_loss, 'b.-', lw=1.8)
    ax2.set_title('(b) Lagrangian Objective Trajectory', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Objective Value', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel (c): Baseline Initial Density Profile
    _, th_init_base = generate_parameterized_ansatz(torch.zeros(6, dtype=torch.float64, device=device))
    im1 = ax3.imshow(th_init_base.detach().cpu().numpy().T, origin='lower', extent=[0, Lx, 0, Ly], cmap='coolwarm', aspect='auto')
    ax3.set_title(r'(c) Baseline Canonical Ansatz $\theta_0(x, y)$', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Distance from Wall x', fontsize=10)
    ax3.set_ylabel('Vertical Coordinate y', fontsize=10)
    fig.colorbar(im1, ax=ax3)
    
    # Panel (d): Inversely Designed Optimal Blowup Profile
    _, th_init_opt = generate_parameterized_ansatz(params)
    im2 = ax4.imshow(th_init_opt.detach().cpu().numpy().T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', aspect='auto')
    ax4.set_title(r'(d) Inversely Designed Optimal Blowup Ansatz $\theta_0^*(x, y)$', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Distance from Wall x', fontsize=10)
    ax4.set_ylabel('Vertical Coordinate y', fontsize=10)
    fig.colorbar(im2, ax=ax4)
    
    out_png = 'lagrangian_blowup_optimization.png'
    plt.savefig(out_png, dpi=300)
    print(f"[+] Optimization benchmark exported: {out_png}")
    plt.show()