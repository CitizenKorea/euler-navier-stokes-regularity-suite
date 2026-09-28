# ==============================================================================
# File: 09_corner_stagnation_blowup_verdict.py
# Description: Final Verdict: Corner-Pinned Stagnation Blowup Verification (N = 256)
# Physics:
#   - Origin-pinned Hou-Luo Ansatz at (x0, y0) = (0, 0) to eliminate convective drift
#   - Exact Spectral DST-I Poisson Solver via GPU FFT (Zero Iteration, O(10^-15))
#   - Real-time Core Drift Tracking & BKM Scaling Law Power-Fit Verdict
# ==============================================================================

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

assert torch.cuda.is_available(), "T4 GPU 런타임을 활성화하세요."
device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
print(f"[*] Final Blowup Verdict running on: {device} ({gpu_name})")

# ==============================================================================
# 1. Exact Spectral DST-I Poisson Solver Class
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
# 2. Kinematics & High-Order Advection
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
# 3. Corner-Pinned Stagnation Model Setup (N = 256)
# ==============================================================================
N = 256
Lx, Ly = 1.0, 1.0
dx = Lx / (N - 1)
dy = Ly / (N - 1)

x = torch.linspace(0, Lx, N, dtype=torch.float64, device=device)
y = torch.linspace(0, Ly, N, dtype=torch.float64, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Hou-Luo pure stagnation ansatz pinned strictly at (0, 0)
Amp = 350.0       # Boundary gradient pump amplitude
cx = 120.0        # Sharp radial steepness near wall
cy = 25.0         # Axial aspect ratio
alpha = 1.75      # Algebraic decay power

denominator = torch.clamp(1.0 + cx * (X**2) + cy * (Y**2), min=1.0)
theta = Amp * X * (1.0 - X**2) * (1.0 / torch.pow(denominator, alpha)) * torch.sin(np.pi * Y)
w = torch.zeros_like(theta)

poisson = SpectralPoissonSolverDST(N, N, Lx, Ly, device=device)

# ==============================================================================
# 4. Adaptive Forward Simulation & Core Tracking
# ==============================================================================
max_steps = 4000
cutoff_w = 40000.0
max_t = 0.25

t = 0.0
t_hist, w_peak_hist, bkm_hist = [], [], []
x_core_hist, y_core_hist = [], []
bkm_sum = 0.0

print("\n" + "="*70)
print(f"[*] Executing Corner-Pinned Stagnation Verdict Simulation (N = {N})...")
print(f"    Ansatz: Pinned at (0, 0) | Amp = {Amp} | cx = {cx} | cy = {cy}")
print("="*70)

t0_start = time.time()
checkpoint_interval = 500

for step in range(max_steps):
    peak_w = torch.max(torch.abs(w)).item()
    t_hist.append(t)
    w_peak_hist.append(peak_w)
    bkm_hist.append(bkm_sum)
    
    # Track physical core coordinate (x_core, y_core)
    idx_max = torch.argmax(torch.abs(w))
    ix_c = (idx_max // N).item()
    iy_c = (idx_max % N).item()
    x_core_hist.append(ix_c * dx)
    y_core_hist.append(iy_c * dy)
    
    if peak_w >= cutoff_w:
        print(f"\n[!] RUNAWAY SINGULARITY REACHED: ||w|| = {peak_w:.1f} at t = {t:.5f}s!")
        break
    if t >= max_t:
        print(f"\n[*] Target physical horizon t = {t:.5f}s reached.")
        break
        
    psi = poisson(w)
    u, v = compute_velocity(psi, dx, dy)
    
    max_vel = max(torch.max(torch.abs(u)).item(), torch.max(torch.abs(v)).item(), 1e-4)
    dt = min(0.30 * min(dx, dy) / max_vel, 0.0006)
    bkm_sum += peak_w * dt
    
    d_theta_dx = torch.zeros_like(theta)
    d_theta_dx[1:-1, :] = (theta[2:, :] - theta[:-2, :]) / (2.0 * dx)
    d_theta_dx[0, :] = (theta[1, :] - theta[0, :]) / dx
    
    # RK2 Integration
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
    
    if (step + 1) % checkpoint_interval == 0:
        print(f"  Step {step+1:4d} | t = {t:.4f}s | dt = {dt:.6f}s | Peak ||w||: {peak_w:7.1f} | Core: ({x_core_hist[-1]:.4f}, {y_core_hist[-1]:.4f})")

elapsed_sim = time.time() - t0_start
print("="*70)
print(f"[+] Simulation completed in {elapsed_sim:.2f}s")
print(f"    Terminal Time: {t_hist[-1]:.5f}s | Max Peak ||w||: {max(w_peak_hist):.1f}")

# ==============================================================================
# 5. Beale-Kato-Majda (BKM) Scaling Law Fitting
# ==============================================================================
t_arr = np.array(t_hist)
w_arr = np.array(w_peak_hist)
mask = w_arr > 1500.0
t_fit = t_arr[mask]
w_fit = w_arr[mask]

best_r2, best_T, best_gamma = -1e9, None, None
if len(t_fit) > 10:
    t_cands = np.linspace(t_fit[-1] + 1e-5, t_fit[-1] + 0.015, 600)
    for T_cand in t_cands:
        dt_v = T_cand - t_fit
        if np.any(dt_v <= 0):
            continue
        log_dt = np.log(dt_v)
        log_w = np.log(w_fit)
        slope, intercept = np.polyfit(log_dt, log_w, 1)
        pred = slope * log_dt + intercept
        ss_res = np.sum((log_w - pred)**2)
        ss_tot = np.sum((log_w - np.mean(log_w))**2)
        r2 = 1.0 - (ss_res / ss_tot)
        if r2 > best_r2:
            best_r2 = r2
            best_T = T_cand
            best_gamma = -slope

print("\n" + "="*70)
print("[*] FINAL VERDICT REPORT:")
print(f"    - Maximum Attained Peak: {max(w_arr):.1f}")
if best_gamma is not None and best_r2 > 0.85:
    print(f"    - Estimated Singularity Time T*: {best_T:.6f} s")
    print(f"    - BKM Scaling Exponent gamma:    {best_gamma:.4f}")
    print(f"    - Goodness of Fit (R^2):         {best_r2:.5f}")
    if best_gamma >= 1.0 and max(w_arr) >= 20000.0:
        print("    >> VERDICT: GENUINE FINITE-TIME BLOWUP CONFIRMED! (gamma >= 1.0)")
    else:
        print("    >> VERDICT: REGULARIZED / SATURATED (No non-integrable blowup)")
else:
    print("    >> VERDICT: INSUFFICIENT GROWTH / BOUNDED REGULARITY")
print("="*70 + "\n")

# ==============================================================================
# 6. Diagnostic Visualization (2x2 Panel)
# ==============================================================================
fig = plt.figure(figsize=(14, 10))

# Panel (a): Vorticity Runaway Trajectory
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(t_arr, w_arr, 'r-', lw=2.4, label='Peak Vorticity')
ax1.axhline(y=10000.0, color='gray', linestyle=':', label='Threshold (10,000)')
ax1.set_title(r'(a) Corner-Pinned Vorticity Evolution $\|\omega(t)\|_\infty$', fontsize=11, fontweight='bold')
ax1.set_xlabel('Time t (seconds)', fontsize=10)
ax1.set_ylabel(r'Maximum Vorticity $\|\omega\|_\infty$', fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc='upper left', fontsize=9.0)

# Panel (b): Core Spatial Trajectory (Drift Suppression)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x_core_hist, y_core_hist, 'b.-', lw=1.5, markersize=3, label='Core Trajectory')
ax2.plot(x_core_hist[0], y_core_hist[0], 'go', markersize=8, label='Start Core')
ax2.plot(x_core_hist[-1], y_core_hist[-1], 'ro', markersize=8, label='Final Core')
ax2.set_xlim([0, 0.25])
ax2.set_ylim([0, 0.35])
ax2.set_title('(b) Vortex Core Confinement (Wall & Stagnation Point)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Distance from Solid Wall x', fontsize=10)
ax2.set_ylabel('Vertical Coordinate y', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9.0)

# Panel (c): BKM Scaling Law Fit
ax3 = fig.add_subplot(2, 2, 3)
if best_gamma is not None and best_r2 > 0.85:
    dt_plot = best_T - t_fit
    ax3.loglog(dt_plot, w_fit, 'ro', markersize=4, label='Simulation')
    fit_line = np.exp(np.polyval([-best_gamma, np.log(w_fit[-1]) + best_gamma * np.log(dt_plot[-1])], np.log(dt_plot)))
    ax3.loglog(dt_plot, fit_line, 'k--', lw=2.0, 
               label=rf'Fit: $(T^* - t)^{{-{best_gamma:.2f}}}$ ($R^2 = {best_r2:.4f}$)')
    ax3.set_title(rf'(c) BKM Self-Similar Scaling ($\gamma = {best_gamma:.2f}$)', fontsize=11, fontweight='bold')
else:
    ax3.plot(t_arr, bkm_hist, 'g-', lw=2.0)
    ax3.set_title('(c) Cumulative BKM Integral Accumulation', fontsize=11, fontweight='bold')
ax3.set_xlabel(r'Time to Singularity $(T^* - t)$ (seconds)', fontsize=10)
ax3.set_ylabel(r'Vorticity $\|\omega\|_\infty$', fontsize=10)
ax3.grid(True, which="both", alpha=0.3)
ax3.legend(loc='lower left', fontsize=9.0)

# Panel (d): Final Vorticity Field Topology
ax4 = fig.add_subplot(2, 2, 4)
im = ax4.imshow(w.cpu().numpy().T, origin='lower', extent=[0, Lx, 0, Ly], cmap='inferno', aspect='auto')
ax4.set_title(r'(d) Terminal Singular Core Topology at $t = ' + f'{t_arr[-1]:.4f}' + r'\,\mathrm{s}$', fontsize=11, fontweight='bold')
ax4.set_xlabel('Distance from Solid Wall x', fontsize=10)
ax4.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im, ax=ax4)

out_png = 'corner_stagnation_verdict_result.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"[+] Verdict benchmark figure exported: {out_png}")
plt.show()