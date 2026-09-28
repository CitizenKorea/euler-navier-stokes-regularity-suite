# ==============================================================================
# Google Colab T4 GPU: Ultra-Precision Spectral Hou-Luo Blowup Verifier
# Physics: 2D Boussinesq System with Swirl at Solid Boundary
# Algorithms:
#   - Exact Spectral 2D Discrete Sine Transform (DST-I) Poisson Solver via FFT
#   - 2nd-Order TVD Advection with Minmod Flux Limiter
#   - Multi-Grid Resolution Benchmark: N = [128, 256, 512] on CUDA
# ==============================================================================

import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Device Setup
assert torch.cuda.is_available(), "T4 GPU 런타임이 활성화되지 않았습니다. '런타임 유형 변경'에서 T4 GPU를 선택하세요."
device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
print(f"[*] Active Compute Device: {device} ({gpu_name})")

# 2. Optimal Blowup Ansatz Parameters (Lagrangian Inverse Designed)
p_opt = torch.tensor([1.652, 1.643, -1.527, 0.276, 1.518, 0.970], dtype=torch.float64, device=device)

# ==============================================================================
# Exact Spectral DST-I Poisson Solver Class (Zero Iteration, Machine Precision)
# ==============================================================================
class SpectralPoissonSolverDST:
    def __init__(self, Nx, Ny, Lx=1.0, Ly=1.0, device=device):
        self.Nx, self.Ny = Nx, Ny
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.Mx, self.My = Nx - 2, Ny - 2
        
        # Discrete Laplacian eigenvalues for 5-point stencil with Dirichlet BC
        jx = torch.arange(1, self.Mx + 1, dtype=torch.float64, device=device)
        jy = torch.arange(1, self.My + 1, dtype=torch.float64, device=device)
        lam_x = 2.0 * (1.0 - torch.cos(jx * np.pi / (self.Mx + 1))) / (self.dx**2)
        lam_y = 2.0 * (1.0 - torch.cos(jy * np.pi / (self.My + 1))) / (self.dy**2)
        self.LAM = lam_x[:, None] + lam_y[None, :]
        self.norm_factor = 1.0 / ((self.Mx + 1) * (self.My + 1))
        
    def solve(self, w):
        """Solves -Laplace(psi) = w with psi = 0 at boundaries in O(N log N) using FFT."""
        w_int = w[1:-1, 1:-1]
        
        # Forward DST-I via Odd Periodic Extension
        ext = torch.zeros(2 * (self.Mx + 1), 2 * (self.My + 1), dtype=torch.float64, device=device)
        ext[1:self.Mx+1, 1:self.My+1] = w_int
        ext[self.Mx+2:, 1:self.My+1] = -torch.flip(w_int, dims=[0])
        ext[1:self.Mx+1, self.My+2:] = -torch.flip(w_int, dims=[1])
        ext[self.Mx+2:, self.My+2:] = torch.flip(w_int, dims=[0, 1])
        
        fft_w = torch.fft.fftn(ext)
        dst_w = -0.25 * torch.real(fft_w[1:self.Mx+1, 1:self.My+1])
        
        # Invert Laplacian in frequency domain
        psi_hat = dst_w / self.LAM
        
        # Inverse DST-I via Odd Extension
        ext_psi = torch.zeros_like(ext)
        ext_psi[1:self.Mx+1, 1:self.My+1] = psi_hat
        ext_psi[self.Mx+2:, 1:self.My+1] = -torch.flip(psi_hat, dims=[0])
        ext_psi[1:self.Mx+1, self.My+2:] = -torch.flip(psi_hat, dims=[1])
        ext_psi[self.Mx+2:, self.My+2:] = torch.flip(psi_hat, dims=[0, 1])
        
        fft_psi = torch.fft.fftn(ext_psi)
        psi_int = -self.norm_factor * torch.real(fft_psi[1:self.Mx+1, 1:self.My+1])
        
        psi = torch.zeros(self.Nx, self.Ny, dtype=torch.float64, device=device)
        psi[1:-1, 1:-1] = psi_int
        return psi

# ==============================================================================
# High-Order TVD Advection Operator (Minmod Limiter)
# ==============================================================================
def minmod(a, b):
    return torch.sign(a) * torch.clamp(torch.min(torch.abs(a), torch.abs(b)), min=0.0) * (torch.sign(a) == torch.sign(b)).double()

def tvd_advection_2d(f, u, v, dx, dy):
    """2nd-Order TVD advection with minmod flux limiter to suppress numerical diffusion."""
    # x-direction slopes
    df_dx = (f[1:, :] - f[:-1, :]) / dx
    slope_x = minmod(df_dx[:-1, :], df_dx[1:, :])
    slope_x = torch.cat([df_dx[:1, :], slope_x, df_dx[-1:, :]], dim=0)
    
    f_L = f[:-1, :] + 0.5 * dx * slope_x[:-1, :]
    f_R = f[1:, :] - 0.5 * dx * slope_x[1:, :]
    u_face = 0.5 * (u[:-1, :] + u[1:, :])
    flux_x = torch.where(u_face >= 0, u_face * f_L, u_face * f_R)
    adv_x = torch.zeros_like(f)
    adv_x[1:-1, :] = (flux_x[1:, :] - flux_x[:-1, :]) / dx
    
    # y-direction slopes
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

# ==============================================================================
# Simulation Engine per Grid Resolution
# ==============================================================================
def run_simulation(N, cutoff_w=50000.0, max_t=0.25):
    Nx = Ny = N
    Lx = Ly = 1.0
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    
    x = torch.linspace(0, Lx, Nx, dtype=torch.float64, device=device)
    y = torch.linspace(0, Ly, Ny, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Instantiate Spectral Solver
    poisson = SpectralPoissonSolverDST(Nx, Ny, Lx, Ly, device=device)
    
    # Generate continuous ansatz
    amp  = torch.exp(p_opt[0]) * 100.0
    cx   = torch.exp(p_opt[1]) * 12.0
    cy   = torch.exp(p_opt[2]) * 15.0
    x0   = torch.sigmoid(p_opt[3]) * 0.15
    y0   = torch.sigmoid(p_opt[4]) * 0.25
    skew = torch.tanh(p_opt[5]) * 10.0
    
    r_sq = cx * (X - x0)**2 + cy * (Y - y0)**2 + skew * (X - x0) * (Y - y0)
    theta = amp * (1.0 - X**2) * torch.exp(-torch.clamp(r_sq, min=0.0, max=50.0)) * torch.sin(np.pi * Y)
    w = torch.zeros_like(theta)
    
    def get_velocity(psi):
        u = torch.zeros_like(psi)
        v = torch.zeros_like(psi)
        u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dy)
        v[1:-1, :] = +(psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
        u[0, :] = 0.0
        v[0, :] = (psi[1, :] - psi[0, :]) / dx
        return u, v

    t = 0.0
    t_hist, w_hist, bkm_hist, n_core_hist = [], [], [], []
    bkm_sum = 0.0
    
    max_steps = int(1200 * (N / 128))
    t0 = time.time()
    
    for step in range(max_steps):
        peak_w = torch.max(torch.abs(w)).item()
        
        # Measure core FWHM along x-axis
        w_abs = torch.abs(w)
        idx = torch.argmax(w_abs)
        iy = idx % Ny
        slice_x = w_abs[:, iy]
        above_half = torch.where(slice_x >= peak_w * 0.5)[0]
        n_core = float(len(above_half))
        
        t_hist.append(t)
        w_hist.append(peak_w)
        bkm_hist.append(bkm_sum)
        n_core_hist.append(n_core)
        
        if peak_w >= cutoff_w or t >= max_t:
            break
            
        psi = poisson.solve(w)
        u, v = get_velocity(psi)
        
        max_vel = max(torch.max(torch.abs(u)).item(), torch.max(torch.abs(v)).item(), 1e-4)
        dt = min(0.30 * min(dx, dy) / max_vel, 0.001)
        bkm_sum += peak_w * dt
        
        d_theta_dx = torch.zeros_like(theta)
        d_theta_dx[1:-1, :] = (theta[2:, :] - theta[:-2, :]) / (2.0 * dx)
        d_theta_dx[0, :] = (theta[1, :] - theta[0, :]) / dx
        
        # RK2 Step 1
        dw1 = -tvd_advection_2d(w, u, v, dx, dy) + d_theta_dx
        dth1 = -tvd_advection_2d(theta, u, v, dx, dy)
        w_mid = w + 0.5 * dt * dw1
        theta_mid = theta + 0.5 * dt * dth1
        
        # RK2 Step 2
        psi_mid = poisson.solve(w_mid)
        u_mid, v_mid = get_velocity(psi_mid)
        
        d_theta_dx_mid = torch.zeros_like(theta_mid)
        d_theta_dx_mid[1:-1, :] = (theta_mid[2:, :] - theta_mid[:-2, :]) / (2.0 * dx)
        d_theta_dx_mid[0, :] = (theta_mid[1, :] - theta_mid[0, :]) / dx
        
        dw2 = -tvd_advection_2d(w_mid, u_mid, v_mid, dx, dy) + d_theta_dx_mid
        dth2 = -tvd_advection_2d(theta_mid, u_mid, v_mid, dx, dy)
        
        w = w + dt * dw2
        theta = theta + dt * dth2
        w[0, :] = 0.0
        t += dt
        
    return (np.array(t_hist), np.array(w_hist), np.array(bkm_hist), 
            np.array(n_core_hist), time.time() - t0, w.cpu().numpy(), theta.cpu().numpy(), dx)

# ==============================================================================
# Execute Multi-Resolution Spectral Benchmark
# ==============================================================================
RESOLUTIONS = [128, 256, 512]
results = {}
scaling_data = {}

print("\n" + "="*75)
print(f"[*] Running Spectral Poisson + TVD Multi-Grid Suite on {gpu_name}")
print("="*75)

for N in RESOLUTIONS:
    th, wh, bkm, n_core, el_s, w_final, th_final, dx_v = run_simulation(N)
    results[N] = (th, wh, bkm, n_core, w_final, th_final, dx_v)
    
    # Fit Beale-Kato-Majda: ||w|| ~ C / (T* - t)^gamma
    mask = wh > 2000.0
    t_blow = th[mask]
    w_blow = wh[mask]
    
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
                
    scaling_data[N] = (best_T, best_gamma, best_r2, max(wh))
    print(f"  [+] Grid {N:3d}x{N:3d} (dx={dx_v:.5f}) | Peak ||w||: {max(wh):8.1f} | "
          f"T*: {best_T:.5f}s | gamma: {best_gamma:.3f} (R^2={best_r2:.4f}) | Time: {el_s:.2f}s")

print("="*75)
print("[*] Spectral Continuum Extrapolation Summary:")
print("    N       dx        T*(s)     gamma      R^2       Peak ||w||")
for N in RESOLUTIONS:
    T_s, gam, r2, pk = scaling_data[N]
    dx_v = results[N][6]
    print(f"   {N:3d}   {dx_v:.5f}   {T_s:.5f}   {gam:.4f}    {r2:.5f}   {pk:8.1f}")
print("="*75 + "\n")

# ==============================================================================
# Publication-Grade Diagnostic Visualization (2x2 Panel)
# ==============================================================================
fig = plt.figure(figsize=(14, 10))
colors = {128: '#1f77b4', 256: '#2ca02c', 512: '#d62728'}

# Subplot 1: Vorticity Trajectory
ax1 = fig.add_subplot(2, 2, 1)
for N in RESOLUTIONS:
    th, wh, _, _, _, _, _ = results[N]
    ax1.plot(th, wh, color=colors[N], lw=2.2, label=f'N = {N} (dx = {1.0/(N-1):.4f})')
ax1.set_title(r'(a) Spectral Vorticity Growth $\|\omega(t)\|_\infty$', fontsize=11, fontweight='bold')
ax1.set_xlabel('Time t (seconds)', fontsize=10)
ax1.set_ylabel(r'Maximum Vorticity $\|\omega\|_\infty$', fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc='upper left', fontsize=9.5)

# Subplot 2: Cumulative BKM Integral
ax2 = fig.add_subplot(2, 2, 2)
for N in RESOLUTIONS:
    th, _, bkm, _, _, _, _ = results[N]
    ax2.plot(th, bkm, color=colors[N], lw=2.2, label=f'N = {N}')
ax2.set_title(r'(b) Cumulative BKM Integral $\int_0^t \|\omega(s)\|_\infty ds$', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time t (seconds)', fontsize=10)
ax2.set_ylabel('BKM Accumulation', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=9.5)

# Subplot 3: Core Resolution Metric
ax3 = fig.add_subplot(2, 2, 3)
for N in RESOLUTIONS:
    th, _, _, n_core, _, _, _ = results[N]
    ax3.plot(th, n_core, color=colors[N], lw=2.0, label=f'N = {N}')
ax3.axhline(y=4.0, color='gray', linestyle='--', lw=1.8, label='Adequate Margin Threshold (N_core = 4)')
ax3.set_title(r'(c) Vortex Core Resolution $N_{\mathrm{core}} = \mathrm{FWHM}_x / \Delta x$', fontsize=11, fontweight='bold')
ax3.set_xlabel('Time t (seconds)', fontsize=10)
ax3.set_ylabel('Grid Points across Core', fontsize=10)
ax3.set_ylim([0, 35])
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right', fontsize=9.0)

# Subplot 4: Scaling Exponent Continuum Limit (h -> 0)
ax4 = fig.add_subplot(2, 2, 4)
inv_N = [1.0 / N for N in RESOLUTIONS]
gamma_vals = [scaling_data[N][1] for N in RESOLUTIONS]

poly = np.polyfit(inv_N, gamma_vals, 1)
h_dense = np.linspace(0.0, max(inv_N) * 1.1, 100)
gamma_extrap = np.polyval(poly, h_dense)
gamma_inf = poly[1]

ax4.scatter(inv_N, gamma_vals, color='red', s=70, zorder=5, label='Simulation Values')
ax4.plot(h_dense, gamma_extrap, 'k--', lw=2.0, 
         label=rf'Continuum Extrapolation ($h \to 0$): $\gamma_\infty \approx {gamma_inf:.3f}$')
ax4.axhline(y=1.0, color='blue', linestyle=':', lw=1.8, label=r'BKM Blowup Threshold ($\gamma = 1.0$)')
ax4.set_title(r'(d) Continuum Limit of Scaling Exponent $\gamma(h \to 0)$', fontsize=11, fontweight='bold')
ax4.set_xlabel(r'Grid Spacing $h = 1/N$', fontsize=10)
ax4.set_ylabel(r'Scaling Exponent $\gamma$', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='lower left', fontsize=9.0)

out_png = 'colab_spectral_convergence_test.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"[+] Diagnostic benchmark successfully exported: {out_png}")
plt.show()