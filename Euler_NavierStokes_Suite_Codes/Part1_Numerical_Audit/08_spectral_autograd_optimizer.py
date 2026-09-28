# ==============================================================================
# Google Colab T4 GPU: Ultra-Stable Spectral Inverse Design Optimizer (Fixed)
# ==============================================================================

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

assert torch.cuda.is_available(), "런타임 유형을 T4 GPU로 변경하세요."
device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
print(f"[*] Robust Spectral Optimizer running on: {device} ({gpu_name})")

# ==============================================================================
# 1. Numerically Stable Differentiable Spectral Poisson Solver
# ==============================================================================
class DifferentiableSpectralPoisson(nn.Module):
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
# 2. Smooth Differentiable Advection Operator
# ==============================================================================
def differentiable_advection(f, u, v, dx, dy):
    eps = 1e-4
    u_p = 0.5 * (u + torch.sqrt(u**2 + eps**2))
    u_m = 0.5 * (u - torch.sqrt(u**2 + eps**2))
    v_p = 0.5 * (v + torch.sqrt(v**2 + eps**2))
    v_m = 0.5 * (v - torch.sqrt(v**2 + eps**2))
    
    df_dx_b = (f[1:, :] - f[:-1, :]) / dx
    df_dx_f = torch.cat([df_dx_b, df_dx_b[-1:]], dim=0)
    df_dx_b = torch.cat([df_dx_b[:1], df_dx_b], dim=0)
    
    df_dy_b = (f[:, 1:] - f[:, :-1]) / dy
    df_dy_f = torch.cat([df_dy_b, df_dy_b[:, -1:]], dim=1)
    df_dy_b = torch.cat([df_dy_b[:, :1], df_dy_b], dim=1)
    
    return u_p * df_dx_b + u_m * df_dx_f + v_p * df_dy_b + v_m * df_dy_f

def compute_velocity(psi, dx, dy):
    u = torch.zeros_like(psi)
    v = torch.zeros_like(psi)
    u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dy)
    v[1:-1, :] = +(psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
    u[0, :] = 0.0
    v[0, :] = (psi[1, :] - psi[0, :]) / dx
    return u, v

# ==============================================================================
# 3. Parametric Generator with Boundary Singularity Protection
# ==============================================================================
class BlowupAnsatzGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial guess from baseline physics
        self.params = nn.Parameter(torch.tensor([
            1.65,    # log_amp
            1.64,    # log_cx
            -1.52,   # log_cy
            0.28,    # logit_x0
            1.52,    # logit_y0
            0.97     # skewness
        ], dtype=torch.float64, device=device))
        
    def forward(self, X, Y):
        amp  = torch.exp(self.params[0]) * 100.0
        cx   = torch.exp(self.params[1]) * 12.0
        cy   = torch.exp(self.params[2]) * 15.0
        x0   = torch.sigmoid(self.params[3]) * 0.15
        y0   = torch.sigmoid(self.params[4]) * 0.25
        skew = torch.tanh(self.params[5]) * 10.0
        
        r_sq = cx * (X - x0)**2 + cy * (Y - y0)**2 + skew * (X - x0) * (Y - y0)
        
        # Protected wall factor to prevent 0 * ln(0) NaN gradient
        wall_factor = torch.clamp(1.0 - X**2, min=1e-5, max=1.0)
        theta_0 = amp * wall_factor * torch.exp(-torch.clamp(r_sq, min=0.0, max=45.0)) * torch.sin(np.pi * Y)
        w_0 = torch.zeros_like(theta_0)
        return w_0, theta_0

# ==============================================================================
# 4. Adaptive-CFL Differentiable Forward Trajectory
# ==============================================================================
N_opt = 96
poisson_opt = DifferentiableSpectralPoisson(N_opt, N_opt, device=device)
x_opt = torch.linspace(0, 1.0, N_opt, dtype=torch.float64, device=device)
y_opt = torch.linspace(0, 1.0, N_opt, dtype=torch.float64, device=device)
X_opt, Y_opt = torch.meshgrid(x_opt, y_opt, indexing='ij')
dx_opt = 1.0 / (N_opt - 1)
dy_opt = 1.0 / (N_opt - 1)

def forward_trajectory_and_loss(ansatz_model, n_steps=35):
    w, theta = ansatz_model(X_opt, Y_opt)
    
    for step in range(n_steps):
        psi = poisson_opt(w)
        u, v = compute_velocity(psi, dx_opt, dy_opt)
        
        # Adaptive CFL time stepping inside Autograd
        max_vel = torch.clamp(torch.max(torch.abs(u)), min=1e-3)
        max_vel = torch.max(max_vel, torch.max(torch.abs(v)))
        dt = torch.clamp(0.28 * dx_opt / max_vel, max=0.0012)
        
        d_theta_dx = torch.zeros_like(theta)
        d_theta_dx[1:-1, :] = (theta[2:, :] - theta[:-2, :]) / (2.0 * dx_opt)
        d_theta_dx[0, :] = (theta[1, :] - theta[0, :]) / dx_opt
        
        # RK2 Step 1
        dw1 = -differentiable_advection(w, u, v, dx_opt, dy_opt) + d_theta_dx
        dth1 = -differentiable_advection(theta, u, v, dx_opt, dy_opt)
        w_mid = w + 0.5 * dt * dw1
        theta_mid = theta + 0.5 * dt * dth1
        
        # RK2 Step 2
        psi_mid = poisson_opt(w_mid)
        u_mid, v_mid = compute_velocity(psi_mid, dx_opt, dy_opt)
        d_theta_dx_mid = torch.zeros_like(theta_mid)
        d_theta_dx_mid[1:-1, :] = (theta_mid[2:, :] - theta_mid[:-2, :]) / (2.0 * dx_opt)
        d_theta_dx_mid[0, :] = (theta_mid[1, :] - theta_mid[0, :]) / dx_opt
        
        dw2 = -differentiable_advection(w_mid, u_mid, v_mid, dx_opt, dy_opt) + d_theta_dx_mid
        dth2 = -differentiable_advection(theta_mid, u_mid, v_mid, dx_opt, dy_opt)
        
        w = w + dt * dw2
        theta = theta + dt * dth2
        w[0, :] = 0.0
        
    # Stable Lp-norm (p=6) to prevent numerical overflow
    p = 6.0
    w_lp = torch.pow(torch.mean(torch.pow(torch.abs(w) + 1.0, p)), 1.0 / p)
    peak_w_actual = torch.max(torch.abs(w))
    
    loss = -torch.log(w_lp)
    return loss, peak_w_actual.item(), w_lp.item()

# ==============================================================================
# 5. Optimization Loop
# ==============================================================================
ansatz_gen = BlowupAnsatzGenerator()
optimizer = torch.optim.Adam(ansatz_gen.parameters(), lr=0.02)

print("\n" + "="*70)
print("[*] Launching Stable Spectral Inverse Design (Backprop via FFT)...")
print("="*70)

t_opt_start = time.time()
loss_history, peak_history = [], []

for epoch in range(1, 41):
    optimizer.zero_grad()
    loss, peak_val, lp_val = forward_trajectory_and_loss(ansatz_gen, n_steps=32)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(ansatz_gen.parameters(), max_norm=2.0)
    optimizer.step()
    
    loss_history.append(loss.item())
    peak_history.append(peak_val)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/40 | Loss: {loss.item():.4f} | Terminal Lp: {lp_val:7.1f} | Peak ||w||: {peak_val:7.1f}")

print(f"[+] Optimization finished in {time.time() - t_opt_start:.2f}s")
torch.save(ansatz_gen.params.detach(), 'optimal_spectral_ansatz_params.pt')
print("[+] Saved parameters: optimal_spectral_ansatz_params.pt\n")

# ==============================================================================
# 6. High-Resolution Verification at N = 256
# ==============================================================================
print("="*70)
print("[*] Running Verification at N = 256 with Exact Spectral Solver...")
print("="*70)

N_ver = 256
dx_v = 1.0 / (N_ver - 1)
dy_v = 1.0 / (N_ver - 1)
x_v = torch.linspace(0, 1.0, N_ver, dtype=torch.float64, device=device)
y_v = torch.linspace(0, 1.0, N_ver, dtype=torch.float64, device=device)
X_v, Y_v = torch.meshgrid(x_v, y_v, indexing='ij')

poisson_ver = DifferentiableSpectralPoisson(N_ver, N_ver, device=device)

with torch.no_grad():
    w_ver, theta_ver = ansatz_gen(X_v, Y_v)

t_ver = 0.0
t_hist_ver, w_peak_hist_ver = [], []
max_v_steps = 750
t0_v = time.time()

for step in range(max_v_steps):
    pk = torch.max(torch.abs(w_ver)).item()
    t_hist_ver.append(t_ver)
    w_peak_hist_ver.append(pk)
    
    if pk >= 35000.0 or t_ver >= 0.22:
        break
        
    psi_v = poisson_ver(w_ver)
    u_v, v_v = compute_velocity(psi_v, dx_v, dy_v)
    max_vel = max(torch.max(torch.abs(u_v)).item(), torch.max(torch.abs(v_v)).item(), 1e-4)
    dt_v = min(0.30 * min(dx_v, dy_v) / max_vel, 0.0008)
    
    d_theta_dx = torch.zeros_like(theta_ver)
    d_theta_dx[1:-1, :] = (theta_ver[2:, :] - theta_ver[:-2, :]) / (2.0 * dx_v)
    d_theta_dx[0, :] = (theta_ver[1, :] - theta_ver[0, :]) / dx_v
    
    dw1 = -differentiable_advection(w_ver, u_v, v_v, dx_v, dy_v) + d_theta_dx
    dth1 = -differentiable_advection(theta_ver, u_v, v_v, dx_v, dy_v)
    w_mid = w_ver + 0.5 * dt_v * dw1
    theta_mid = theta_ver + 0.5 * dt_v * dth1
    
    psi_mid = poisson_ver(w_mid)
    u_mid, v_mid = compute_velocity(psi_mid, dx_v, dy_v)
    d_theta_dx_mid = torch.zeros_like(theta_mid)
    d_theta_dx_mid[1:-1, :] = (theta_mid[2:, :] - theta_mid[:-2, :]) / (2.0 * dx_v)
    d_theta_dx_mid[0, :] = (theta_mid[1, :] - theta_mid[0, :]) / dx_v
    
    dw2 = -differentiable_advection(w_mid, u_mid, v_mid, dx_v, dy_v) + d_theta_dx_mid
    dth2 = -differentiable_advection(theta_mid, u_mid, v_mid, dx_v, dy_v)
    
    w_ver = w_ver + dt_v * dw2
    theta_ver = theta_ver + dt_v * dth2
    w_ver[0, :] = 0.0
    t_ver += dt_v

print(f"[+] Verification completed in {time.time() - t0_v:.2f}s")
print(f"    Initial: {w_peak_hist_ver[0]:.1f} | Max Peak: {max(w_peak_hist_ver):.1f} | End Time: {t_hist_ver[-1]:.4f}s")

# ==============================================================================
# 7. Visualization
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

# Subplot 1: Optimization Curve
ax1.plot(range(1, 41), peak_history, 'o-', color='#1f77b4', lw=2.0, markersize=5)
ax1.set_title('(a) Spectral Inverse Design Convergence', fontsize=11, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=10)
ax1.set_ylabel(r'Terminal Peak Vorticity $\|\omega\|_\infty$', fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, which="both", alpha=0.3)

# Subplot 2: Forward Trajectory at N = 256
ax2.plot(t_hist_ver, w_peak_hist_ver, '-', color='#d62728', lw=2.5, label='N = 256 (Spectral DST)')
ax2.set_title('(b) Vorticity Growth Trajectory (N = 256)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time t (seconds)', fontsize=10)
ax2.set_ylabel(r'Vorticity $\|\omega(t)\|_\infty$', fontsize=10)
ax2.set_yscale('log')
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc='upper left', fontsize=9.0)

# Subplot 3: Vorticity Field
im = ax3.imshow(w_ver.cpu().numpy().T, origin='lower', extent=[0, 1, 0, 1], cmap='inferno', aspect='auto')
ax3.set_title(r'(c) Vorticity Field Core at $t_{\mathrm{final}}$', fontsize=11, fontweight='bold')
ax3.set_xlabel('Distance from Wall x', fontsize=10)
ax3.set_ylabel('Vertical Coordinate y', fontsize=10)
fig.colorbar(im, ax=ax3)

out_png = 'spectral_optimization_robust_result.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"[+] Output figure exported: {out_png}")
plt.show()