# c depth

import os
import time
import math
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

DEVICE_ID = 0
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)
    DEVICE = torch.device(f"cuda:{DEVICE_ID}")
else:
    DEVICE = torch.device("cpu")

PI = 3.1415926536
GAUSSIAN_PEAK_INTENSITY = 0.4000222589 / 1000000.0
SCALE_FACTOR = 1.0e3
MIN_HALF_KERNEL_SIZE = 12
MIN_NUM_BLOCK = 8192
MAX_SPLIT_K = 1024

C_WATER = 1482.0
C_SKULL_P = 2700.0
C_SKULL_S = 1320.0  
RHO_WATER = 1000.0
RHO_SKULL = 1850.0

ALPHA_SKULL_P = 50.0
ALPHA_SKULL_S = 85.0

fs = 10.0e6
delta_time = 1.0 / fs

DATA_PATH = "./data/signal_ws_fig4_c.mat"
LOCATION_PATH = "./data/sensor_location.mat"
SKULL_PATH = "./data/skull/skull_mask.mat"
DATA_VAR = "real_signal"
LOCATION_VAR = "sensor_location"

res = 0.5e-3
x_range, y_range, z_range = [-40.0e-3, 40.0e-3], [-35.0e-3, 45.0e-3], [-45.0e-3, -39.0e-3]
x_start, x_end = x_range[0], x_range[1]
y_start, y_end = y_range[0], y_range[1]
z_start, z_end = z_range[0], z_range[1]

num_x = int(round((x_end - x_start) / res)) + 1
num_y = int(round((y_end - y_start) / res)) + 1
num_z = int(round((z_end - z_start) / res)) + 1
num_voxels = num_x * num_y * num_z
sampled_indices = torch.arange(10, 51200, 10)[:5120]

try:
    detector_locations = torch.from_numpy(sio.loadmat(LOCATION_PATH)[LOCATION_VAR][sampled_indices]).to(DEVICE).contiguous()
    simulation_data = torch.from_numpy(sio.loadmat(DATA_PATH)[DATA_VAR][sampled_indices]).to(DEVICE).contiguous() * SCALE_FACTOR
except FileNotFoundError:
    detector_locations = torch.rand((5120, 3), device=DEVICE) * 0.05
    simulation_data = torch.rand((5120, 1000), device=DEVICE)

sorted_indices = torch.argsort(detector_locations[:, 2], descending=True)
detector_locations = detector_locations[sorted_indices].contiguous()
simulation_data = simulation_data[sorted_indices].contiguous()
num_detectors, num_times = simulation_data.shape

t_geom = time.time()

try:
    skull_mask = sio.loadmat(SKULL_PATH)['skull_mask'] 
except FileNotFoundError:
    skull_mask = np.ones((100, 100, 100), dtype=bool)

skull_mask_tsr = torch.tensor(skull_mask, dtype=torch.bool, device=DEVICE)
mask_shape = skull_mask_tsr.shape
CX_mask, CY_mask, CZ_mask = mask_shape[0] // 2, mask_shape[1] // 2, mask_shape[2] // 2

z_start_idx = torch.argmax(skull_mask_tsr.to(torch.uint8), dim=2)
z_end_idx = mask_shape[2] - 1 - torch.argmax(torch.flip(skull_mask_tsr.to(torch.uint8), dims=[2]), dim=2)
valid_xy_mask = skull_mask_tsr.any(dim=2)
z_start_idx[~valid_xy_mask] = CZ_mask
z_end_idx[~valid_xy_mask] = CZ_mask

sigma_smooth = 1.0
z_start_smooth = gaussian_filter(z_start_idx.cpu().float().numpy(), sigma=sigma_smooth)
z_end_smooth   = gaussian_filter(z_end_idx.cpu().float().numpy(), sigma=sigma_smooth)

z_in_phys = (z_start_smooth - CZ_mask) * res
z_out_phys = (z_end_smooth - CZ_mask) * res

gx_in_np, gy_in_np = np.gradient(z_in_phys, res, res)
gx_out_np, gy_out_np = np.gradient(z_out_phys, res, res)

z_in_tsr_full = torch.from_numpy(z_in_phys).contiguous().to(DEVICE)
z_out_tsr_full = torch.from_numpy(z_out_phys).contiguous().to(DEVICE)
gx_in_tsr_full = torch.from_numpy(gx_in_np).contiguous().to(DEVICE)
gy_in_tsr_full = torch.from_numpy(gy_in_np).contiguous().to(DEVICE)
gx_out_tsr_full = torch.from_numpy(gx_out_np).contiguous().to(DEVICE)
gy_out_tsr_full = torch.from_numpy(gy_out_np).contiguous().to(DEVICE)

Grid_X, Grid_Y, Grid_Z = np.meshgrid(
    np.arange(x_start, x_end + res/2, res),
    np.arange(y_start, y_end + res/2, res),
    np.arange(z_start, z_end + res/2, res), 
    indexing='ij'
)
sources_tsr = torch.tensor(np.stack([Grid_X.ravel(), Grid_Y.ravel(), Grid_Z.ravel()], axis=1), dtype=torch.float32, device=DEVICE).contiguous()
print(f"Geometry time: {time.time()-t_geom:.2f} s")

@triton.jit
def get_interp_weights(p_x, p_y, inv_res, cx_mask, cy_mask, max_x, max_y):
    fx = p_x * inv_res + cx_mask
    fy = p_y * inv_res + cy_mask
    x0 = tl.cast(tl.floor(fx), tl.int32)
    y0 = tl.cast(tl.floor(fy), tl.int32)
    x0 = tl.maximum(0, tl.minimum(x0, max_x))
    y0 = tl.maximum(0, tl.minimum(y0, max_y))
    x1 = tl.maximum(0, tl.minimum(x0 + 1, max_x))
    y1 = tl.maximum(0, tl.minimum(y0 + 1, max_y))
    wx = fx - x0
    wy = fy - y0
    return x0, y0, x1, y1, wx, wy

@triton.jit
def interp2d(ptr, x0, y0, x1, y1, wx, wy, stride_x, stride_y, mask):
    i00 = x0 * stride_x + y0 * stride_y
    i10 = x1 * stride_x + y0 * stride_y
    i01 = x0 * stride_x + y1 * stride_y
    i11 = x1 * stride_x + y1 * stride_y
    v00 = tl.load(ptr + i00, mask=mask, other=0.0)
    v10 = tl.load(ptr + i10, mask=mask, other=0.0)
    v01 = tl.load(ptr + i01, mask=mask, other=0.0)
    v11 = tl.load(ptr + i11, mask=mask, other=0.0)
    v0 = v00 * (1.0 - wx) + v10 * wx
    v1 = v01 * (1.0 - wx) + v11 * wx
    return v0 * (1.0 - wy) + v1 * wy

@triton.jit
def compute_zoeppritz_transmission(cos_w, c_w, c_p, c_s, rho_w, rho_s, is_p_wave: tl.constexpr):
    sin_w = tl.sqrt(tl.maximum(1.0 - cos_w*cos_w, 0.0))
    p_param = sin_w / c_w
    
    sin_p = p_param * c_p
    sin_s = p_param * c_s
    
    Z_w = rho_w * c_w / tl.maximum(cos_w, 1e-8)
    
    sin_s_clamp = tl.minimum(sin_s, 1.0)
    cos_s = tl.sqrt(tl.maximum(1.0 - sin_s_clamp*sin_s_clamp, 1e-8))
    Z_s = rho_s * c_s / cos_s
    
    is_p_real = sin_p <= 1.0
    cos_p_sq = tl.abs(1.0 - sin_p*sin_p)
    cos_p_mag = tl.sqrt(tl.maximum(cos_p_sq, 1e-8))
    R_p = rho_s * c_p / cos_p_mag
    
    Z_p_re = tl.where(is_p_real, R_p, 0.0)
    Z_p_im = tl.where(is_p_real, 0.0, R_p)
    
    cos_2s = 1.0 - 2.0 * sin_s_clamp * sin_s_clamp
    sin2_2s = 4.0 * sin_s_clamp * sin_s_clamp * cos_s * cos_s
    
    D_re = Z_w + Z_p_re * (cos_2s * cos_2s) + Z_s * sin2_2s
    D_im = Z_p_im * (cos_2s * cos_2s)
    D_sq = D_re * D_re + D_im * D_im + 1e-12
    
    if is_p_wave:
        T_total = tl.where(is_p_real, (4.0 * Z_w * R_p * cos_2s * cos_2s) / D_sq, 0.0)
    else:
        is_s_real = sin_s <= 1.0
        T_total = tl.where(is_s_real, (4.0 * Z_w * Z_s * sin2_2s) / D_sq, 0.0)
        
    return T_total

@triton.jit
def two_snap_fermat_ray_kernel(
    src_ptr, sen_ptr, 
    z_in_ptr, z_out_ptr, 
    gx_in_ptr, gy_in_ptr, gx_out_ptr, gy_out_ptr,
    T_out_ptr, W_out_ptr,
    n_sources: tl.constexpr, n_sensors: tl.constexpr,
    c_w: tl.constexpr, c_p: tl.constexpr, c_s: tl.constexpr, 
    rho_w: tl.constexpr, rho_s: tl.constexpr, 
    alpha_p: tl.constexpr, alpha_s: tl.constexpr,
    is_p_wave: tl.constexpr,
    inv_res: tl.constexpr, cx_mask: tl.constexpr, cy_mask: tl.constexpr,
    max_x: tl.constexpr, max_y: tl.constexpr, stride_x: tl.constexpr, stride_y: tl.constexpr,
    BLOCK_SRC: tl.constexpr, BLOCK_SEN: tl.constexpr
):
    pid_src = tl.program_id(0)
    pid_sen = tl.program_id(1)
    
    offs_src = pid_src * BLOCK_SRC + tl.arange(0, BLOCK_SRC)
    offs_sen = pid_sen * BLOCK_SEN + tl.arange(0, BLOCK_SEN)
    mask_src = offs_src < n_sources
    mask_sen = offs_sen < n_sensors
    valid_ray = mask_src[:, None] & mask_sen[None, :]
    
    sx = tl.load(src_ptr + offs_src * 3 + 0, mask=mask_src, other=0.0)[:, None]
    sy = tl.load(src_ptr + offs_src * 3 + 1, mask=mask_src, other=0.0)[:, None]
    sz = tl.load(src_ptr + offs_src * 3 + 2, mask=mask_src, other=0.0)[:, None]

    dx = tl.load(sen_ptr + offs_sen * 3 + 0, mask=mask_sen, other=0.0)[None, :]
    dy = tl.load(sen_ptr + offs_sen * 3 + 1, mask=mask_sen, other=0.0)[None, :]
    dz = tl.load(sen_ptr + offs_sen * 3 + 2, mask=mask_sen, other=0.0)[None, :]
    
    c_skull = c_p if is_p_wave else c_s
    alpha_skull = alpha_p if is_p_wave else alpha_s
    
    x0_s, y0_s, x1_s, y1_s, wx_s, wy_s = get_interp_weights(sx, sy, inv_res, cx_mask, cy_mask, max_x, max_y)
    z_in_top = interp2d(z_in_ptr, x0_s, y0_s, x1_s, y1_s, wx_s, wy_s, stride_x, stride_y, mask_src[:, None])
    z_out_top = interp2d(z_out_ptr, x0_s, y0_s, x1_s, y1_s, wx_s, wy_s, stride_x, stride_y, mask_src[:, None])
    
    t_guess_in = (z_in_top - sz) / (dz - sz + 1e-8)
    t_guess_out = (z_out_top - sz) / (dz - sz + 1e-8)
    
    px_guess_in = sx + t_guess_in * (dx - sx)
    py_guess_in = sy + t_guess_in * (dy - sy)
    px_guess_out = sx + t_guess_out * (dx - sx)
    py_guess_out = sy + t_guess_out * (dy - sy)

    x0_i, y0_i, x1_i, y1_i, wx_i, wy_i = get_interp_weights(px_guess_in, py_guess_in, inv_res, cx_mask, cy_mask, max_x, max_y)
    z_in_snap = interp2d(z_in_ptr, x0_i, y0_i, x1_i, y1_i, wx_i, wy_i, stride_x, stride_y, valid_ray)
    gxi_snap  = interp2d(gx_in_ptr, x0_i, y0_i, x1_i, y1_i, wx_i, wy_i, stride_x, stride_y, valid_ray)
    gyi_snap  = interp2d(gy_in_ptr, x0_i, y0_i, x1_i, y1_i, wx_i, wy_i, stride_x, stride_y, valid_ray)
    
    x0_o, y0_o, x1_o, y1_o, wx_o, wy_o = get_interp_weights(px_guess_out, py_guess_out, inv_res, cx_mask, cy_mask, max_x, max_y)
    z_out_snap = interp2d(z_out_ptr, x0_o, y0_o, x1_o, y1_o, wx_o, wy_o, stride_x, stride_y, valid_ray)
    gxo_snap   = interp2d(gx_out_ptr, x0_o, y0_o, x1_o, y1_o, wx_o, wy_o, stride_x, stride_y, valid_ray)
    gyo_snap   = interp2d(gy_out_ptr, x0_o, y0_o, x1_o, y1_o, wx_o, wy_o, stride_x, stride_y, valid_ray)
    
    px_in, py_in = px_guess_in, py_guess_in
    px_out, py_out = px_guess_out, py_guess_out
    lr = 1.0 
    
    for _ in tl.static_range(10):
        cur_z_in = z_in_snap + gxi_snap * (px_in - px_guess_in) + gyi_snap * (py_in - py_guess_in)
        cur_z_out = z_out_snap + gxo_snap * (px_out - px_guess_out) + gyo_snap * (py_out - py_guess_out)

        d1 = tl.sqrt((px_in-sx)*(px_in-sx) + (py_in-sy)*(py_in-sy) + (cur_z_in-sz)*(cur_z_in-sz) + 1e-12)
        d2 = tl.sqrt((px_out-px_in)*(px_out-px_in) + (py_out-py_in)*(py_out-py_in) + (cur_z_out-cur_z_in)*(cur_z_out-cur_z_in) + 1e-12)
        d3 = tl.sqrt((dx-px_out)*(dx-px_out) + (dy-py_out)*(dy-py_out) + (dz-cur_z_out)*(dz-cur_z_out) + 1e-12)
        
        grad_xin = ((px_in - sx) + (cur_z_in - sz)*gxi_snap)/(c_w * d1) + ((px_in - px_out) + (cur_z_in - cur_z_out)*gxi_snap)/(c_skull * d2)
        grad_yin = ((py_in - sy) + (cur_z_in - sz)*gyi_snap)/(c_w * d1) + ((py_in - py_out) + (cur_z_in - cur_z_out)*gyi_snap)/(c_skull * d2)
        grad_xout = ((px_out - px_in) + (cur_z_out - cur_z_in)*gxo_snap)/(c_skull * d2) + ((px_out - dx) + (cur_z_out - dz)*gxo_snap)/(c_w * d3)
        grad_yout = ((py_out - py_in) + (cur_z_out - cur_z_in)*gyo_snap)/(c_skull * d2) + ((py_out - dy) + (cur_z_out - dz)*gyo_snap)/(c_w * d3)

        px_in -= lr * grad_xin
        py_in -= lr * grad_yin
        px_out -= lr * grad_xout
        py_out -= lr * grad_yout

    px_guess_in, py_guess_in = px_in, py_in
    px_guess_out, py_guess_out = px_out, py_out

    x0_i, y0_i, x1_i, y1_i, wx_i, wy_i = get_interp_weights(px_guess_in, py_guess_in, inv_res, cx_mask, cy_mask, max_x, max_y)
    z_in_snap = interp2d(z_in_ptr, x0_i, y0_i, x1_i, y1_i, wx_i, wy_i, stride_x, stride_y, valid_ray)
    gxi_snap  = interp2d(gx_in_ptr, x0_i, y0_i, x1_i, y1_i, wx_i, wy_i, stride_x, stride_y, valid_ray)
    gyi_snap  = interp2d(gy_in_ptr, x0_i, y0_i, x1_i, y1_i, wx_i, wy_i, stride_x, stride_y, valid_ray)
    
    x0_o, y0_o, x1_o, y1_o, wx_o, wy_o = get_interp_weights(px_guess_out, py_guess_out, inv_res, cx_mask, cy_mask, max_x, max_y)
    z_out_snap = interp2d(z_out_ptr, x0_o, y0_o, x1_o, y1_o, wx_o, wy_o, stride_x, stride_y, valid_ray)
    gxo_snap   = interp2d(gx_out_ptr, x0_o, y0_o, x1_o, y1_o, wx_o, wy_o, stride_x, stride_y, valid_ray)
    gyo_snap   = interp2d(gy_out_ptr, x0_o, y0_o, x1_o, y1_o, wx_o, wy_o, stride_x, stride_y, valid_ray)

    for _ in tl.static_range(0):
        cur_z_in = z_in_snap + gxi_snap * (px_in - px_guess_in) + gyi_snap * (py_in - py_guess_in)
        cur_z_out = z_out_snap + gxo_snap * (px_out - px_guess_out) + gyo_snap * (py_out - py_guess_out)

        d1 = tl.sqrt((px_in-sx)*(px_in-sx) + (py_in-sy)*(py_in-sy) + (cur_z_in-sz)*(cur_z_in-sz) + 1e-12)
        d2 = tl.sqrt((px_out-px_in)*(px_out-px_in) + (py_out-py_in)*(py_out-py_in) + (cur_z_out-cur_z_in)*(cur_z_out-cur_z_in) + 1e-12)
        d3 = tl.sqrt((dx-px_out)*(dx-px_out) + (dy-py_out)*(dy-py_out) + (dz-cur_z_out)*(dz-cur_z_out) + 1e-12)
        
        grad_xin = ((px_in - sx) + (cur_z_in - sz)*gxi_snap)/(c_w * d1) + ((px_in - px_out) + (cur_z_in - cur_z_out)*gxi_snap)/(c_skull * d2)
        grad_yin = ((py_in - sy) + (cur_z_in - sz)*gyi_snap)/(c_w * d1) + ((py_in - py_out) + (cur_z_in - cur_z_out)*gyi_snap)/(c_skull * d2)
        grad_xout = ((px_out - px_in) + (cur_z_out - cur_z_in)*gxo_snap)/(c_skull * d2) + ((px_out - dx) + (cur_z_out - dz)*gxo_snap)/(c_w * d3)
        grad_yout = ((py_out - py_in) + (cur_z_out - cur_z_in)*gyo_snap)/(c_skull * d2) + ((py_out - dy) + (cur_z_out - dz)*gyo_snap)/(c_w * d3)

        px_in -= lr * grad_xin
        py_in -= lr * grad_yin
        px_out -= lr * grad_xout
        py_out -= lr * grad_yout
        
    cur_z_in = z_in_snap + gxi_snap * (px_in - px_guess_in) + gyi_snap * (py_in - py_guess_in)
    cur_z_out = z_out_snap + gxo_snap * (px_out - px_guess_out) + gyo_snap * (py_out - py_guess_out)
    
    d1 = tl.sqrt((px_in-sx)*(px_in-sx) + (py_in-sy)*(py_in-sy) + (cur_z_in-sz)*(cur_z_in-sz) + 1e-12)
    d2 = tl.sqrt((px_out-px_in)*(px_out-px_in) + (py_out-py_in)*(py_out-py_in) + (cur_z_out-cur_z_in)*(cur_z_out-cur_z_in) + 1e-12)
    d3 = tl.sqrt((dx-px_out)*(dx-px_out) + (dy-py_out)*(dy-py_out) + (dz-cur_z_out)*(dz-cur_z_out) + 1e-12)
    total_time = (d1/c_w) + (d2/c_skull) + (d3/c_w)
    
    norm_len_in = tl.sqrt(gxi_snap*gxi_snap + gyi_snap*gyi_snap + 1.0 + 1e-12)
    dot_prod_in = (-(px_in - sx)*gxi_snap - (py_in - sy)*gyi_snap + (cur_z_in - sz)) / (d1 * norm_len_in)
    cos_in = tl.abs(dot_prod_in)
    
    T_total_coeff = compute_zoeppritz_transmission(
        cos_in, c_w, c_p, c_s, rho_w, rho_s, is_p_wave
    )
    
    skull_attenuation = tl.exp(-alpha_skull * d2)
    
    final_weight = (T_total_coeff * skull_attenuation) / (d1 + d2 + d3 + 1e-12)
    out_idx = offs_sen[None, :] * n_sources + offs_src[:, None]
    
    tl.store(T_out_ptr + out_idx, total_time, mask=valid_ray)
    tl.store(W_out_ptr + out_idx, final_weight.to(tl.float16), mask=valid_ray)

def compute_physics_matrices_snap(sources, sensors, mode='P'):
    N_src, N_sen = len(sources), len(sensors)
    is_p_wave = (mode == 'P')
    T_out = torch.empty((N_sen, N_src), dtype=torch.float32, device=DEVICE)
    W_out = torch.empty((N_sen, N_src), dtype=torch.float16, device=DEVICE)
    
    grid = (triton.cdiv(N_src, 64), triton.cdiv(N_sen, 32))
    two_snap_fermat_ray_kernel[grid](
        sources, sensors, 
        z_in_tsr_full, z_out_tsr_full, 
        gx_in_tsr_full, gy_in_tsr_full, gx_out_tsr_full, gy_out_tsr_full,
        T_out, W_out,
        n_sources=N_src, n_sensors=N_sen,
        c_w=C_WATER, c_p=C_SKULL_P, c_s=C_SKULL_S, 
        rho_w=RHO_WATER, rho_s=RHO_SKULL, 
        alpha_p=ALPHA_SKULL_P, alpha_s=ALPHA_SKULL_S,
        is_p_wave=is_p_wave,
        inv_res=1.0/res, cx_mask=CX_mask, cy_mask=CY_mask,
        max_x=mask_shape[0]-1, max_y=mask_shape[1]-1, 
        stride_x=mask_shape[1], stride_y=1,
        BLOCK_SRC=64, BLOCK_SEN=32
    )
    return T_out, W_out

t0 = time.time()
Tp_map, Wp_map = compute_physics_matrices_snap(sources_tsr, detector_locations, mode='P')
Ts_map, Ws_map = compute_physics_matrices_snap(sources_tsr, detector_locations, mode='S')
print(f"Physics time: {(time.time()-t0):.2f} s")

def generate_adaptive_conv_kernel(res_val, vs, fs):
    time_interval_length_half = int(3.0 * res_val / vs * fs + 1)
    adaptive_ratio = int(MIN_HALF_KERNEL_SIZE / time_interval_length_half + 1)
    time_interval_length_half = time_interval_length_half * adaptive_ratio
    fs_upsampled = fs * adaptive_ratio
    const_factor = 2.0 * PI * GAUSSIAN_PEAK_INTENSITY * vs / res_val
    conv_input = vs / fs_upsampled * torch.arange(
        time_interval_length_half, -time_interval_length_half - 1, -1, device=DEVICE, dtype=torch.float32)
    conv_kernel = ((const_factor * conv_input * torch.exp(-(conv_input**2) / (2.0 * res_val**2)))
                   .unsqueeze(0).unsqueeze(0))
    return conv_kernel, time_interval_length_half, adaptive_ratio, fs_upsampled

conv_kernel, time_interval_length_half, adaptive_ratio, fs_upsampled = generate_adaptive_conv_kernel(1*res, C_WATER, fs)
num_times_upsampled = num_times * adaptive_ratio

split_k = min(MAX_SPLIT_K, 2 ** int(math.log2(max(1, (MIN_NUM_BLOCK + num_detectors - 1) // num_detectors)) + 0.5))

@triton.jit
def forward_kernel_splitk_ps(
    x_ptr, Tp_ptr, Ts_ptr, Wp_ptr, Ws_ptr, partial_output_ptr,
    num_voxels: tl.constexpr, num_detectors: tl.constexpr, num_times_upsampled: tl.constexpr,
    fs_upsampled: tl.constexpr,
    stride_partial_k: tl.constexpr, stride_partial_d: tl.constexpr,
    stride_map_d: tl.constexpr, stride_map_v: tl.constexpr,
    split_k: tl.constexpr, BLOCK_VOXEL: tl.constexpr, BLOCK_DET: tl.constexpr
):
    det_block_idx = tl.program_id(0)
    split_k_idx = tl.program_id(1)
    det_start = det_block_idx * BLOCK_DET
    out_start = partial_output_ptr + split_k_idx * stride_partial_k
    voxels_per_split = tl.cdiv(num_voxels, split_k)
    v_start_base = split_k_idx * voxels_per_split
    v_end = tl.minimum(v_start_base + voxels_per_split, num_voxels)

    for v_start in tl.range(v_start_base, v_end, BLOCK_VOXEL):
        v_offsets = v_start + tl.arange(0, BLOCK_VOXEL)
        v_mask = v_offsets < v_end
        x_val = tl.load(x_ptr + v_offsets, mask=v_mask, other=0.0)

        for d in tl.static_range(0, BLOCK_DET):
            det_id = det_start + d
            d_mask = det_id < num_detectors
            map_idx = det_id * stride_map_d + v_offsets * stride_map_v
            tp = tl.load(Tp_ptr + map_idx, mask=v_mask & d_mask, other=-1.0)
            wp = tl.load(Wp_ptr + map_idx, mask=v_mask & d_mask, other=0.0).to(tl.float32)
            ts = tl.load(Ts_ptr + map_idx, mask=v_mask & d_mask, other=-1.0)
            ws = tl.load(Ws_ptr + map_idx, mask=v_mask & d_mask, other=0.0).to(tl.float32)

            idx_p = (tp * fs_upsampled + 0.5).to(tl.int32)
            idx_s = (ts * fs_upsampled + 0.5).to(tl.int32)
            mask_p = (idx_p >= 0) & (idx_p < num_times_upsampled) & v_mask & d_mask
            mask_s = (idx_s >= 0) & (idx_s < num_times_upsampled) & v_mask & d_mask

            out_ptr = out_start + det_id * stride_partial_d
            tl.atomic_add(out_ptr + idx_p, x_val * wp, mask=mask_p, sem="relaxed")
            tl.atomic_add(out_ptr + idx_s, x_val * ws, mask=mask_s, sem="relaxed")

@triton.jit
def transpose_kernel_ps(
    dx_conv_ptr, Tp_ptr, Ts_ptr, Wp_ptr, Ws_ptr, output_ptr,
    num_voxels: tl.constexpr, num_detectors: tl.constexpr, num_times_upsampled: tl.constexpr,
    fs_upsampled: tl.constexpr,
    stride_dx_d: tl.constexpr, stride_map_d: tl.constexpr, stride_map_v: tl.constexpr,
    BLOCK_VOXEL: tl.constexpr
):
    voxel_block_idx = tl.program_id(0)
    v_start = voxel_block_idx * BLOCK_VOXEL
    v_offsets = v_start + tl.arange(0, BLOCK_VOXEL)
    v_mask = v_offsets < num_voxels
    accum = tl.zeros((BLOCK_VOXEL,), dtype=tl.float32)

    for det_idx in tl.range(0, num_detectors):
        map_idx = det_idx * stride_map_d + v_offsets * stride_map_v
        tp = tl.load(Tp_ptr + map_idx, mask=v_mask, other=-1.0)
        wp = tl.load(Wp_ptr + map_idx, mask=v_mask, other=0.0).to(tl.float32)
        ts = tl.load(Ts_ptr + map_idx, mask=v_mask, other=-1.0)
        ws = tl.load(Ws_ptr + map_idx, mask=v_mask, other=0.0).to(tl.float32)

        idx_p = (tp * fs_upsampled + 0.5).to(tl.int32)
        idx_s = (ts * fs_upsampled + 0.5).to(tl.int32)
        mask_p = (idx_p >= 0) & (idx_p < num_times_upsampled) & v_mask
        mask_s = (idx_s >= 0) & (idx_s < num_times_upsampled) & v_mask

        row_ptr = dx_conv_ptr + det_idx * stride_dx_d
        g_p = tl.load(row_ptr + idx_p, mask=mask_p, other=0.0).to(tl.float32)
        g_s = tl.load(row_ptr + idx_s, mask=mask_s, other=0.0).to(tl.float32)

        accum += tl.where(mask_p, g_p * wp, 0.0) + tl.where(mask_s, g_s * ws, 0.0)

    tl.store(output_ptr + v_offsets, accum, mask=v_mask)

def forward_operator_triton(x: torch.Tensor) -> torch.Tensor:
    partial_output = torch.zeros((split_k, num_detectors, num_times_upsampled), device=DEVICE, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(num_detectors, META["BLOCK_DET"]), split_k)
    forward_kernel_splitk_ps[grid](
        x, Tp_map, Ts_map, Wp_map, Ws_map, partial_output,
        num_voxels=num_voxels, num_detectors=num_detectors, num_times_upsampled=num_times_upsampled,
        fs_upsampled=fs_upsampled,
        stride_partial_k=num_detectors * num_times_upsampled, stride_partial_d=num_times_upsampled,
        stride_map_d=Tp_map.stride(0), stride_map_v=Tp_map.stride(1),
        split_k=split_k, BLOCK_VOXEL=512, BLOCK_DET=8
    )
    dy_conv_transpose = F.conv_transpose1d(
        partial_output.sum(dim=0).unsqueeze(1), conv_kernel, padding=time_interval_length_half).squeeze(1)
    return dy_conv_transpose[:, ::adaptive_ratio].contiguous().flatten()

def transpose_operator_triton(x: torch.Tensor) -> torch.Tensor:
    dx_upsampling = torch.zeros((num_detectors, num_times_upsampled), device=DEVICE, dtype=torch.float32)
    dx_upsampling[:, ::adaptive_ratio] = x.reshape(num_detectors, num_times)
    dx_conv_batch = F.conv1d(dx_upsampling.unsqueeze(1), conv_kernel, padding=time_interval_length_half).squeeze(1).contiguous()
    y = torch.empty(num_voxels, device=DEVICE, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(num_voxels, META["BLOCK_VOXEL"]),)
    transpose_kernel_ps[grid](
        dx_conv_batch, Tp_map, Ts_map, Wp_map, Ws_map, y,
        num_voxels=num_voxels, num_detectors=num_detectors, num_times_upsampled=num_times_upsampled,
        fs_upsampled=fs_upsampled,
        stride_dx_d=num_times_upsampled, stride_map_d=Tp_map.stride(0), stride_map_v=Tp_map.stride(1),
        BLOCK_VOXEL=512
    )
    return y

class PS_Operator_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return forward_operator_triton(x)
    @staticmethod
    def backward(ctx, grad_output): return transpose_operator_triton(grad_output)

def A_operator_triton(x): return PS_Operator_Function.apply(x)

def vessel_continuity_loss(x_3d, alpha=0.0, beta=2.0, eps=1e-8):
    grad_x = torch.zeros_like(x_3d)
    grad_y = torch.zeros_like(x_3d)
    grad_z = torch.zeros_like(x_3d)
    grad_x[1:] = x_3d[1:] - x_3d[:-1]
    grad_y[:, 1:] = x_3d[:, 1:] - x_3d[:, :-1]
    grad_z[:, :, 1:] = x_3d[:, :, 1:] - x_3d[:, :, :-1]
    tv_norm = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + grad_z.pow(2) + eps).sum()
    
    hxx, hyy, hzz = torch.zeros_like(x_3d), torch.zeros_like(x_3d), torch.zeros_like(x_3d)
    hxx[1:] = grad_x[1:] - grad_x[:-1]
    hyy[:, 1:] = grad_y[:, 1:] - grad_y[:, :-1]
    hzz[:, :, 1:] = grad_z[:, :, 1:] - grad_z[:, :, :-1]
    hxy, hxz, hyz = torch.zeros_like(x_3d), torch.zeros_like(x_3d), torch.zeros_like(x_3d)
    hxy[:, 1:] = grad_x[:, 1:] - grad_x[:, :-1]
    hxz[:, :, 1:] = grad_x[:, :, 1:] - grad_x[:, :, :-1]
    hyz[:, :, 1:] = grad_y[:, :, 1:] - grad_y[:, :, :-1]
    hessian_norm = torch.sqrt(hxx.pow(2) + hyy.pow(2) + hzz.pow(2) + 2*(hxy.pow(2)+hxz.pow(2)+hyz.pow(2)) + eps).sum()
    
    l1_norm = torch.abs(x_3d).sum()
    return 1.5*hessian_norm + beta * tv_norm + alpha * l1_norm

def nonnegative(z, eps=1e-8): return (z + eps) * (z + eps)

vessel_continuity_loss = torch.compile(vessel_continuity_loss)
nonnegative = torch.compile(nonnegative)

def gradient_descent_reconstruction_ps(observed_data, max_iterations=100, learning_rate=5e-1, lambda_reg=5e-1):
    z = torch.zeros(num_voxels, device=DEVICE, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-4)
    losses = []

    for iteration in range(max_iterations):
        optimizer.zero_grad()
        x = nonnegative(z)
        predicted_data = A_operator_triton(x)
        data_fidelity = F.mse_loss(predicted_data, observed_data)
        regularization = lambda_reg * vessel_continuity_loss(x.view(num_x, num_y, num_z))
        loss = data_fidelity + regularization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # if iteration % 5 == 0 or iteration == max_iterations - 1:
        #     losses.append(loss.item())
        #     print(f"[Iter {iteration:03d}] Loss = {loss.item():.4e}")

    return nonnegative(z).detach(), losses

b = simulation_data.flatten().contiguous()
torch.cuda.reset_peak_memory_stats()
t_start = torch.cuda.Event(enable_timing=True)
t_end = torch.cuda.Event(enable_timing=True)
t_start.record()

x_reconstructed, losses = gradient_descent_reconstruction_ps(b, max_iterations=15, learning_rate=35e-0, lambda_reg=1.0e-1)

t_end.record()
t_end.synchronize()


x_3d = x_reconstructed.reshape(num_x, num_y, num_z).detach().cpu().numpy()

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
mip_recon = np.max(x_3d, axis=2)
plt.imshow(mip_recon.T, cmap='gray', aspect='equal', origin='lower', extent=[x_start * 1e3, x_end * 1e3, y_start * 1e3, y_end * 1e3])
plt.title("GPAIR PS-Wave (Two-Snap, Z-MIP)")
plt.colorbar()

plt.subplot(1, 2, 2)
center_slice = x_3d[:, :, num_z // 2]
plt.imshow(center_slice.T, cmap='gray', aspect='equal', origin='lower', extent=[x_start * 1e3, x_end * 1e3, y_start * 1e3, y_end * 1e3])
plt.title(f"Center Slice (z={z_start*1e3 + (num_z//2)*res*1e3:.1f} mm)")
plt.colorbar()


# c depth
t_start.record()

x_reconstructed, losses = gradient_descent_reconstruction_ps(b, max_iterations=15, learning_rate=35e-0, lambda_reg=1.0e-1)

t_end.record()
t_end.synchronize()
print(f"Reconstruction time: {t_start.elapsed_time(t_end)/1000:.3f} s")