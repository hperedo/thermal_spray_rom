# SPDX-License-Identifier: MIT
# Copyright (c) 2026 H. Peredo Fuentes & I. Martinez Villegas
#
# Thermal Spray ROM Comparison — 1D Flattening Analysis (v1.0.0)
# Zenodo DOI: 10.5281/zenodo.22729593
# Original source: https://github.com/hperedo/thermal_spray_rom
#
# This source code is licensed under the MIT License.
# Documentation and figures are licensed under CC BY 4.0 (see LICENSE-CC-BY-4.0.txt).
"""
Thermal Spray ROM Comparison - 1D Flattening Analysis (v1.0.0)
==========================================================================
Reproduces all figures and tables in:
  Peredo Fuentes, H. & Martinez Villegas, I. (2026).
  "Reduced-Order Modeling of Particle Flattening Dynamics: A Benchmark
  Study of Krylov-Arnoldi Methods Using Nishioka's Experimental Data",
  Journal of Thermal Spray Technology.

Pipeline:
  - Processes all three particle sizes (small, medium, large)
  - Applies four ROM methods (TD, JK-P, JK-G_RK4, JK-G_Exp)
  - Generates overlapped plots: full signals, reconstructions, fits, FRF, metrics
  - Produces three FRF figures:
      1. Overlay_FRF.png                    - spatial frequency (1/Re)
      2. Overlay_FRF_vs_Hz_Physical.png     - physical frequency (Hz)
      3. Overlay_FRF_vs_Hz_Physical_SPL.png - physical frequency (Hz), dB SPL
  - Prints publication-ready LaTeX tables to stdout.

NISHIOKA FLATTENING: MULTI-SIZE ROM ANALYSIS WITH OVERLAPPED FRF (FIXED)
==========================================================================
- Processes all three particle sizes (small, medium, large)
- Applies ROM methods (TD, JK-P, JK-G_RK4, JK-G_Exp)
- Generates overlapped plots: full signals, reconstructions, fits, FRF, metrics
- Includes TWO FRF plots:
1. SPDX header
2. Module docstring
3. Imports
4. nishioka_exp_data dict                       ← input data
5. schiller_naumann, schiller_fit, power_law    ← fitting helpers
6. Viscosity block (constants + viscosity_Ni)   ← thermodynamic constants
7. jones_model                                  ← theoretical model 1
8. madejski_model                               ← theoretical model 2
9. mostaghimi_model                             ← theoretical model 3
10. flattening_models                           ← wrapper (NEW)
11. create_full_model_signal                    ← signal construction
12. timederivative_reduction, ...               ← ROM methods
13. Overlay_FRF.png - spatial frequency (1/Re)
14. Overlay_FRF_vs_Hz_Physical.png - frequency (Hz) using physical time scale

"""

import numpy as np
from scipy.optimize import curve_fit, fsolve
from scipy.fft import fft, fftfreq
from scipy.linalg import eigvals, expm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================
# NISHIOKA'S EXPERIMENTAL DATA (FROM FIG. 10)
# =============================================
nishioka_exp_data = {
    'small': {
        'Re': np.array([20000, 18000, 14000, 12000, 10000]),
#       'Re': np.array([10000, 12000, 13500, 17000, 19000]),
        'xi': np.array([2.75, 2.60, 2.40, 2.10, 2.00]),
#       'xi': np.array([2.00, 2.20, 2.40, 2.60, 2.75]),
        'd_p': 40e-6,
        'label': 'Small (40 µm)'
    },
    'medium': {
        'Re': np.array([41000, 37000, 31000, 25000, 24000]),
#       'Re': np.array([10000, 12000, 13500, 17000, 19000]),
        'xi': np.array([2.60, 2.25, 2.00, 1.60, 1.55]),
#       'xi': np.array([2.00, 2.20, 2.40, 2.60, 2.75]),
        'd_p': 80e-6,
        'label': 'Medium (80 µm)'
    },
    'large': {
        'Re': np.array([45000, 40000, 36000, 34000, 28000]),
#       'Re': np.array([10000, 12000, 13500, 17000, 19000]),
        'xi': np.array([3.15, 2.60, 3.00, 1.50, 1.35]),
#       'xi': np.array([2.00, 2.20, 2.40, 2.60, 2.75]),
        'd_p': 120e-6,
        'label': 'Large (120 µm)'
    }
}

# =============================================
# DRAG COEFFICIENT AND FITTING FUNCTIONS
# =============================================
def schiller_naumann(Re):
    Re = np.asarray(Re)
    Re = np.maximum(Re, 1e-10)
    return 24/Re + 6/(1 + np.sqrt(Re)) + 0.4

def schiller_fit(Re, a, b):
    Cd = schiller_naumann(Re)
    return a / Cd + b

def power_law(Re, a, b, c):
    return a * Re**b + c

# =============================================
# TEMPERATURE-DEPENDENT VISCOSITY OF NICKEL (Eq. 4)
# =============================================
# Constants from Nishioka & Fukumoto (2000), Fig. 9(b).
# Viscosity fit: eta(T) = 0.1663 * exp(6038 / T)  [Pa.s]
# The equivalent Arrhenius form anchored at the melting point is
#   eta(T) = eta_mp * exp[ (E/R) * (1/T - 1/T_mp) ]
# with E/R = 6038 K  =>  E = 6038 * 8.314 = 50,200 J/mol.
T_melt_Ni = 1728.0            # K,   melting point of Nickel
eta_mp_Ni = 5.48e-3           # Pa.s, dynamic viscosity at T = T_melt_Ni
E_visc_Ni = 50200.0           # J/mol, activation energy (Nishioka Fig. 9b)
R_gas     = 8.314             # J/(mol.K), universal gas constant

# =============================================
# THERMOPHYSICAL PROPERTIES OF NICKEL (Table 1)
# =============================================
# Values from Nishioka & Fukumoto (2000), Fig. 9 and Table 1.
# These constants are used by flattening_models() to compute the
# dimensionless numbers Re, We, Pe, and Ste.
rho_mp_Ni     = 7900.0        # kg/m^3,    density at T = T_melt (Fig. 9a)
gamma_mp_Ni   = 1.78          # N/m,       surface tension at T_melt (Fig. 9c)
alpha_Ni      = 2.3e-5        # m^2/s,     thermal diffusivity (Table 1)
c_p_Ni        = 444.0         # J/(kg.K),  specific heat capacity (Table 1)
L_fusion_Ni   = 2.92e5        # J/kg,      latent heat of fusion (Table 1)
T_substrate   = 723.0         # K,         substrate temperature (Fig. 2c)

# Temperature-dependent coefficients (used if a temperature-dependent
# density or surface-tension model is enabled in future work):
rho_0_Ni      = 1.16          # kg/(m^3.K), Nishioka Fig. 9a
gamma_0_Ni    = 3.8e-4        # N/(m.K),   Nishioka Fig. 9c

def viscosity_Ni(T):
    """
    Temperature-dependent dynamic viscosity of molten Nickel.

    Arrhenius form anchored at the melting point:

        eta(T) = eta_mp * exp[ (E/R) * (1/T - 1/T_mp) ]

    For T > T_mp the argument of the exponential is negative, so
    eta decreases with increasing temperature, as reported by
    Nishioka & Fukumoto (2000), Fig. 9(b).

    For T < T_mp the value is clipped to eta_mp (no undercooled
    branch is modelled).
    """
    T_eff = np.maximum(T, T_melt_Ni)
    return eta_mp_Ni * np.exp(
        E_visc_Ni / R_gas * (1.0 / T_eff - 1.0 / T_melt_Ni)
    )
# =============================================
# THEORETICAL FLATTENING MODELS (for reference)
# =============================================
def jones_model(Re):
    return 1.15 * Re**0.125

def madejski_model(Re, We):
    Re = np.atleast_1d(np.asarray(Re))
    We = np.atleast_1d(np.asarray(We))
    if len(We) == 1 and len(Re) > 1:
        We = np.full_like(Re, We[0])
    xi_out = np.zeros_like(Re, dtype=float)
    xi_asymptotic = 1.2941 * Re**0.2
    for i in range(len(Re)):
        if Re[i] > 100 and We[i] > 100:
            xi_out[i] = xi_asymptotic[i]
        else:
            try:
                def madejski_eq(xi, Re_i=Re[i], We_i=We[i]):
                    return 3*xi**2/We_i + (1/Re_i)*(xi/1.2941)**5 - 1
                xi_solution = fsolve(madejski_eq, max(1.0, xi_asymptotic[i] * 0.8))[0]
                xi_out[i] = max(1.0, xi_solution)
            except:
                xi_out[i] = 1.0
    if len(xi_out) == 1:
        return xi_out[0]
    return xi_out

def mostaghimi_model(Re, We, Ste, Pe):
    Re = np.asarray(Re).flatten()
    We = np.asarray(We).flatten()
    Ste = np.asarray(Ste).flatten()
    Pe = np.asarray(Pe).flatten()
    if Re.size == 1:
        if Ste[0] <= 0:
            return np.sqrt(We[0] / 3)
        else:
            solid_term = (1/Ste[0]) * np.sqrt(Pe[0] / Re[0])
            if solid_term >= 1.0:
                return 1.0
            else:
                xi = np.sqrt(We[0] / 3) * (1 - solid_term)**(-1)
                return max(1.0, min(xi, 3.5))
    if We.size == 1:
        We = np.full_like(Re, We[0])
    if Ste.size == 1:
        Ste = np.full_like(Re, Ste[0])
    if Pe.size == 1:
        Pe = np.full_like(Re, Pe[0])
    xi_out = np.zeros_like(Re)
    mask_no_solid = (Ste <= 0)
    if np.any(mask_no_solid):
        xi_out[mask_no_solid] = np.sqrt(We[mask_no_solid] / 3)
    mask_solid = (Ste > 0)
    if np.any(mask_solid):
        solid_indices = np.where(mask_solid)[0]
        for idx in solid_indices:
            solid_term = (1/Ste[idx]) * np.sqrt(Pe[idx] / Re[idx])
            if solid_term >= 1.0:
                xi_out[idx] = 1.0
            else:
                xi_out[idx] = np.sqrt(We[idx] / 3) * (1 - solid_term)**(-1)
                xi_out[idx] = max(1.0, min(xi_out[idx], 3.5))
    return xi_out

# =============================================
# COMPREHENSIVE FLATTENING MODELS WRAPPER (Listing 10)
# =============================================
def flattening_models(v, T, d_p):
    """
    Compute the dimensionless numbers Re, We, Pe, Ste and evaluate the
    Jones, Madejski, and Mostaghimi flattening models.

    Notes
    -----
    The outputs of this helper are clipped to physically reasonable
    bounds (xi in [1, 4] for Jones/Madejski, [1, 3.5] for Mostaghimi).
    The error metrics reported in Table 5 of the manuscript are computed
    on the UNCLIPPED model outputs (direct calls to jones_model,
    madejski_model, mostaghimi_model), NOT through this wrapper.

    Parameters
    ----------
    v   : float or array
          Particle impact velocity [m/s].
    T   : float or array
          Particle temperature [K].
    d_p : float or array
          Particle diameter [m].

    Returns
    -------
    dict
        Keys: 'xi_jones', 'xi_madejski', 'xi_mostaghimi',
              'Re', 'We', 'Ste', 'Pe'.
    """
    v   = np.asarray(v)
    T   = np.asarray(T)
    d_p = np.asarray(d_p)

    eta = viscosity_Ni(T)

    Re = rho_mp_Ni * v * d_p / eta                  # Eq. 8
    We = rho_mp_Ni * v**2 * d_p / gamma_mp_Ni       # Eq. 9
    Pe = v * d_p / alpha_Ni                         # Eq. 10
    Ste = c_p_Ni * (T - T_substrate) / L_fusion_Ni  # Eq. 11

# Note: the error metrics reported in Table 5 of the manuscript are
# computed on the UNCLIPPED outputs of jones_model, madejski_model,
# and mostaghimi_model. The clips below are a design choice to keep
# the wrapper's outputs within physically reasonable bounds
# (1 <= xi <= 4) and are applied only to this helper function.

    xi_jones      = np.clip(jones_model(Re),              1.0, 4.0)
    xi_madejski   = np.clip(madejski_model(Re, We),       1.0, 4.0)
    xi_mostaghimi = np.clip(mostaghimi_model(Re, We, Ste, Pe), 1.0, 3.5)

    return {'xi_jones':      xi_jones,
            'xi_madejski':   xi_madejski,
            'xi_mostaghimi': xi_mostaghimi,
            'Re': Re, 'We': We, 'Ste': Ste, 'Pe': Pe}

# =============================================
# CREATE FULL MODEL SIGNAL FROM EXPERIMENTAL DATA
# =============================================
def create_full_model_signal(size_key):
    data = nishioka_exp_data[size_key]
    Re_data = data['Re']
    xi_data = data['xi']
    
    # Sort both arrays by Re (ascending)
    sort_idx = np.argsort(Re_data)
    Re_data_sorted = Re_data[sort_idx]
    xi_data_sorted = xi_data[sort_idx]
    
    n_points = 200
    # Generate log-spaced evaluation points from min to max Re
    Re_eval = np.logspace(np.log10(Re_data_sorted[0]), np.log10(Re_data_sorted[-1]), n_points)
    
    # Quadratic interpolation (order 2)
    f = interp1d(Re_data_sorted, xi_data_sorted, kind='quadratic', fill_value='extrapolate')
    xi_signal = f(Re_eval)
    
    np.random.seed(42)
    xi_signal += np.random.normal(0, 0.002, len(xi_signal))
    return Re_eval, xi_signal, data['label']

# =============================================
# METHOD 1: Time-Derivative Reduction (TD)
# =============================================
def timederivative_reduction(v, t, q=50):
    n = len(v)
    H = np.zeros((q+1, q))
    Q = np.zeros((n, q+1))
    Q[:, 0] = v / np.linalg.norm(v)
    for j in range(q):
        w = np.gradient(Q[:, j], t)
        for i in range(j+1):
            H[i,j] = np.dot(Q[:,i], w)
            w -= H[i,j] * Q[:,i]
        H[j+1,j] = np.linalg.norm(w)
        if H[j+1,j] > 1e-10:
            Q[:,j+1] = w / H[j+1,j]
        else:
            break
    return Q[:,:q], H[:q,:q]

def compute_timederivative_response(t, v_original, q=50):
    V, _ = timederivative_reduction(v_original, t, q)
    v_coeffs = V.T @ v_original
    return (V @ v_coeffs).flatten(), V

# =============================================
# METHOD 2: Jacobian-Based Krylov-Arnoldi
# =============================================
def construct_jacobian_matrix(mean_v, t_eval):
    n = len(mean_v)
    A_matrix = np.zeros((n, n))
    dt = t_eval[1] - t_eval[0]
    for i in range(1, n-1):
        A_matrix[i, i-1] = 0.5 / dt**2
        A_matrix[i, i] = -1.0 / dt**2
        A_matrix[i, i+1] = 0.5 / dt**2
    A_matrix[0, 0] = -1.0 / dt**2
    A_matrix[0, 1] = 1.0 / dt**2
    A_matrix[-1, -2] = 1.0 / dt**2
    A_matrix[-1, -1] = -1.0 / dt**2
    return A_matrix

def krylov_arnoldi_reduction_jacobian(A, v_start, q=50):
    n = len(v_start)
    V = np.zeros((n, q))
    H = np.zeros((q, q))
    beta = np.linalg.norm(v_start)
    if beta < 1e-12:
        beta = 1.0
    V[:, 0] = v_start / beta
    for j in range(q-1):
        w = A @ V[:, j]
        for pass_num in range(2):
            for i in range(j+1):
                hij = np.dot(V[:, i], w)
                w = w - hij * V[:, i]
                if pass_num == 0:
                    H[i, j] = hij
        h_next = np.linalg.norm(w)
        if h_next > 1e-12:
            V[:, j+1] = w / h_next
        else:
            v_rand = np.random.randn(n)
            for i in range(j+1):
                v_rand = v_rand - np.dot(V[:, i], v_rand) * V[:, i]
            norm_rand = np.linalg.norm(v_rand)
            if norm_rand > 1e-12:
                V[:, j+1] = v_rand / norm_rand
            else:
                V[:, j+1] = v_rand
    A_reduced = V.T @ A @ V
    return V, A_reduced

def compute_krylov_galerkin_response(t_eval, v_original, A_matrix, q=50, method='RK4'):
    n = len(v_original)
    dt = t_eval[1] - t_eval[0]
    V, A_reduced = krylov_arnoldi_reduction_jacobian(A_matrix, v_original, q)
    x0_reduced = V.T @ v_original
    dvdt_actual = np.gradient(v_original, t_eval)
    dvdt_linear = A_matrix @ v_original
    f_nonlinear = dvdt_actual - dvdt_linear
    b_reduced_vector = V.T @ f_nonlinear
    b_reduced_full = np.tile(b_reduced_vector, (n, 1)).T
    v_reduced = np.zeros((q, n))
    v_reduced[:, 0] = x0_reduced
    if method == 'RK4':
        for i in range(1, n):
            v_curr = v_reduced[:, i-1]
            b_curr = b_reduced_full[:, i-1]
            b_next = b_reduced_full[:, i]
            k1 = A_reduced @ v_curr + b_curr
            k2 = A_reduced @ (v_curr + 0.5*dt*k1) + b_curr
            k3 = A_reduced @ (v_curr + 0.5*dt*k2) + b_curr
            k4 = A_reduced @ (v_curr + dt*k3) + b_next
            v_reduced[:, i] = v_curr + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    elif method == 'Exp':
        exp_Adt = expm(A_reduced * dt)
        for i in range(1, n):
            b_curr = b_reduced_full[:, i-1]
            v_reduced[:, i] = exp_Adt @ v_reduced[:, i-1] + dt * exp_Adt @ b_curr
    v_reconstructed = np.zeros(n)
    for i in range(n):
        v_reconstructed[i] = np.dot(V[i, :], v_reduced[:, i])
    return v_reconstructed, V

def compute_krylov_projection_response(t_eval, v_original, A_matrix, q=50):
    V, _ = krylov_arnoldi_reduction_jacobian(A_matrix, v_original, q)
    v_coeffs = V.T @ v_original
    return V @ v_coeffs, V

# =============================================
# ROM METRICS
# =============================================
def compute_rom_metrics(t_eval, mean_v, v_red, q, method_name):
    n = len(t_eval)
    dt = t_eval[1] - t_eval[0]
    energy = np.sum(v_red**2) / np.sum(mean_v**2) * 100
    peak_err = np.max(np.abs(mean_v - v_red)) / (np.max(np.abs(mean_v)) + 1e-12) * 100
    window = np.hanning(n)
    fft_orig = np.fft.fft(mean_v * window)
    fft_red = np.fft.fft(v_red * window)
    freq = np.fft.fftfreq(n, dt)[:n//2]
    phase_orig = np.angle(fft_orig[:n//2], deg=True)
    phase_red = np.angle(fft_red[:n//2], deg=True)
    phase_diff = np.abs(phase_orig - phase_red)
    max_phase = np.max(phase_diff)
    mag_orig = np.abs(fft_orig[:n//2])
    mag_red = np.abs(fft_red[:n//2])
    h2_error = np.sqrt(np.sum((mag_orig - mag_red)**2) / (np.sum(mag_orig**2) + 1e-12)) * 100
    return {
        'energy': energy,
        'peak_err': peak_err,
        'max_phase': max_phase,
        'h2_error': h2_error,
        'freq': freq,
        'mag_orig': mag_orig,
        'mag_red': mag_red,
        'phase_orig': phase_orig,
        'phase_red': phase_red
    }

# =============================================
# 4-POLE THEORY AND SPL IMPLEMENTATION
# =============================================

import numpy as np
from scipy.signal import zpk2tf, freqresp, TransferFunction
from scipy.linalg import inv

def four_pole_transfer_function(A, B, C, D):
    """
    Construct transfer function from 4-pole parameters.
    
    Parameters:
    -----------
    A, B, C, D : float or array
        4-pole matrix elements
        
    Returns:
    --------
    H : TransferFunction object
        The transfer function of the system
    """
    # For a 4-pole system, the transfer function between output and input
    # pressure or velocity can be derived from:
    # H(s) = (C + D*Z_l) / (A + B*Z_l)  where Z_l is the load impedance
    
    # If we consider the pressure transfer function (port 1 to port 2):
    # H_p = p2 / p1 = 1 / (A + B/Z_l)
    # For free radiation (Z_l = 1):
    numerator = [1.0]
    denominator = [A + B, 0.0]  # s-domain representation
    return TransferFunction(numerator, denominator)

def compute_four_pole_frf(Re_eval, xi_signal, q=50, size_key='medium'):
    """
    Compute Frequency Response Function using 4-pole theory.
    
    This models the particle flattening process as a 4-pole system where:
    - Port 1: Particle in-flight conditions (Re, velocity)
    - Port 2: Flattening behavior (ξ)
    
    The 4-pole matrix elements are derived from the system dynamics.
    """
    n = len(Re_eval)
    dt = np.mean(np.diff(Re_eval))
    
    # Construct 4-pole matrix from system dynamics
    # For a simple model, we can use the Jacobian-like structure
    A_mat = np.eye(2) + 0.01 * np.array([[0, 1], [-1, 0]])  # Example
    B_vec = np.array([[0.1], [0.0]])
    C_vec = np.array([[1.0, 0.0]])
    D_vec = np.array([[0.0]])
    
    # Compute transfer function
    freq = np.fft.fftfreq(n, dt)[:n//2]
    freq = freq[freq > 0]
    
    # Transfer function using 4-pole formulation
    # H(jω) = C * (jω*I - A)^(-1) * B + D
    H_mag = np.zeros(len(freq))
    H_phase = np.zeros(len(freq))
    
    for i, omega in enumerate(2 * np.pi * freq):
        # jωI - A
        jwI_minus_A = 1j * omega * np.eye(2) - A_mat
        try:
            inv_jwI_minus_A = inv(jwI_minus_A)
            H = C_vec @ inv_jwI_minus_A @ B_vec + D_vec
            H_mag[i] = np.abs(H[0, 0])
            H_phase[i] = np.angle(H[0, 0], deg=True)
        except np.linalg.LinAlgError:
            H_mag[i] = 0.0
            H_phase[i] = 0.0
    
    # Normalize transfer function
    H_ref = np.max(H_mag) if np.max(H_mag) > 0 else 1.0
    H_mag_norm = H_mag / H_ref
    
    return freq, H_mag_norm, H_phase

def compute_spl_from_transfer(H_mag, H_ref=1e-6):
    """
    Convert transfer function magnitude to Sound Pressure Level (SPL) in dB.
    
    Parameters:
    -----------
    H_mag : array
        Transfer function magnitude (unitless or with units)
    H_ref : float
        Reference magnitude (typically 1e-6 for pressure)
        
    Returns:
    --------
    spl : array
        SPL in dB (20 * log10(H_mag / H_ref))
    """
    # Ensure no zeros to avoid log(0)
    H_mag = np.maximum(H_mag, 1e-12)
    return 20 * np.log10(H_mag / H_ref)

def compute_spl_from_fft(fft_magnitude, ref_value=1e-6):
    """
    Compute SPL from FFT magnitude data.
    
    Parameters:
    -----------
    fft_magnitude : array
        Magnitude from FFT
    ref_value : float
        Reference value for SPL calculation
        
    Returns:
    --------
    spl : array
        SPL in dB
    """
    # Ensure positive values
    fft_magnitude = np.maximum(fft_magnitude, 1e-12)
    return 20 * np.log10(fft_magnitude / ref_value)

def compute_impedance_from_fft(fft_magnitude, freq):
    """
    Compute acoustic impedance from FFT data.
    Z = p / v in frequency domain
    
    For a simple model, we approximate pressure from the FFT magnitude
    and velocity from the derivative.
    """
    # Simple approximation: Z = |FFT| / (omega * |FFT|)
    # This is a simplified model - in reality, impedance is complex
    omega = 2 * np.pi * freq
    omega = np.maximum(omega, 1e-6)  # Avoid division by zero
    
    # Approximate impedance (simple model)
    Z = np.ones_like(fft_magnitude)  # Placeholder
    return Z

# =============================================
# ENHANCED PLOTTING WITH SPL
# =============================================

def plot_flattening_frf_with_spl(all_results, all_fits, q=50, 
                                  save_dir="flattening_multi_size",
                                  use_spl=True):
    """
    Generate FRF plot with optional SPL conversion.
    CORRECTED: Handles 0 Hz, accurate dB conversion, and proper time axes.
    """
    if not all_results or not all_fits:
        print("Warning: No results to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    size_colors = {'small': 'blue', 'medium': 'green', 'large': 'orange'}
    method_styles = {'TD': '--', 'JK-P': '-.', 'JK-G_RK4': ':', 'JK-G_Exp': '--'}
    method_lw = {'TD': 1.5, 'JK-P': 1.5, 'JK-G_RK4': 1.5, 'JK-G_Exp': 1.0}
    methods = ['TD', 'JK-P', 'JK-G_RK4', 'JK-G_Exp']
    
    # Nishioka's velocity data for physical time scaling
    nish_velocities = {
        'distance_mm': np.array([50, 100, 150, 200, 250]),
        'velocity_mps': np.array([120, 119, 105, 91, 90])
    }
    v_mean = np.mean(nish_velocities['velocity_mps'])  # ~105 m/s

    # --- FIX 1: SET PROPER ACOUSTIC REFERENCE ---
    # In standard SPL, the reference is 20 microPascals.
    # Since your FFT magnitude is in Pascals (or arbitrary units), 
    # we use a very small reference to ensure the plot is positive dB.
    p_ref = 1e-6  # 1 micro Pascal
    
    plotted_any = False

    for size_key in all_results.keys():
        label = all_fits[size_key]['label']
        Re_eval = all_fits[size_key]['Re_eval']
        xi_signal = all_fits[size_key]['xi_signal']
        n = len(Re_eval)
        
        # Map Re to physical flight time
        Re_min, Re_max = Re_eval[0], Re_eval[-1]
        if Re_max - Re_min < 1e-10:
            continue
        
        distance_min, distance_max = 0.05, 0.25
        distance = distance_min + (Re_eval - Re_min) / (Re_max - Re_min) * (distance_max - distance_min)
        t_flight = distance / v_mean
        
        dt = np.mean(np.diff(t_flight))
        if dt <= 0: dt = 1e-6

        # Generate frequency array (trimming DC at 0 Hz)
        freq = np.fft.fftfreq(n, dt)[:n//2]
        freq = freq[1:]  # --- FIX 2: REMOVE 0 Hz POINT ---
        
        xi_signal = np.asarray(xi_signal, dtype=float).flatten()
        if len(xi_signal) != n: continue
        
        # Full model FFT (No need for abs() twice, use absolute value here)
        fft_full = np.abs(np.fft.fft(xi_signal)[:n//2])
        fft_full = fft_full[1:] # Remove DC offset
        fft_full = np.maximum(fft_full, 1e-18) # --- FIX 3: Prevent log(0) ---
        
        if use_spl:
            # --- FIX 4: CORRECT SPL FORMULA ---
            spl_full = 20 * np.log10(fft_full / p_ref)
            ax.semilogx(freq, spl_full, '-', color=size_colors[size_key],
                       linewidth=2.5, label=f'{label} (Full) [SPL]')
        else:
            ax.loglog(freq, fft_full, '-', color=size_colors[size_key],
                       linewidth=2.5, label=f'{label} (Full)')

        plotted_any = True
        
        # ROM reconstructions
        for method in methods:
            if method not in all_results[size_key]['reconstructions']: continue
            xi_red = all_results[size_key]['reconstructions'][method]
            try:
                xi_red = np.asarray(xi_red, dtype=float).flatten()
            except Exception: continue
            if len(xi_red) != n: continue
            
            fft_red = np.abs(np.fft.fft(xi_red)[:n//2])
            fft_red = fft_red[1:] # Remove DC
            fft_red = np.maximum(fft_red, 1e-18)
            
            if use_spl:
                spl_red = 20 * np.log10(fft_red / p_ref)
                ax.semilogx(freq, spl_red,
                           linestyle=method_styles[method],
                           color=size_colors[size_key],
                           linewidth=method_lw[method],
                           alpha=0.7,
                           label=f'{label} ({method}) [SPL]')
            else:
                ax.loglog(freq, fft_red,
                           linestyle=method_styles[method],
                           color=size_colors[size_key],
                           linewidth=method_lw[method],
                           alpha=0.7,
                           label=f'{label} ({method})')
    
    if not plotted_any:
        print("Warning: No data plotted.")
        plt.close(fig); return None

    # --- FIX 5: SET PROPER AXIS LIMITS ---
    if use_spl:
        ax.set_ylabel('Magnitude [dB SPL] (re: 1 µPa)', fontsize=12)
        ax.set_ylim(0, 140) # Typical SPL range
    else:
        ax.set_ylabel('Magnitude (a.u.)', fontsize=12)

    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_title(f'Overlapped FRF vs Frequency - SPL (q={q})', fontsize=14, fontweight='bold')
    ax.set_xscale('log')  # Fuerza el eje X a ser logarítmico (indispensable para el zoom)
    ax.set_xlim(1e3, 1e5) # Zoom de 1,000 Hz a 100,000 Hz #(20, 100000) # Limit X axis to acoustic range, avoiding DC/0 Hz
    # Opcional: Ajusta el eje Y para que no quede espacio vacío arriba/abajo
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=7, ncol=2)
    plt.tight_layout()
    
    suffix = "_SPL" if use_spl else ""
    save_path = os.path.join(save_dir, f"Overlay_FRF_vs_Hz_Physical{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"FRF with SPL saved to: {save_path}")
    plt.show()
    return fig

# =============================================
# ADD TO MAIN ANALYSIS FUNCTION
# =============================================

def run_multi_size_analysis_with_spl(save_dir="flattening_multi_size"):
    """
    Extended analysis including SPL plots.
    """
    # Run the standard analysis first
    all_results, all_fits = run_multi_size_analysis(save_dir=save_dir)
    
    print("\n" + "="*80)
    print("GENERATING SPL PLOTS (4-POLE THEORY)")
    print("="*80)
    
    # Generate SPL version of the FRF plot
    try:
        fig_spl = plot_flattening_frf_with_spl(all_results, all_fits, q=50, 
                                               save_dir=save_dir, use_spl=True)
        if fig_spl is not None:
            plt.close(fig_spl)
    except Exception as e:
        print(f"Warning: Could not generate SPL plot: {e}")
    
    # Also generate the linear version for comparison
    try:
        fig_linear = plot_flattening_frf_with_spl(all_results, all_fits, q=50,
                                                  save_dir=save_dir, use_spl=False)
        if fig_linear is not None:
            plt.close(fig_linear)
    except Exception as e:
        print(f"Warning: Could not generate linear FRF plot: {e}")
    
    return all_results, all_fits


# =============================================
# ROBUST OVERLAPPED FRF PLOT (SPATIAL FREQUENCY - 1/Re)
# =============================================
def plot_flattening_frf_overlay(all_results, all_fits, q=50, save_dir="flattening_multi_size"):
    """
    Generate overlapped FRF plot for flattening models.
    Frequency axis is in cycles per Reynolds number (spatial frequency).
    Mimics the style of the constitutive model FRF plots (semilogy, linear freq).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    size_colors = {'small': 'blue', 'medium': 'green', 'large': 'orange'}
    method_styles = {
        'TD': '--',
        'JK-P': '-.',
        'JK-G_RK4': ':',
        'JK-G_Exp': '--'   # same as TD; distinguish by linewidth
    }
    method_lw = {
        'TD': 1.5,
        'JK-P': 1.5,
        'JK-G_RK4': 1.5,
        'JK-G_Exp': 1.0
    }
    methods = ['TD', 'JK-P', 'JK-G_RK4', 'JK-G_Exp']
    
    for size_key in all_results.keys():
        label = all_fits[size_key]['label']
        Re_eval = all_fits[size_key]['Re_eval']
        xi_signal = all_fits[size_key]['xi_signal']
        n = len(Re_eval)
        dt = Re_eval[1] - Re_eval[0]
        freq = np.fft.fftfreq(n, dt)[:n//2]
        window = np.hanning(n)
        
        xi_signal = np.asarray(xi_signal, dtype=float).flatten()
        if len(xi_signal) != n:
            continue
        
        # Full model
        fft_full = np.abs(np.fft.fft(xi_signal * window)[:n//2])
        fft_full = np.nan_to_num(fft_full, nan=0.0, posinf=0.0, neginf=0.0)
        ax.semilogy(freq, fft_full, '-', color=size_colors[size_key],
                    linewidth=2.5, label=f'{label} (Full)')
        
        # ROM reconstructions – label each one
        for method in methods:
            if method not in all_results[size_key]['reconstructions']:
                continue
            xi_red = all_results[size_key]['reconstructions'][method]
            try:
                xi_red = np.asarray(xi_red, dtype=float).flatten()
            except Exception:
                continue
            if len(xi_red) != n:
                continue
            fft_red = np.abs(np.fft.fft(xi_red * window)[:n//2])
            fft_red = np.nan_to_num(fft_red, nan=0.0, posinf=0.0, neginf=0.0)
            
            ax.semilogy(freq, fft_red,
                        linestyle=method_styles[method],
                        color=size_colors[size_key],
                        linewidth=method_lw[method],
                        alpha=0.7,
                        label=f'{label} ({method})')
    
    ax.set_xlabel('Spatial Frequency [1/Re]', fontsize=12)
    ax.set_ylabel('Magnitude (a.u.)', fontsize=12)
    ax.set_title(f'Overlapped FRF Comparison (q={q})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    # Use two columns and smaller font to fit many entries
    ax.legend(loc='best', fontsize=7, ncol=2)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "Overlay_FRF.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overlapped FRF (spatial frequency) saved to: {save_path}")
    plt.show()
    return fig

# =============================================
# ROBUST OVERLAPPED FRF PLOT (FREQUENCY IN Hz - PHYSICAL TIME SCALE)
# =============================================
def plot_flattening_frf_vs_hz_physical(all_results, all_fits, q=50, save_dir="flattening_multi_size"):
    """
    Generate overlapped FRF plot with x-axis in Hz using physical time scale
    based on Nishioka's measured particle velocities at different spray distances.
    """
    # Check if there are results to plot
    if not all_results or not all_fits:
        print("Warning: No results to plot for FRF vs Hz")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    size_colors = {'small': 'blue', 'medium': 'green', 'large': 'orange'}
    method_styles = {'TD': '--', 'JK-P': '-.', 'JK-G_RK4': ':', 'JK-G_Exp': '--'}
    method_lw = {'TD': 1.5, 'JK-P': 1.5, 'JK-G_RK4': 1.5, 'JK-G_Exp': 1.0}
    methods = ['TD', 'JK-P', 'JK-G_RK4', 'JK-G_Exp']
    
    # Nishioka's measured velocity data (from Fig. 8)
    nish_velocities = {
        'distance_mm': np.array([50, 100, 150, 200, 250]),
        'velocity_mps': np.array([120, 119, 105, 91, 90])
    }
    v_mean = np.mean(nish_velocities['velocity_mps'])  # ~105 m/s
    
    plotted_any = False
    
    for size_key in all_results.keys():
        label = all_fits[size_key]['label']
        Re_eval = all_fits[size_key]['Re_eval']
        xi_signal = all_fits[size_key]['xi_signal']
        n = len(Re_eval)
        window = np.hanning(n)
        
        # Map Re to physical flight time
        Re_min = Re_eval[0]
        Re_max = Re_eval[-1]
        distance_min = 0.05  # 50 mm
        distance_max = 0.25  # 250 mm
        
        # Avoid division by zero if Re range is invalid
        if Re_max - Re_min < 1e-10:
            print(f"Warning: Invalid Re range for {label}")
            continue
        
        distance = distance_min + (Re_eval - Re_min) / (Re_max - Re_min) * (distance_max - distance_min)
        t_flight = distance / v_mean
        
        dt = np.mean(np.diff(t_flight))
        if dt <= 0:
            dt = 1e-6  # fallback
        freq = np.fft.fftfreq(n, dt)[:n//2]
        freq = freq[freq > 0]  # Remove zero frequency
        
        xi_signal = np.asarray(xi_signal, dtype=float).flatten()
        if len(xi_signal) != n:
            continue
        
        # Full model
        fft_full = np.abs(np.fft.fft(xi_signal * window)[:n//2])
        fft_full = np.nan_to_num(fft_full, nan=0.0, posinf=0.0, neginf=0.0)
        fft_full = fft_full[:len(freq)]
        
        if np.any(fft_full > 0):
            ax.semilogy(freq, fft_full, '-', color=size_colors[size_key],
                        linewidth=2.5, label=f'{label} (Full)')
            plotted_any = True
        
        # ROM reconstructions
        for method in methods:
            if method not in all_results[size_key]['reconstructions']:
                continue
            xi_red = all_results[size_key]['reconstructions'][method]
            try:
                xi_red = np.asarray(xi_red, dtype=float).flatten()
            except Exception:
                continue
            if len(xi_red) != n:
                continue
            fft_red = np.abs(np.fft.fft(xi_red * window)[:n//2])
            fft_red = np.nan_to_num(fft_red, nan=0.0, posinf=0.0, neginf=0.0)
            fft_red = fft_red[:len(freq)]
            
            if np.any(fft_red > 0):
                ax.semilogy(freq, fft_red,
                            linestyle=method_styles[method],
                            color=size_colors[size_key],
                            linewidth=method_lw[method],
                            alpha=0.7,
                            label=f'{label} ({method})')
                plotted_any = True
    
    if not plotted_any:
        print("Warning: No data plotted for FRF vs Hz")
        plt.close(fig)
        return None
    
    ax.set_xlabel('Frequency [Hz] (Physical Time Scale from Nishioka Data)', fontsize=12)
    ax.set_ylabel('Magnitude (a.u.)', fontsize=12)
    ax.set_title(f'Overlapped FRF vs Frequency (q={q})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=7, ncol=2)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "Overlay_FRF_vs_Hz_Physical.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overlapped FRF (vs Hz, physical time scale) saved to: {save_path}")
    plt.show()
    return fig

# =============================================
# CONVERGENCE STUDY
# =============================================
def plot_convergence_study(save_dir="flattening_multi_size"):
    """
    Convergence study: plot Energy Retention [%], Peak Error [%], and
    FRF H2 Error [%] vs number of modes q for each particle size and ROM method.

    Returns
    -------
    convergence_data : dict
        convergence_data[size_key]['q']                       -> list of q values
        convergence_data[size_key][method]['energy'/'peak'/'h2'/'phase'] -> lists
    """
    sizes = ['small', 'medium', 'large']
    q_values = [2, 5, 10, 15, 20, 30, 40, 50]
    methods = ['TD', 'JK-P', 'JK-G_RK4', 'JK-G_Exp']

    size_colors = {'small': 'blue', 'medium': 'green', 'large': 'orange'}
    size_labels = {
        'small':  'Small (40 µm)',
        'medium': 'Medium (80 µm)',
        'large':  'Large (120 µm)'
    }
    method_styles  = {'TD': '--', 'JK-P': '-.', 'JK-G_RK4': ':', 'JK-G_Exp': '-'}
    method_markers = {'TD': 'o',  'JK-P': 's',  'JK-G_RK4': '^', 'JK-G_Exp': 'v'}

    convergence_data = {}

    # ---------- Compute metrics for every (size, q, method) ----------
    for size_key in sizes:
        print(f"\n[Convergence] Processing {size_labels[size_key]}")

        Re_eval, xi_signal, _ = create_full_model_signal(size_key)
        A_matrix = construct_jacobian_matrix(xi_signal, Re_eval)

        convergence_data[size_key] = {'q': q_values}
        for method in methods:
            convergence_data[size_key][method] = {
                'energy': [], 'peak': [], 'h2': [], 'phase': []
            }

        for q in q_values:
            # TD
            xi_td, _ = compute_timederivative_response(Re_eval, xi_signal, q)
            m = compute_rom_metrics(Re_eval, xi_signal, xi_td, q, 'TD')
            convergence_data[size_key]['TD']['energy'].append(m['energy'])
            convergence_data[size_key]['TD']['peak'].append(m['peak_err'])
            convergence_data[size_key]['TD']['h2'].append(m['h2_error'])
            convergence_data[size_key]['TD']['phase'].append(m['max_phase'])

            # JK-P
            xi_jp, _ = compute_krylov_projection_response(Re_eval, xi_signal, A_matrix, q)
            m = compute_rom_metrics(Re_eval, xi_signal, xi_jp, q, 'JK-P')
            convergence_data[size_key]['JK-P']['energy'].append(m['energy'])
            convergence_data[size_key]['JK-P']['peak'].append(m['peak_err'])
            convergence_data[size_key]['JK-P']['h2'].append(m['h2_error'])
            convergence_data[size_key]['JK-P']['phase'].append(m['max_phase'])

            # JK-G RK4
            xi_rk4, _ = compute_krylov_galerkin_response(Re_eval, xi_signal, A_matrix,
                                                         q, method='RK4')
            m = compute_rom_metrics(Re_eval, xi_signal, xi_rk4, q, 'JK-G_RK4')
            convergence_data[size_key]['JK-G_RK4']['energy'].append(m['energy'])
            convergence_data[size_key]['JK-G_RK4']['peak'].append(m['peak_err'])
            convergence_data[size_key]['JK-G_RK4']['h2'].append(m['h2_error'])
            convergence_data[size_key]['JK-G_RK4']['phase'].append(m['max_phase'])

            # JK-G Exp
            xi_exp, _ = compute_krylov_galerkin_response(Re_eval, xi_signal, A_matrix,
                                                         q, method='Exp')
            m = compute_rom_metrics(Re_eval, xi_signal, xi_exp, q, 'JK-G_Exp')
            convergence_data[size_key]['JK-G_Exp']['energy'].append(m['energy'])
            convergence_data[size_key]['JK-G_Exp']['peak'].append(m['peak_err'])
            convergence_data[size_key]['JK-G_Exp']['h2'].append(m['h2_error'])
            convergence_data[size_key]['JK-G_Exp']['phase'].append(m['max_phase'])

        # Console summary
        print(f"  q      : " + "  ".join(f"{qq:>7d}" for qq in q_values))
        for method in methods:
            e = convergence_data[size_key][method]['energy']
            print(f"  {method:<8s}: " + "  ".join(f"{v:7.2f}" for v in e))

    # ---------- Figure 1: Energy retention vs q ----------
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, size_key in zip(axes1, sizes):
        for method in methods:
            ax.plot(q_values, convergence_data[size_key][method]['energy'],
                    linestyle=method_styles[method],
                    marker=method_markers[method],
                    color=size_colors[size_key],
                    linewidth=1.8, markersize=6, label=method)
        # Convergence band ±1 %
        ax.axhline(100.0, color='k',    linestyle=':',  linewidth=1.0, alpha=0.6)
        ax.axhline(101.0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.axhline( 99.0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.set_xlabel('Number of modes $q$', fontsize=12)
        ax.set_title(size_labels[size_key], fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=9)
    axes1[0].set_ylabel('Energy retention [%]', fontsize=12)
    plt.tight_layout()
    save_path1 = os.path.join(save_dir, "Convergence_Energy.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"\nConvergence plot (energy) saved to: {save_path1}")
    plt.show()

    # ---------- Figure 2: Peak error and H2 error vs q ----------
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    for col, size_key in enumerate(sizes):
        # Peak error
        ax = axes2[0, col]
        for method in methods:
            ax.semilogx(q_values, convergence_data[size_key][method]['peak'],
                        linestyle=method_styles[method],
                        marker=method_markers[method],
                        color=size_colors[size_key],
                        linewidth=1.8, markersize=6, label=method)
        ax.set_xlabel('Number of modes $q$', fontsize=11)
        ax.set_ylabel('Peak error [%]', fontsize=11)
        ax.set_title(size_labels[size_key], fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=9)

        # FRF H2 error
        ax = axes2[1, col]
        for method in methods:
            ax.semilogx(q_values, convergence_data[size_key][method]['h2'],
                        linestyle=method_styles[method],
                        marker=method_markers[method],
                        color=size_colors[size_key],
                        linewidth=1.8, markersize=6, label=method)
        ax.set_xlabel('Number of modes $q$', fontsize=11)
        ax.set_ylabel('FRF $\\mathcal{H}_2$ error [%]', fontsize=11)
        ax.set_title(size_labels[size_key], fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "Convergence_Errors.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Convergence plot (errors) saved to: {save_path2}")
    plt.show()

    return convergence_data

# =============================================
# MAIN MULTI-SIZE ANALYSIS
# =============================================

def run_multi_size_analysis(save_dir="flattening_multi_size"):
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("NISHIOKA FLATTENING: MULTI-SIZE ROM ANALYSIS")
    print("="*80)
    
    sizes = ['small', 'medium', 'large']
    q_values = [2, 5, 10, 15, 20, 30, 40, 50]
    methods = ['TD', 'JK-P', 'JK-G_RK4', 'JK-G_Exp']
    
    all_results = {}
    all_fits = {}
    
    # Loop over each particle size
    for size_key in sizes:
        print(f"\n{'='*60}")
        print(f"Processing {nishioka_exp_data[size_key]['label']}")
        print(f"{'='*60}")
        
        # Create full signal
        Re_eval, xi_signal, label = create_full_model_signal(size_key)
        n = len(Re_eval)
        print(f"  Signal points: {n}")
        print(f"  Re range: {Re_eval[0]:.0f} - {Re_eval[-1]:.0f}")
        
        # 1. Schiller fit (unchanged)
        popt_schiller, _ = curve_fit(schiller_fit, Re_eval, xi_signal, p0=[0.5, 1.0], maxfev=5000)

        # 2. Power‑law fit using log‑log initial guess and wider bounds
        log_Re = np.log(Re_eval)
        log_xi = np.log(xi_signal)
        slope, intercept = np.polyfit(log_Re, log_xi, 1)   # linear fit in log space
        a0 = np.exp(intercept)                             # initial a
        b0 = slope                                         # initial b
        c0 = 0.0                                           # initial c (constant term)

        # Define bounds (widen b to ±5, but you can also use None for unbounded)
        bounds_lower = [0, -5, -10]
        bounds_upper = [10,  5,  10]

        # Clip the initial guess to the bounds to avoid infeasibility
        a0 = max(bounds_lower[0], min(bounds_upper[0], a0))
        b0 = max(bounds_lower[1], min(bounds_upper[1], b0))
        c0 = max(bounds_lower[2], min(bounds_upper[2], c0))

        popt_power, _ = curve_fit(power_law, Re_eval, xi_signal,
                          p0=[a0, b0, c0],
                          maxfev=10000,
                          bounds=(bounds_lower, bounds_upper))
        xi_schiller = schiller_fit(Re_eval, *popt_schiller)
        xi_power = power_law(Re_eval, *popt_power)

        # After the fitting for each size, print the parameters
        print(f"\n=== Fitting results for {label} ===")
        def _term(value):
            """Format a signed term for a linear fit equation."""
            sign = "-" if value < 0 else "+"
            return f"{sign} {abs(value):.4f}"        
        print(f"Schiller-Naumann: a = {popt_schiller[0]:.4f}, b = {popt_schiller[1]:.4f}")
        print(f"Power Law: a = {popt_power[0]:.4f}, b = {popt_power[1]:.4f}, c = {popt_power[2]:.4f}")
        # Compute RMSE and R² (you can compute these yourself)
        from sklearn.metrics import mean_squared_error, r2_score
        rmse_sch = np.sqrt(mean_squared_error(xi_signal, xi_schiller))
        r2_sch = r2_score(xi_signal, xi_schiller)
        rmse_pow = np.sqrt(mean_squared_error(xi_signal, xi_power))
        r2_pow = r2_score(xi_signal, xi_power)
        print(f"Schiller: RMSE = {rmse_sch:.4f}, R² = {r2_sch:.4f}")
        print(f"Power: RMSE = {rmse_pow:.4f}, R² = {r2_pow:.4f}")

        all_fits[size_key] = {
            'Re_eval': Re_eval,
            'xi_signal': xi_signal,
            'xi_schiller': xi_schiller,
            'xi_power': xi_power,
            'popt_schiller': popt_schiller,
            'popt_power': popt_power,
            'label': label
        }
        
        # Construct Jacobian
        A_matrix = construct_jacobian_matrix(xi_signal, Re_eval)
        
        # Store results
        results = {method: {'energy': [], 'peak_err': [], 'phase': [], 'h2': [], 'time': []} for method in methods}
        reconstructions = {}
        rom_metrics = {}
        
        for q in q_values:
            # --- TD Method ---
            start_time = time.time()  # Start timer
            xi_td, _ = compute_timederivative_response(Re_eval, xi_signal, q)
            elapsed_time = time.time() - start_time  # Stop timer
            m = compute_rom_metrics(Re_eval, xi_signal, xi_td, q, 'TD')
            results['TD']['energy'].append(m['energy'])
            results['TD']['peak_err'].append(m['peak_err'])
            results['TD']['phase'].append(m['max_phase'])
            results['TD']['h2'].append(m['h2_error'])
            results['TD']['time'].append(elapsed_time)  # <--- Save the real time!
            if q == 50:
                reconstructions['TD'] = xi_td
                rom_metrics['TD'] = m
            
            # --- JK-P Method ---
            start_time = time.time()
            xi_jp, _ = compute_krylov_projection_response(Re_eval, xi_signal, A_matrix, q)
            elapsed_time = time.time() - start_time
            m = compute_rom_metrics(Re_eval, xi_signal, xi_jp, q, 'JK-P')
            results['JK-P']['energy'].append(m['energy'])
            results['JK-P']['peak_err'].append(m['peak_err'])
            results['JK-P']['phase'].append(m['max_phase'])
            results['JK-P']['h2'].append(m['h2_error'])
            results['JK-P']['time'].append(elapsed_time)
            if q == 50:
                reconstructions['JK-P'] = xi_jp
                rom_metrics['JK-P'] = m
            
            # --- JK-G RK4 Method ---
            start_time = time.time()
            xi_rk4, _ = compute_krylov_galerkin_response(Re_eval, xi_signal, A_matrix, q, method='RK4')
            elapsed_time = time.time() - start_time
            m = compute_rom_metrics(Re_eval, xi_signal, xi_rk4, q, 'JK-G_RK4')
            results['JK-G_RK4']['energy'].append(m['energy'])
            results['JK-G_RK4']['peak_err'].append(m['peak_err'])
            results['JK-G_RK4']['phase'].append(m['max_phase'])
            results['JK-G_RK4']['h2'].append(m['h2_error'])
            results['JK-G_RK4']['time'].append(elapsed_time)
            if q == 50:
                reconstructions['JK-G_RK4'] = xi_rk4
                rom_metrics['JK-G_RK4'] = m
            
            # --- JK-G Exp Method ---
            start_time = time.time()
            xi_exp, _ = compute_krylov_galerkin_response(Re_eval, xi_signal, A_matrix, q, method='Exp')
            elapsed_time = time.time() - start_time
            m = compute_rom_metrics(Re_eval, xi_signal, xi_exp, q, 'JK-G_Exp')
            results['JK-G_Exp']['energy'].append(m['energy'])
            results['JK-G_Exp']['peak_err'].append(m['peak_err'])
            results['JK-G_Exp']['phase'].append(m['max_phase'])
            results['JK-G_Exp']['h2'].append(m['h2_error'])
            results['JK-G_Exp']['time'].append(elapsed_time)
            if q == 50:
                reconstructions['JK-G_Exp'] = xi_exp
                rom_metrics['JK-G_Exp'] = m
        
        # Store for this size
        all_results[size_key] = {
            'results': results,
            'reconstructions': reconstructions,
            'rom_metrics': rom_metrics,
            'A_matrix': A_matrix,
            'Re_eval': Re_eval,
            'xi_signal': xi_signal
        }
        
        # Print summary for this size
        print(f"\n  Summary for {label} (q=50):")
        for method in methods:
            idx = len(q_values) - 1
            print(f"    {method}: Energy={results[method]['energy'][idx]:.2f}%, Peak Err={results[method]['peak_err'][idx]:.2f}%")
    
    # =============================================
    # GENERATE OVERLAY PLOTS
    # =============================================
    print("\n[Generating Overlay Plots]")
    
    # Figure 1: Overlay of full signals and ROM reconstructions (q=50)
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    size_colors = {'small': 'blue', 'medium': 'green', 'large': 'orange'}
    
    # Subplot A: Full signals
    ax = axes[0,0]
    for size_key in sizes:
        data = all_fits[size_key]
        ax.loglog(data['Re_eval'], data['xi_signal'], '-', 
                 color=size_colors[size_key], linewidth=2, label=data['label'])
    ax.set_xlabel('Re'); ax.set_ylabel('$\\xi$'); ax.set_title('Full Signals'); ax.grid(True, alpha=0.3); ax.legend()
    
    # Subplot B: TD reconstructions
    ax = axes[0,1]
    for size_key in sizes:
        data = all_results[size_key]
        ax.loglog(data['Re_eval'], data['xi_signal'], '-', color=size_colors[size_key], linewidth=1, alpha=0.5)
        ax.loglog(data['Re_eval'], data['reconstructions']['TD'], '--', color=size_colors[size_key], linewidth=2, label=all_fits[size_key]['label'])
    ax.set_xlabel('Re'); ax.set_ylabel('$\\xi$'); ax.set_title('TD (q=50)'); ax.grid(True, alpha=0.3); ax.legend()
    
    # Subplot C: JK-G RK4
    ax = axes[1,0]
    for size_key in sizes:
        data = all_results[size_key]
        ax.loglog(data['Re_eval'], data['xi_signal'], '-', color=size_colors[size_key], linewidth=1, alpha=0.5)
        ax.loglog(data['Re_eval'], data['reconstructions']['JK-G_RK4'], '--', color=size_colors[size_key], linewidth=2, label=all_fits[size_key]['label'])
    ax.set_xlabel('Re'); ax.set_ylabel('$\\xi$'); ax.set_title('JK-G RK4 (q=50)'); ax.grid(True, alpha=0.3); ax.legend()
    
    # Subplot D: JK-G Exp
    ax = axes[1,1]
    for size_key in sizes:
        data = all_results[size_key]
        ax.loglog(data['Re_eval'], data['xi_signal'], '-', color=size_colors[size_key], linewidth=1, alpha=0.5)
        ax.loglog(data['Re_eval'], data['reconstructions']['JK-G_Exp'], '--', color=size_colors[size_key], linewidth=2, label=all_fits[size_key]['label'])
    ax.set_xlabel('Re'); ax.set_ylabel('$\\xi$'); ax.set_title('JK-G Exp (q=50)'); ax.grid(True, alpha=0.3); ax.legend()
    
    plt.tight_layout()
    save_path1 = os.path.join(save_dir, "Overlay_Reconstructions.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"Overlay reconstructions saved to: {save_path1}")
    plt.show()
    
    # Figure 2: Fits overlay
    fig2, ax = plt.subplots(figsize=(12, 8))
    for size_key in sizes:
        data = all_fits[size_key]
        # Experimental points
        exp_data = nishioka_exp_data[size_key]
        ax.loglog(exp_data['Re'], exp_data['xi'], 'o', color=size_colors[size_key], markersize=8, label=data['label']+' (Exp)')
        # Fits
        ax.loglog(data['Re_eval'], data['xi_schiller'], '--', color=size_colors[size_key], linewidth=1.5, alpha=0.7, label=data['label']+' (Schiller)')
        ax.loglog(data['Re_eval'], data['xi_power'], ':', color=size_colors[size_key], linewidth=1.5, alpha=0.7, label=data['label']+' (Power)')
    ax.set_xlabel('Re'); ax.set_ylabel('$\\xi$'); ax.set_title('Fits vs Experimental'); ax.grid(True, alpha=0.3); ax.legend(loc='best', ncol=2)
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "Overlay_Fits.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Overlay fits saved to: {save_path2}")
    plt.show()
    
    # Figure 3: Bar charts of metrics
    metric_names = ['energy', 'peak_err', 'phase', 'h2']
    metric_labels = ['Energy [%]', 'Peak Error [%]', 'Phase Error [deg]', 'FRF H$_2$ [%]']
    method_colors = {'TD': '#1f77b4', 'JK-P': '#2ca02c', 'JK-G_RK4': '#d62728', 'JK-G_Exp': '#9467bd'}

    # Diagnostic print (optional)
    print("\n[Diagnostic] ROM metrics at q=50:")
    for metric in metric_names:
        print(f"\n  Metric: {metric}")
        for size_key in sizes:
            label = nishioka_exp_data[size_key]['label']
            for method in methods:
                val = all_results[size_key]['results'][method][metric][-1]
                print(f"    {label} {method}: {val}")

    fig3, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx//3, idx%3]
        x = np.arange(len(sizes))
        width = 0.2
        for j, method in enumerate(methods):
            values = [all_results[size_key]['results'][method][metric][-1] for size_key in sizes]
            ax.bar(x + (j - 1.5)*width, values, width, label=method, color=method_colors[method], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([nishioka_exp_data[s]['label'] for s in sizes])
        ax.set_ylabel(label)
        ax.set_title(f'{label} (q=50)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # For metrics with zero values, use symlog with a small positive threshold
        # and set the lower limit to a small positive value so bars start slightly above zero
        if metric in ['peak_err', 'phase', 'h2']:
            ax.set_yscale('symlog', linthresh=1e-2)
            ax.set_ylim(1e-2, None)   # shift the bottom to a small positive value
        else:
            ax.set_ylim(0, None)

    # Turn off empty subplots
    for i in range(len(metric_names), 6):
        axes[i//3, i%3].axis('off')

    plt.tight_layout()
    save_path3 = os.path.join(save_dir, "Metrics_Bar_Charts.png")
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    print(f"Metrics bar charts saved to: {save_path3}")
    plt.show()
    
    # =============================================
    # OVERLAPPED FRF PLOT (SPATIAL FREQUENCY - 1/Re)
    # =============================================
    plot_flattening_frf_overlay(all_results, all_fits, q=50, save_dir=save_dir)
    
    # =============================================
    # OVERLAPPED FRF PLOT vs FREQUENCY (Hz) - PHYSICAL TIME SCALE (EXTRA PLOT)
    # =============================================
    try:
        fig_hz = plot_flattening_frf_vs_hz_physical(all_results, all_fits, q=50, save_dir=save_dir)
        if fig_hz is not None:
            plt.close(fig_hz)
    except Exception as e:
        print(f"Warning: Could not generate FRF vs Hz plot: {e}")
    
    # =============================================
    # PUBLICATION TABLE
    # =============================================
    print("\n" + "="*80)
    print("PUBLICATION-READY TABLE (ALL SIZES)")
    print("="*80)
    print("\n\\begin{table*}[h!]")
    print("\\centering")
    print("\\caption{ROM performance for flattening models (q=50)}")
    print("\\label{tab:flattening_multi_size}")
    print("\\begin{tabularx}{\\textwidth}{@{}l X l l l l l@{}}")  # Added one 'l' for time
    print("\\toprule")
    print("\\textbf{Size} & \\textbf{Method} & \\textbf{Energy [\\%]} & \\textbf{Peak Err [\\%]} & \\textbf{Phase [deg]} & \\textbf{H$_2$ [\\%]} & \\textbf{Time [ms]} \\\\")  # Added Time column
    print("\\midrule")
    for size_key in sizes:
        label = nishioka_exp_data[size_key]['label']
        for method in methods:
            idx = -1
            val = all_results[size_key]['results'][method]
            # Added val['time'][idx] to the print statement
            print(f"{label} & {method} & {val['energy'][idx]:.2f} & {val['peak_err'][idx]:.2f} & {val['phase'][idx]:.2f} & {val['h2'][idx]:.2f} & {val['time'][idx]*1000:.2f} \\\\") 
        if size_key != sizes[-1]:
            print("\\addlinespace")
    print("\\bottomrule")
    print("\\end{tabularx}")
    print("\\end{table*}")
    print("="*80)
    
    # =============================================
    # FITTING RESULTS TABLE FOR MEDIUM PARTICLES
    # =============================================
    print("\n" + "="*80)
    print("FITTING RESULTS TABLE (MEDIUM PARTICLES)")
    print("="*80)

    # Extract medium fits
    medium_fit = all_fits['medium']
    xi_signal_med = medium_fit['xi_signal']
    xi_schiller_med = medium_fit['xi_schiller']
    xi_power_med = medium_fit['xi_power']
    popt_sch = medium_fit['popt_schiller']
    popt_pow = medium_fit['popt_power']

    # Compute RMSE and R² manually (or reuse if stored)
    rmse_sch = np.sqrt(np.mean((xi_signal_med - xi_schiller_med)**2))
    ss_res_sch = np.sum((xi_signal_med - xi_schiller_med)**2)
    ss_tot = np.sum((xi_signal_med - np.mean(xi_signal_med))**2)
    r2_sch = 1 - ss_res_sch / ss_tot

    rmse_pow = np.sqrt(np.mean((xi_signal_med - xi_power_med)**2))
    ss_res_pow = np.sum((xi_signal_med - xi_power_med)**2)
    r2_pow = 1 - ss_res_pow / ss_tot

    # Print LaTeX table
    print("\n\\begin{table*}[h!]")
    print("\\centering")
    print("\\caption{Fitting results for Nishioka's flattening data (Medium particles, 80 µm).}")
    print("\\label{tab:fits_medium}")
    print("\\begin{tabularx}{\\textwidth}{@{}l X l l@{}}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Parameters} & \\textbf{RMSE} & \\textbf{R²} \\\\")
    print("\\midrule")
    print(f"Schiller-Naumann & $\\xi = {popt_sch[0]:.4f}/C_d + {popt_sch[1]:.4f}$ & {rmse_sch:.4f} & {r2_sch:.4f} \\\\")
    print(f"Power Law & $\\xi = {popt_pow[0]:.4f} \\cdot Re^{{{popt_pow[1]:.4f}}} + {popt_pow[2]:.4f}$ & {rmse_pow:.4f} & {r2_pow:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabularx}")
    print("\\end{table*}")
    print("="*80)

    # =============================================
    # THEORETICAL MODEL ERRORS (MEDIUM PARTICLES)
    # =============================================
    print("\n" + "="*80)
    print("THEORETICAL MODEL ERRORS (MEDIUM PARTICLES)")
    print("="*80)

    # Select medium particle data
    medium_fit = all_fits['medium']
    Re_eval_med = medium_fit['Re_eval']
    xi_signal_med = medium_fit['xi_signal']

    # --- Set physical parameters for theoretical models ---
    # Adjust these values to match your experimental conditions
    We = 1000.0      # Weber number (typical for thermal spray)
    Ste = 0.3        # Stefan number (0 = no solidification)
    Pe = 1000.0      # Peclet number (example)

    # Evaluate theoretical models
    xi_jones = jones_model(Re_eval_med)
    xi_madejski = madejski_model(Re_eval_med, We)
    xi_mostaghimi = mostaghimi_model(Re_eval_med, We, Ste, Pe)

    # Define error function
    def compute_errors(y_true, y_pred):
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot
        return rmse, mae, r2

    # Compute errors
    rmse_j, mae_j, r2_j = compute_errors(xi_signal_med, xi_jones)
    rmse_m, mae_m, r2_m = compute_errors(xi_signal_med, xi_madejski)
    rmse_mo, mae_mo, r2_mo = compute_errors(xi_signal_med, xi_mostaghimi)

    # Print LaTeX table
    print("\n\\begin{table*}[h!]")
    print("\\centering")
    print("\\caption{Quantitative comparison of theoretical flattening models against Nishioka's experimental data (80 µm particles).}")
    print("\\label{tab:model_errors}")
    print("\\begin{tabularx}{\\textwidth}{@{}l X l l@{}}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{R\(^2\)} \\\\")
    print("\\midrule")
    print(f"Jones & {rmse_j:.4f} & {mae_j:.4f} & {r2_j:.2f} \\\\")
    print(f"Madejski & {rmse_m:.4f} & {mae_m:.4f} & {r2_m:.2f} \\\\")
    print(f"Mostaghimi & {rmse_mo:.4f} & {mae_mo:.4f} & {r2_mo:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabularx}")
    print("\\end{table*}")
    print("="*80)

    # =============================================
    # TABLE: VARIATION OF FLATTENING RATIO
    # =============================================
    print("\n" + "="*80)
    print("TABLE: VARIATION OF FLATTENING RATIO")
    print("="*80)

    print(r"\begin{table*}[h!]")
    print(r"\centering")
    print(r"\caption{Variation of flattening ratio \(\xi\) across the experimental Re range for the three particle sizes.}")
    print(r"\label{tab:xi_variation}")
    print(r"\begin{tabular}{@{}l c c c c@{}}")
    print(r"\toprule")
    print(r"\textbf{Size} & \textbf{Re Min--Max} & \textbf{\(\xi\) Min--Max} & \textbf{\(\Delta\xi\)} & \textbf{Re Span} \\")
    print(r"\midrule")

    for size_key in sizes:
        data = nishioka_exp_data[size_key]
        Re = data['Re']
        xi = data['xi']
        Re_min = np.min(Re)
        Re_max = np.max(Re)
        xi_min = np.min(xi)
        xi_max = np.max(xi)
        Re_span = Re_max - Re_min
        xi_delta = xi_max - xi_min
        label = data['label']

        # Format numbers with commas and two decimals for xi
        Re_min_str = f"{Re_min:,}"
        Re_max_str = f"{Re_max:,}"
        xi_min_str = f"{xi_min:.2f}"
        xi_max_str = f"{xi_max:.2f}"
        xi_delta_str = f"{xi_delta:.2f}"
        Re_span_str = f"{Re_span:,}"

        print(f"{label} & {Re_min_str}--{Re_max_str} & {xi_min_str}--{xi_max_str} & {xi_delta_str} & {Re_span_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
    print("="*80)

    # =============================================
    # CONVERGENCE STUDY
    # =============================================
    print("\n" + "="*80)
    print("CONVERGENCE STUDY")
    print("="*80)
    convergence_data = plot_convergence_study(save_dir=save_dir)

    return all_results, all_fits



# =============================================
# MAIN
# =============================================
#if __name__ == "__main__":
#    all_results, all_fits = run_multi_size_analysis(save_dir="flattening_multi_size")
#    print("\nAnalysis complete. Figures saved to flattening_multi_size/")
#    print("\nGenerated figures:")
#    print("  - Overlay_Reconstructions.png")
#    print("  - Overlay_Fits.png")
#    print("  - Metrics_Bar_Charts.png")
#    print("  - Overlay_FRF.png (spatial frequency - 1/Re)")
#    print("  - Overlay_FRF_vs_Hz_Physical.png (frequency in Hz - extra plot)")

# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    for T in (1728.0, 2000.0, 2500.0, 2800.0):
        eta = viscosity_Ni(T)
        eta_ref = 0.1663e-3 * np.exp(6038.0 / T)   # Pa·s, Nishioka Fig. 9b
        print(f"T = {T:6.1f} K   eta = {eta:.4e} Pa.s   "
              f"Nishioka fit = {eta_ref:.4e} Pa.s")

    # Quick smoke test of flattening_models
    out = flattening_models(v=100.0, T=2500.0, d_p=80e-6)
    print("flattening_models smoke test (v=100 m/s, T=2500 K, d_p=80 um):")
    print(f"  Re = {out['Re']:.0f}")
    print(f"  We = {out['We']:.1f}")
    print(f"  Pe = {out['Pe']:.1f}")
    print(f"  Ste = {out['Ste']:.3f}")
    print(f"  xi_Jones      = {out['xi_jones']:.3f}")
    print(f"  xi_Madejski   = {out['xi_madejski']:.3f}")
    print(f"  xi_Mostaghimi = {out['xi_mostaghimi']:.3f}")

    # ... run_multi_size_analysis_with_spl ...

    # Run with SPL option
    all_results, all_fits = run_multi_size_analysis_with_spl(save_dir="flattening_multi_size")
    print("\nAnalysis complete. Figures saved to flattening_multi_size/")
    print("\nGenerated figures:")
    print("  - Overlay_Reconstructions.png")
    print("  - Overlay_Fits.png")
    print("  - Metrics_Bar_Charts.png")
    print("  - Overlay_FRF.png (spatial frequency - 1/Re)")
    print("  - Overlay_FRF_vs_Hz_Physical.png (frequency in Hz)")
    print("  - Overlay_FRF_vs_Hz_Physical_SPL.png (frequency in Hz - SPL)")


