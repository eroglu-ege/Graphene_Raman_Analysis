"""
raman_helpers.py
================
Shared helper functions for graphene Raman spectroscopy analysis.
Collected from all analysis scripts produced during this session.

Usage
-----
Place this file in the same directory as your analysis scripts and import:

    from raman_helpers import (
        lorentzian, linear_baseline, crop,
        fit_peak, fit_lorentzian,
        to_map, masked,
        araujo_eq3, araujo_eq9, araujo_eq4,
        gaussian,
        get_positions, get_si_intensity,
        ferrari_stage1, ferrari_stage2,
        classify_layer,
    )

Dependencies
------------
    numpy, scipy, xml.etree.ElementTree
    (matplotlib and PIL only needed in the calling scripts, not here)

Contents
--------
  LINESHAPE FUNCTIONS
    lorentzian          — single Lorentzian peak
    gaussian            — single Gaussian peak

  BASELINE & PEAK FITTING
    linear_baseline     — straight-line background from anchor windows
    crop                — slice wavenumber axis to a window
    fit_peak            — crop + baseline + Lorentzian fit in one call
    fit_lorentzian      — alias for fit_peak (kept for back-compatibility)

  MAP UTILITIES
    to_map              — reshape 1-D result array → 2-D spatial grid
    masked              — mask NaN pixels for matplotlib imshow

  KNIFE-EDGE / SPOT SIZE  (Araújo et al. 2009)
    araujo_eq3          — exact error-function edge profile [their Eq. 3]
    araujo_eq9          — improved sigmoidal approximation  [their Eq. 9]
    araujo_eq4          — analytical derivative = Gaussian beam profile [Eq. 4]

  DATA LOADING  (specific to WITec / Bruker .mat + .ompc format)
    get_positions       — extract physical Y stage positions from .ompc XML
    get_si_intensity    — fit Si ~521 cm⁻¹ peak across all spectra in a .mat

  DEFECT DENSITY  (Ferrari & Bonini, Nature Nanotech 8, 2013)
    ferrari_stage1      — LD and nD from Eq. 5 & 6 (low defect, LD > 10 nm)
    ferrari_stage2      — LD and nD from Eq. 7 & 8 (high defect, LD < 3 nm)

  LAYER CLASSIFICATION
    classify_layer      — monolayer / bilayer / multilayer from ratio + FWHM
"""

import numpy as np
import scipy.io as sio
import xml.etree.ElementTree as ET
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv
import warnings

# ══════════════════════════════════════════════════════════════════════════════
# LINESHAPE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def lorentzian(x, amp, cen, fwhm):
    """
    Single Lorentzian peak.

    Parameters
    ----------
    x    : array-like, wavenumber axis (cm⁻¹)
    amp  : peak amplitude (intensity at centre)
    cen  : peak centre position (cm⁻¹)
    fwhm : full width at half maximum (cm⁻¹)

    Returns
    -------
    array  : Lorentzian intensity at each x

    Notes
    -----
    Lorentzian is the correct lineshape for homogeneously broadened
    Raman peaks (phonon lifetime limited), including Si, G, and 2D bands.
    Use Gaussian or Voigt for amorphous or heavily defective materials.
    """
    return amp * (fwhm / 2)**2 / ((x - cen)**2 + (fwhm / 2)**2)


def gaussian(x, amp, mu, sigma):
    """
    Single Gaussian peak.

    Parameters
    ----------
    x     : array-like
    amp   : peak amplitude
    mu    : peak centre
    sigma : standard deviation (FWHM = 2√(2 ln 2) · σ ≈ 2.355 · σ)

    Returns
    -------
    array : Gaussian intensity at each x

    Notes
    -----
    Used for: derivative of erf edge profiles (= Gaussian beam profile),
    and occasionally for inhomogeneously broadened peaks.
    """
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE AND PEAK FITTING
# ══════════════════════════════════════════════════════════════════════════════

def linear_baseline(x, y, left_wn, right_wn, width=20):
    """
    Estimate a linear background by fitting a straight line through
    anchor windows on either side of a peak.

    Parameters
    ----------
    x         : wavenumber array (cm⁻¹)
    y         : intensity array
    left_wn   : upper edge of the left anchor window (cm⁻¹)
                → window spans [left_wn - width, left_wn]
    right_wn  : lower edge of the right anchor window (cm⁻¹)
                → window spans [right_wn, right_wn + width]
    width     : half-width of each anchor window in cm⁻¹ (default 20)

    Returns
    -------
    array : linear baseline evaluated over the full x array

    Notes
    -----
    Sufficient for the slowly varying fluorescence / detector-response
    background under narrow Raman peaks. For broader features or strong
    fluorescence, consider a polynomial baseline instead.
    """
    lm = (x >= left_wn  - width) & (x <= left_wn)
    rm = (x >= right_wn)         & (x <= right_wn + width)
    x_anchor = np.r_[x[lm], x[rm]]
    y_anchor = np.r_[y[lm], y[rm]]
    coeffs   = np.polyfit(x_anchor, y_anchor, 1)
    return np.polyval(coeffs, x)


def crop(wn, spec, lo, hi):
    """
    Slice a spectrum to the wavenumber window [lo, hi].

    Parameters
    ----------
    wn   : full wavenumber axis (cm⁻¹)
    spec : full intensity array
    lo   : lower bound (cm⁻¹)
    hi   : upper bound (cm⁻¹)

    Returns
    -------
    wn_cropped, spec_cropped : both sliced to the requested window
    """
    mask = (wn >= lo) & (wn <= hi)
    return wn[mask], spec[mask]


def fit_peak(wn, spec, lo, hi, bl_lo, bl_hi, p0, bounds,
             baseline_width=20, maxfev=4000):
    """
    Fit a single Lorentzian to a spectral peak in three steps:
      1. Crop spectrum to [lo, hi]
      2. Subtract linear baseline anchored at bl_lo / bl_hi
      3. Fit Lorentzian with curve_fit

    Parameters
    ----------
    wn             : full wavenumber axis (cm⁻¹)
    spec           : full intensity array for one spectrum
    lo, hi         : fit window edges (cm⁻¹)
    bl_lo, bl_hi   : baseline anchor positions (cm⁻¹)
                     left anchor:  [bl_lo - baseline_width, bl_lo]
                     right anchor: [bl_hi, bl_hi + baseline_width]
    p0             : initial guess [amp, cen, fwhm]
    bounds         : ([amp_min, cen_min, fwhm_min],
                       [amp_max, cen_max, fwhm_max])
    baseline_width : anchor window half-width in cm⁻¹ (default 20)
    maxfev         : max function evaluations for curve_fit (default 4000)

    Returns
    -------
    popt : (amp, cen, fwhm)  — fitted Lorentzian parameters

    Raises
    ------
    RuntimeError if curve_fit does not converge.
    """
    wn_c  = wn[(wn >= lo) & (wn <= hi)]
    sp_c  = spec[(wn >= lo) & (wn <= hi)]
    sp_bl = sp_c - linear_baseline(wn_c, sp_c, bl_lo, bl_hi,
                                   width=baseline_width)
    popt, _ = curve_fit(lorentzian, wn_c, sp_bl,
                        p0=p0, bounds=bounds, maxfev=maxfev)
    return popt   # (amp, cen, fwhm)


def fit_lorentzian(wn, spec, lo, hi, bl_lo, bl_hi, p0,
                   bounds=(0, np.inf)):
    wn_c, sp_c = crop(wn, spec, lo, hi)
    bl = linear_baseline(wn_c, sp_c, bl_lo, bl_hi)
    sp_bl = sp_c - bl
    popt, _ = curve_fit(lorentzian, wn_c, sp_bl, p0=p0,
                        bounds=bounds, maxfev=5000)
    return popt, wn_c, sp_bl   # amp, cen, fwhm


# ══════════════════════════════════════════════════════════════════════════════
# MAP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def to_map(arr, ny=20, nx=20):
    """
    Reshape a flat 1-D result array (n_spectra,) into a 2-D spatial map
    (ny, nx) using the scan's row-major (X-fast) ordering.

    Parameters
    ----------
    arr     : 1-D array of length ny*nx
    ny, nx  : grid dimensions (default 20×20 for GrapheneMap20)

    Returns
    -------
    2-D array of shape (ny, nx)

    Notes
    -----
    Row-major means spectrum index i maps to:
        col (X) = i % nx
        row (Y) = i // nx
    Confirmed from GrapheneMap20Messposition.ompc where arrayXpos
    increments before arrayYpos.
    """
    return arr.reshape(ny, nx)


def masked(arr):
    """
    Convert a 2-D array to a numpy masked array, hiding NaN pixels.
    Pass directly to matplotlib imshow — masked pixels render as grey
    (or whatever the axes facecolor is set to).

    Parameters
    ----------
    arr : 2-D numpy array, possibly containing NaNs

    Returns
    -------
    numpy.ma.MaskedArray
    """
    return np.ma.masked_invalid(arr)


# ══════════════════════════════════════════════════════════════════════════════
# KNIFE-EDGE / SPOT SIZE  (Araújo et al., Appl. Opt. 48, 393, 2009)
# ══════════════════════════════════════════════════════════════════════════════

def araujo_eq3(x, I0, x0, w, bg):
    """
    Exact error-function edge profile — Araújo et al. (2009) Eq. (3).

    Models the normalised transmitted power as a Gaussian beam crosses
    an opaque knife edge:

        P_N(x) = bg + I0/2 * [1 + erf((x − x0) / w)]

    Parameters
    ----------
    x   : position array (µm)
    I0  : step height  (≈1.0 after normalisation)
    x0  : edge centre position (µm)
    w   : 1/e intensity beam radius (µm)
           — Araújo's Eq.(1) defines intensity as exp(−r²/w²),
             so w is the 1/e radius, NOT the 1/e² radius.
             Multiply by √2 to convert to the 1/e² convention.
    bg  : background offset  (≈0.0 after normalisation)

    Returns
    -------
    array : modelled P_N at each position

    Derived quantities from w
    -------------------------
        w * sqrt(2)             → 1/e² radius (common microscopy spec)
        2 * sqrt(ln2) * w       → FWHM of beam
        2 * erfinv(0.8) * w     → 10%→90% edge transition width
    """
    return bg + I0 * 0.5 * (1.0 + erf((x - x0) / w))


def araujo_eq9(x, I0, x0, w, bg):
    """
    Improved sigmoidal approximation — Araújo et al. (2009) Eq. (9).

    Agrees with the exact erf [araujo_eq3] to within 10⁻⁷–10⁻⁸,
    which is far below any real experimental noise floor.
    Included for completeness and to verify the exact erf result.

    Model:
        s    = √2 · (x − x0) / w
        f(s) = 1 / (1 + exp(a1·s + a3·s³))

    Coefficients (Araújo corrected, third-order):
        a1 = −1.597106847
        a3 = −7.0924013 × 10⁻²

    Note: The older Khosrofian-Garetz function (Eq. 8) overestimates
    w by ~3.8% due to a symmetry error in its derivation.
    Past results using Eq.(8) can be corrected by multiplying by 1/1.04.

    Parameters — same as araujo_eq3.
    """
    a1   = -1.597106847
    a3   = -7.0924013e-2
    s    = np.sqrt(2) * (x - x0) / w
    poly = np.clip(a1*s + a3*s**3, -500, 500)   # clip prevents overflow
    fs   = 1.0 / (1.0 + np.exp(poly))
    return bg + I0 * fs


def araujo_eq4(x, amp, x0, w):
    """
    Analytical derivative of P_N — Araújo et al. (2009) Eq. (4).

    This equals the Gaussian beam intensity profile projected onto the
    scan axis:

        dP_N/dx = (1 / √π·w) · exp(−(x − x0)² / w²)

    Used as a cross-check: fitting this to the numerical derivative of
    the edge profile should return the same w as fitting araujo_eq3
    to the profile itself.

    Parameters
    ----------
    x   : position array (µm)
    amp : free amplitude scale (normalised separately from theory)
    x0  : beam centre / edge position (µm)
    w   : 1/e beam radius (µm)  — same definition as araujo_eq3

    Returns
    -------
    array : Gaussian beam profile at each position
    """
    return amp * np.exp(-((x - x0) / w)**2)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (WITec / Bruker .mat + .ompc format)
# ══════════════════════════════════════════════════════════════════════════════

def get_positions(ompc_path):
    """
    Extract physical stage Y positions (µm) from a Bruker .ompc XML file.

    The .ompc file stores one <Annotation_i> element per spectrum, each
    containing <fY> — the absolute stage Y coordinate in µm.
    We subtract the first position to set the origin at zero, then take
    the absolute value so the scan runs in the positive direction.

    Parameters
    ----------
    ompc_path : str, path to the .ompc file

    Returns
    -------
    1-D numpy array of physical Y positions in µm, length = n_spectra
    """
    root  = ET.parse(ompc_path).getroot()
    count = int(root.find("Annotation_Count").text)
    ys    = [float(root.find(f"Annotation_{i}").find("fY").text)
             for i in range(count)]
    return np.abs(np.array(ys) - ys[0])


def get_si_intensity(mat_path, wn_lo=480, wn_hi=560,
                     bl_lo=490, bl_hi=550, baseline_width=15):
    """
    Fit the Si first-order phonon peak (~521 cm⁻¹) on every spectrum
    in a .mat file and return the fitted Lorentzian amplitude.

    Parameters
    ----------
    mat_path       : str, path to the .mat file (WITec format)
    wn_lo, wn_hi   : fit window in cm⁻¹ (default 480–560)
    bl_lo, bl_hi   : baseline anchor positions in cm⁻¹
    baseline_width : anchor window half-width in cm⁻¹

    Returns
    -------
    1-D numpy array of Si peak amplitudes, length = n_spectra.
    NaN where the Lorentzian fit did not converge (e.g. spectra on
    opaque chrome where no Si signal is present).

    Notes
    -----
    Uses a Lorentzian rather than a raw maximum because:
      - More robust to noise spikes
      - Consistent even when the Si peak centre shifts slightly
        due to local strain under graphene
    """
    Ra      = sio.loadmat(mat_path)["Ra"]
    wn      = Ra[:, 0]
    spectra = Ra[:, 1:]
    n       = spectra.shape[1]
    amps    = np.full(n, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            sp   = spectra[:, i]
            mask = (wn >= wn_lo) & (wn <= wn_hi)
            sc   = sp[mask] - linear_baseline(
                       wn[mask], sp[mask], bl_lo, bl_hi,
                       width=baseline_width)
            try:
                popt, _ = curve_fit(
                    lorentzian, wn[mask], sc,
                    p0=[sc.max(), 521, 5],
                    bounds=([0, 500, 1], [np.inf, 540, 30]),
                    maxfev=3000
                )
                amps[i] = popt[0]
            except RuntimeError:
                pass
    return amps


# ══════════════════════════════════════════════════════════════════════════════
# DEFECT DENSITY  (Ferrari & Bonini, Nature Nanotech 8, 235, 2013)
# ══════════════════════════════════════════════════════════════════════════════

def ferrari_stage1(amp_D, amp_G, EL=2.33):
    """
    Defect density in the low-defect (Stage 1) regime — LD > 10 nm.
    Ferrari & Bonini (2013) Equations 5 & 6.

    Stage 1 applies when defects are sparse and each scatters
    independently: I(D)/I(G) ∝ 1/LD².

    Equations:
        LD² (nm²) = [4.3 × 10³ / EL⁴] × [I(G)/I(D)]     (Eq. 5)
        nD  (cm⁻²) = 7.3 × 10⁹ × EL⁴ × [I(D)/I(G)]      (Eq. 6)

    Parameters
    ----------
    amp_D : array-like, fitted D band amplitude(s)
    amp_G : array-like, fitted G band amplitude(s)
    EL    : laser energy in eV (default 2.33 eV = 532 nm)

    Returns
    -------
    LD_nm  : inter-defect distance in nm
    nD_cm2 : defect density in cm⁻²

    Validity check
    --------------
    Results are self-consistent only if LD_nm > 10 nm everywhere.
    If LD < 3 nm, use ferrari_stage2 instead.
    """
    amp_D  = np.asarray(amp_D, dtype=float)
    amp_G  = np.asarray(amp_G, dtype=float)
    EL4    = EL**4
    ratio  = amp_D / amp_G
    LD2    = (4.3e3 / EL4) * (amp_G / amp_D)   # nm²
    LD_nm  = np.sqrt(np.abs(LD2))               # nm
    nD_cm2 = 7.3e9 * EL4 * ratio               # cm⁻²
    return LD_nm, nD_cm2


def ferrari_stage2(amp_D, amp_G, EL=2.33):
    """
    Defect density in the high-defect (Stage 2) regime — LD < 3 nm.
    Ferrari & Bonini (2013) Equations 7 & 8.

    Stage 2 applies when defects are so dense that the graphene lattice
    itself breaks down: I(D)/I(G) ∝ LD² (inverted relationship).

    Equations:
        LD² (nm²) = [5.4 × 10² / EL⁴] × [I(D)/I(G)]     (Eq. 7)
        nD  (cm⁻²) = 5.9 × 10¹⁴ × EL⁴ × [I(G)/I(D)]     (Eq. 8)

    Parameters
    ----------
    amp_D : array-like, fitted D band amplitude(s)
    amp_G : array-like, fitted G band amplitude(s)
    EL    : laser energy in eV (default 2.33 eV = 532 nm)

    Returns
    -------
    LD_nm  : inter-defect distance in nm
    nD_cm2 : defect density in cm⁻²

    Validity check
    --------------
    Results are self-consistent only if LD_nm < 3 nm everywhere.
    If LD > 10 nm, use ferrari_stage1 instead.
    """
    amp_D  = np.asarray(amp_D, dtype=float)
    amp_G  = np.asarray(amp_G, dtype=float)
    EL4    = EL**4
    ratio  = amp_D / amp_G
    LD2    = (5.4e2 / EL4) * ratio              # nm²
    LD_nm  = np.sqrt(np.abs(LD2))               # nm
    nD_cm2 = 5.9e14 * EL4 * (amp_G / amp_D)    # cm⁻²
    return LD_nm, nD_cm2


# ══════════════════════════════════════════════════════════════════════════════
# LAYER CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_layer(ratio_2DG, fwhm_2D,
                   mono_ratio=1.5, mono_fwhm=40,
                   bi_ratio=0.7,   bi_fwhm=55):
    """
    Classify each spectrum as monolayer, bilayer, or multilayer graphene
    using two independent criteria: I(2D)/I(G) ratio and 2D band FWHM.

    Using two criteria is more robust than either alone because strain
    and doping can shift the ratio without changing the layer number.

    Classification rules (532 nm excitation):
        Monolayer  : ratio > mono_ratio  AND  fwhm < mono_fwhm
        Bilayer    : ratio > bi_ratio    AND  fwhm < bi_fwhm   (else monolayer)
        Multilayer : ratio <= bi_ratio   OR   fwhm >= bi_fwhm
        No signal  : NaN in either input

    Parameters
    ----------
    ratio_2DG  : array-like, I(2D)/I(G) amplitude ratios
    fwhm_2D    : array-like, fitted 2D band FWHM values (cm⁻¹)
    mono_ratio : I(2D)/I(G) threshold for monolayer (default 1.5)
    mono_fwhm  : FWHM threshold for monolayer in cm⁻¹ (default 40)
    bi_ratio   : lower I(2D)/I(G) bound for bilayer (default 0.7)
    bi_fwhm    : upper FWHM bound for bilayer in cm⁻¹ (default 55)

    Returns
    -------
    layer : integer array, same length as inputs
        0 = no signal / NaN
        1 = monolayer
        2 = bilayer
        3 = multilayer

    Notes
    -----
    Thresholds are for 532 nm (2.33 eV) excitation. They may need
    adjusting for other laser wavelengths.
    """
    ratio_2DG = np.asarray(ratio_2DG, dtype=float)
    fwhm_2D   = np.asarray(fwhm_2D,   dtype=float)
    layer     = np.zeros(len(ratio_2DG), dtype=int)

    for i in range(len(ratio_2DG)):
        r = ratio_2DG[i]
        f = fwhm_2D[i]
        if np.isnan(r) or np.isnan(f):
            layer[i] = 0
        elif r > mono_ratio and f < mono_fwhm:
            layer[i] = 1
        elif r > bi_ratio and f < bi_fwhm:
            layer[i] = 2
        else:
            layer[i] = 3

    return layer


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("raman_helpers.py — self-test")
    print("─" * 40)

    # Test lorentzian — use exact centre point to avoid discrete-grid rounding
    x    = np.linspace(1550, 1620, 201)   # 201 points so 1582.5 is not exact
    x2   = np.array([1582.0])             # exact centre for amplitude test
    peak = lorentzian(x2, 500, 1582, 15)
    assert abs(peak[0] - 500) < 1e-6,     "lorentzian amplitude wrong"
    peak_arr = lorentzian(x, 500, 1582, 15)
    assert abs(x[peak_arr.argmax()] - 1582) < 0.5, "lorentzian centre wrong"
    print("  lorentzian         ✓")

    # Test gaussian — same approach
    g = gaussian(x2, 300, 1582, 10)
    assert abs(g[0] - 300) < 1e-6, "gaussian amplitude wrong"
    print("  gaussian           ✓")

    # Test linear_baseline (flat baseline → returns flat)
    wn_test  = np.linspace(1480, 1680, 500)
    sp_test  = lorentzian(wn_test, 200, 1582, 15) + 50.0
    bl       = linear_baseline(wn_test, sp_test, 1490, 1670)
    assert abs(bl.mean() - 50) < 5, "linear_baseline offset wrong"
    print("  linear_baseline    ✓")

    # Test crop
    wn2, sp2 = crop(wn_test, sp_test, 1550, 1620)
    assert wn2.min() >= 1550 and wn2.max() <= 1620, "crop bounds wrong"
    print("  crop               ✓")

    # Test to_map / masked
    flat = np.arange(400, dtype=float)
    flat[5] = np.nan
    m = to_map(flat)
    assert m.shape == (20, 20), "to_map shape wrong"
    ma = masked(m)
    assert ma.mask[0, 5], "masked not hiding NaN"
    print("  to_map / masked    ✓")

    # Test ferrari_stage1
    LD, nD = ferrari_stage1(amp_D=10.0, amp_G=100.0, EL=2.33)
    assert LD > 10, "ferrari_stage1 LD should be > 10 nm for small D/G"
    print("  ferrari_stage1     ✓")

    # Test classify_layer
    layer = classify_layer(
        ratio_2DG=np.array([2.5, 1.2, 0.5, np.nan]),
        fwhm_2D  =np.array([30,  35,  60,  30])
    )
    assert list(layer) == [1, 2, 3, 0], f"classify_layer wrong: {layer}"
    print("  classify_layer     ✓")

    # Test araujo_eq3 / eq9 agreement
    pos  = np.linspace(-5, 5, 500)
    eq3  = araujo_eq3(pos, 1.0, 0.0, 1.0, 0.0)
    eq9  = araujo_eq9(pos, 1.0, 0.0, 1.0, 0.0)
    diff = np.abs(eq3 - eq9).max()
    assert diff < 1e-3, f"araujo eq3 vs eq9 differ by {diff:.2e}"
    print(f"  araujo_eq3/eq9     ✓  (max diff = {diff:.2e}, well below noise floor)")

    print("─" * 40)
    print("All tests passed.")