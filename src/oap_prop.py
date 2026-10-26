"""
oap_prop.py  --  OAP_propagation stage (ideal parent paraboloid, vectorial).

Design choices, stated plainly:

* Follows the Debye-Wolf structure of the library's `focus_richards_wolf`
  (pupil -> apodised angular spectrum -> transform to focal plane), NOT the
  spec's scattered-Huygens pipeline.

* Resampling is done as a PULL-BACK: a regular (kv,kw) grid is inverted
  analytically to the pupil, and the SMOOTH input field is interpolated there.
  This avoids scattered interpolation of a phase-carrying field. It is valid for
  the ideal parent; the transformed/tilted off-centre segment is future work.

* The output transform
        E(v,w) = sum_kv sum_kw  P(kv,kw) exp(+i (kv v + kw w)) dkv dkw
  is implemented as an EXACT separable matrix-DFT. Reasons: (1) it is O(N^3),
  not O(N^4) -- acceptable for a first, transparent, provable version; (2) its
  sign is exp(+i k.r) by construction, matching the Debye convergent-wave
  convention, and is checked against a brute-force 4-D sum in the tests.
  Swapping in the library Bluestein CZT for O(N^2 log N) is a later optimisation
  and requires reconciling that routine's internal sign -- deliberately not done
  blind here.

Grid / metadata conventions are taken verbatim from the library:
    size = (size_x, size_y);  pix = (pix_x, pix_y);  field.shape = (pix_y, pix_x)
    x[n] = (n - N/2) * (size/N)
"""

from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import oap_core as core


# --------------------------------------------------------------------------- #
# Minimal, convention-faithful optical_field (the real library needs numba +
# diffractio, unavailable here; this reproduces the constructor, attributes,
# grid convention and axis metadata used by OAP_propagation's outputs).
# --------------------------------------------------------------------------- #
def _as_size_tuple(size):
    if np.isscalar(size):
        return (float(size), float(size))
    return (float(size[0]), float(size[1]))


def _as_pix_tuple(pix):
    if np.isscalar(pix):
        return (int(pix), int(pix))
    return (int(pix[0]), int(pix[1]))


def _xy_grid(size, pix):
    (sx, sy), (nx, ny) = _as_size_tuple(size), _as_pix_tuple(pix)
    dx, dy = sx / nx, sy / ny
    x = (np.arange(nx) - nx / 2) * dx
    y = (np.arange(ny) - ny / 2) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y, dx, dy


class optical_field:
    """Faithful subset of the library class + OAP axis metadata."""
    def __init__(self, size, pix, wavelength):
        self.size = _as_size_tuple(size)
        self.pix = _as_pix_tuple(pix)
        self.wavelength = float(wavelength)
        nx, ny = self.pix
        self.field = np.zeros((ny, nx), dtype=np.complex128)
        # OAP metadata: transverse spatial axes and nominal propagation axis
        self.transverse_axes = ("v", "w")
        self.propagation_axis = "u"

    def xy_grid(self):
        return _xy_grid(self.size, self.pix)


# --------------------------------------------------------------------------- #
# Pull-back resample: regular (kv,kw) grid  ->  pupil (xp,yp)  ->  input field
# --------------------------------------------------------------------------- #
def resample_to_kspace(E_in, x_in, y_in, KV, KW, k, f_parent):
    """
    For each (kv,kw) sample return the interpolated complex input field at the
    corresponding parent-pupil point, plus geometry needed downstream.

    Azimuth convention: the reflected transverse-k of an axial (-z) wave points
    radially inward, so pupil azimuth = atan2(-kw,-kv) (see oap_core).
    """
    q = np.sqrt((KV / k) ** 2 + (KW / k) ** 2)         # = sin(theta)
    inside = q <= 1.0
    q_c = np.clip(q, 0.0, 1.0)
    cos_t = np.sqrt(1.0 - q_c ** 2)
    r_p = 2.0 * f_parent * q_c / (1.0 + cos_t)          # = 2 f tan(theta/2)
    phi = np.arctan2(-KW, -KV)                           # inward correction
    xp = r_p * np.cos(phi)
    yp = r_p * np.sin(phi)

    interp = RegularGridInterpolator(
        (y_in, x_in), E_in, bounds_error=False, fill_value=0.0)
    E_samp = interp(np.stack([yp.ravel(), xp.ravel()], axis=-1)).reshape(KV.shape)

    return dict(xp=xp, yp=yp, cos_t=cos_t, sin_t=q_c, inside=inside, E_samp=E_samp)


# --------------------------------------------------------------------------- #
# Exact separable Debye transform:  E(v,w) = SUM P exp(+i(kv v + kw w)) dkv dkw
# --------------------------------------------------------------------------- #
def _debye_transform(P, kv_axis, kw_axis, v_axis, w_axis):
    """
    P has shape (Nkv, Nkw). Returns E with shape (Nw, Nv) to match the library's
    (pix_y, pix_x) = (w, v) ordering. Kernel exp(+i k.r), spacing folded in.
    """
    dkv = kv_axis[1] - kv_axis[0]
    dkw = kw_axis[1] - kw_axis[0]
    Mv = np.exp(1j * np.outer(v_axis, kv_axis))         # (Nv, Nkv)
    Mw = np.exp(1j * np.outer(w_axis, kw_axis))         # (Nw, Nkw)
    # E[v,w] = sum_{a,b} Mv[v,a] P[a,b] Mw[w,b]
    A = Mv @ P                                           # (Nv, Nkw)
    E = A @ Mw.T                                         # (Nv, Nw)
    E *= dkv * dkw
    return E.T                                           # -> (Nw, Nv) = (pix_y,pix_x)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def OAP_propagation(field, oap, output_size, output_shape,
                    output_distance=0.0, polarization=(0.0, 1.0, 0.0),
                    output_basis=None, n_k=None, return_fields=True):
    """
    Vectorial propagation of `field` through the ideal parent paraboloid of
    `oap` into a small focal-plane window.

    Parameters
    ----------
    field         : optical_field on the input/pupil plane (regular x,y grid)
    oap           : oap_core.OAP  (geometry; wavelength-independent)
    output_size   : (Lv, Lw)  focal-window size [m]
    output_shape  : (Nv, Nw)  focal-window pixels
    output_distance : delta_u along the nominal output axis [m] (axial scan/defocus)
    polarization  : incident E unit vector in the lab frame (default y_hat)
    output_basis  : (u_hat, v_hat, w_hat). Default for the ideal parent is
                    u=+z, v=+x, w=+y  (focus on the parent axis). The spec's
                    +x/90-degree case is this SAME code with a rotated basis.
    n_k           : k-grid samples per axis (default = max input pix)

    Returns
    -------
    (Ex, Ey, Ez) as optical_field objects (return_fields=True), else raw dict.
    Intensity is |Ex|^2+|Ey|^2+|Ez|^2.
    """
    lam = field.wavelength
    k = 2.0 * np.pi / lam
    f = oap.f

    if output_basis is None:
        u_hat = np.array([0.0, 0.0, 1.0])   # ideal parent axis
        v_hat = np.array([1.0, 0.0, 0.0])
        w_hat = np.array([0.0, 1.0, 0.0])
    else:
        u_hat, v_hat, w_hat = (np.asarray(a, float) for a in output_basis)

    x_in, y_in, _, _, _, _ = field.xy_grid()
    E_in = np.asarray(field.field, dtype=np.complex128)

    # angular support from the aperture
    r_max = oap.aperture / 2.0
    theta_max = 2.0 * np.arctan(r_max / (2.0 * f))
    sin_max = np.sin(theta_max)

    Nk = int(n_k) if n_k is not None else max(field.pix)
    kmax = k * sin_max * 1.02                     # small pad
    kv_axis = (np.arange(Nk) - Nk / 2) * (2 * kmax / Nk)
    kw_axis = (np.arange(Nk) - Nk / 2) * (2 * kmax / Nk)
    KV, KW = np.meshgrid(kv_axis, kw_axis, indexing="ij")   # (Nkv, Nkw)

    rs = resample_to_kspace(E_in, x_in, y_in, KV, KW, k, f)
    cos_t, sin_t, inside = rs["cos_t"], rs["sin_t"], rs["inside"]
    aperture_mask = inside & (sin_t <= sin_max)
    E_samp = rs["E_samp"] * aperture_mask

    # geometry at the pupil points (uses oap_core; supports rigid transform)
    n_par = core.paraboloid_normal(rs["xp"], rs["yp"], f)
    n_lab = core.transform_normal(n_par, oap.R)
    e_r = core.reflect_direction(oap.e_i, n_lab)

    # incident vector field (transverse to e_i for the default axial case)
    pol = np.asarray(polarization, float)
    pol = pol / np.linalg.norm(pol)
    E_i_vec = E_samp[..., None] * pol
    E_r_vec = core.reflect_vector_field(E_i_vec, n_lab)

    # physical apodisation and k-space Jacobian, kept SEPARATE (spec) then multiplied
    apod = core.physical_apodisation(cos_t)       # sec^2(theta/2)
    jac = np.where(aperture_mask, 1.0 / np.maximum(cos_t, 1e-12), 0.0)
    weight = (apod * jac)[..., None]

    # OPD: constant for the ideal axial plane wave -> global phase, dropped.
    # (Kept as a hook: multiply E_r_vec by exp(i*phi_OPD) once tilt/decenter added.)

    P_vec = E_r_vec * weight * aperture_mask[..., None]

    # axial scan / defocus along u:  multiply angular spectrum by exp(i ku delta_u)
    if output_distance != 0.0:
        ku = k * cos_t
        P_vec = P_vec * np.exp(1j * ku * output_distance)[..., None]

    # output grid
    Lv, Lw = _as_size_tuple(output_size)
    Nv, Nw = _as_pix_tuple(output_shape)
    v_axis = (np.arange(Nv) - Nv / 2) * (Lv / Nv)
    w_axis = (np.arange(Nw) - Nw / 2) * (Lw / Nw)

    comps = {}
    for i, name in enumerate(("x", "y", "z")):
        comps[name] = _debye_transform(P_vec[..., i], kv_axis, kw_axis, v_axis, w_axis)

    if not return_fields:
        return dict(Ex=comps["x"], Ey=comps["y"], Ez=comps["z"],
                    kv_axis=kv_axis, kw_axis=kw_axis, P_vec=P_vec,
                    cos_t=cos_t, aperture_mask=aperture_mask, E_samp=E_samp,
                    v_axis=v_axis, w_axis=w_axis)

    def wrap(F):
        o = optical_field(size=(Lv, Lw), pix=(Nv, Nw), wavelength=lam)
        o.field = np.asarray(F, dtype=np.complex128)
        o.propagation_axis = tuple(u_hat)
        o.transverse_axes = (tuple(v_hat), tuple(w_hat))
        return o

    return wrap(comps["x"]), wrap(comps["y"]), wrap(comps["z"])


def intensity(Ex, Ey, Ez):
    def arr(o):
        return o.field if hasattr(o, "field") else o
    return np.abs(arr(Ex)) ** 2 + np.abs(arr(Ey)) ** 2 + np.abs(arr(Ez)) ** 2
