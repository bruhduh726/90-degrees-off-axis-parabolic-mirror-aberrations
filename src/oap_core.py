"""
oap_core.py  --  convention-INDEPENDENT physics core for the OAP propagator.

This file deliberately stops BEFORE the CZT. Everything here (surface, normals,
rigid transform, ray/field reflection, optical path, apodisation, analytic
k-mapping) is independent of any FFT/CZT sign or scaling convention, so it can be
unit-tested on its own and later wired to the host library's CZT unchanged.

Coordinate model (lab frame, fixed):
    parent paraboloid   z_p = (x_p^2 + y_p^2) / (4 f_parent)
    rigid transform     r_lab = R @ r_parent + t
    normal transform    n_lab = R @ n_parent
Nothing here hard-codes the output axis; pass an output basis (u,v,w) explicitly.
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------- 
# Parent paraboloid surface + normal
# -----------------------------------------------------------------------------
def paraboloid_surface(xp, yp, f_parent):
    """z_p on the parent paraboloid. Note: uses f_parent (7.5 mm), never RFL."""
    return (xp**2 + yp**2) / (4.0 * f_parent)


def paraboloid_point(xp, yp, f_parent):
    """Stack (xp, yp, zp) along last axis."""
    zp = paraboloid_surface(xp, yp, f_parent)
    return np.stack(np.broadcast_arrays(xp, yp, zp), axis=-1)


def paraboloid_normal(xp, yp, f_parent):
    """
    Unit outward normal of F = z - (x^2+y^2)/(4f) = 0, i.e.
        n ~ (-xp/(2f), -yp/(2f), 1) / sqrt(1 + (xp^2+yp^2)/(4 f^2)).
    """
    nx = -xp / (2.0 * f_parent)
    ny = -yp / (2.0 * f_parent)
    nz = np.ones_like(np.asarray(nx, dtype=float))
    n = np.stack(np.broadcast_arrays(nx, ny, nz), axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


# ----------------------------------------------------------------------------- 
# Rigid-body transform (this is the ONLY misalignment mechanism; no Zernikes)
# -----------------------------------------------------------------------------
def rigid_transform(r_parent, R, t):
    """r_lab = R @ r_parent + t.  r_parent shape (..., 3)."""
    return r_parent @ R.T + t


def transform_normal(n_parent, R):
    """n_lab = R @ n_parent. Pure rotation -> stays unit; no re-normalise needed."""
    return n_parent @ R.T


def rotation_matrix(axis, angle):
    """Right-handed rotation by `angle` (rad) about unit `axis` (Rodrigues)."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ----------------------------------------------------------------------------- 
# Reflection of ray direction and of the vector E-field
# -----------------------------------------------------------------------------
def reflect_direction(e_i, n):
    """e_r = e_i - 2 (e_i . n) n.  e_i may be (3,) or (...,3); n is (...,3)."""
    e_i = np.asarray(e_i, float)
    dot = np.sum(e_i * n, axis=-1, keepdims=True)
    return e_i - 2.0 * dot * n


def reflect_vector_field(E_i, n):
    """
    Ideal-conductor reflection: E_r = -E_i + 2 (E_i . n) n.
    Kept isolated so it can be swapped for a Fresnel s/p model later.
    Preserves transversality: E_r . e_r = -(E_i . e_i)  (see tests).
    """
    dot = np.sum(E_i * n, axis=-1, keepdims=True)
    return -E_i + 2.0 * dot * n


# ----------------------------------------------------------------------------- 
# Optical path (kept as actual geometry, so tilt/decenter give real OPD)
# -----------------------------------------------------------------------------
def optical_path_to_point(r_mirror, focus, e_i, ref_point):
    """
    Path length: from the reference plane (perpendicular to e_i, containing
    `ref_point`) along e_i to the mirror, then straight to `focus`.
    L_in = (r_mirror - ref_point) . e_i    [distance travelled along e_i]
    For the ideal axial case (e_i=-z, ref_point=(0,0,z0)) this is z0 - zp, and
    L_in + |focus - r_mirror| = z0 + f = CONSTANT.
    """
    e_i = np.asarray(e_i, float)
    e_i = e_i / np.linalg.norm(e_i)
    ref_point = np.asarray(ref_point, float)
    L_in = np.sum((r_mirror - ref_point) * e_i, axis=-1)
    L_out = np.linalg.norm(focus - r_mirror, axis=-1)
    return L_in + L_out


# ----------------------------------------------------------------------------- 
# Apodisation and Jacobian  --  DELIBERATELY SEPARATE (do not merge)
# -----------------------------------------------------------------------------
def physical_apodisation(cos_theta):
    """Parabolic amplitude apodisation q = 2/(1+cos theta) = sec^2(theta/2)."""
    return 2.0 / (1.0 + cos_theta)


def angular_to_kspace_jacobian(k, ku):
    """dOmega = dk_v dk_w / (k * ku).  Returns the multiplicative factor 1/(k ku)."""
    return 1.0 / (k * ku)


# ----------------------------------------------------------------------------- 
# Analytic k <-> parent-radius map (ideal parent only; use pull-back, see notes)
# -----------------------------------------------------------------------------
def kspace_to_parent(kv, kw, k, f_parent):
    """
    Inverse map for the IDEAL parent about its own axis:
        q = sqrt((kv/k)^2 + (kw/k)^2),  r = 2 f q / (1 + sqrt(1-q^2)),
        xp = r cos(phi), yp = r sin(phi),  phi = atan2(kw, kv).
    This is the well-conditioned direction (regular k-grid -> mirror).

    IMPORTANT azimuth convention: for an axial (-z) plane wave on the parent, the
    reflected transverse wavevector points radially INWARD, so its azimuth is
    (phi_mirror + pi). The naive `phi = atan2(kw, kv)` from the spec therefore
    returns a POINT-INVERTED parent coordinate (flips vortex charge). We undo
    that flip here so forward/inverse are true inverses.
    """
    q = np.sqrt((kv / k) ** 2 + (kw / k) ** 2)
    r = 2.0 * f_parent * q / (1.0 + np.sqrt(1.0 - q**2))
    phi = np.arctan2(-kw, -kv)          # inward -> undo the pi azimuth flip
    return r * np.cos(phi), r * np.sin(phi)


def parent_to_kspace(xp, yp, k, f_parent):
    """
    Forward map for the ideal parent about +z (u_hat = +z):
    reflect an axial plane wave and read off (kv,kw)=(kx,ky)->(kv,kw). Provided
    mainly to check consistency against kspace_to_parent.
    """
    n = paraboloid_normal(xp, yp, f_parent)
    e_r = reflect_direction((0.0, 0.0, -1.0), n)
    kv = k * e_r[..., 0]
    kw = k * e_r[..., 1]
    ku = k * e_r[..., 2]
    return kv, kw, ku


# ----------------------------------------------------------------------------- 
# OAP object: caches everything geometric (wavelength-independent geometry)
# -----------------------------------------------------------------------------
class OAP:
    """
    Geometry container. Wavelength is intentionally NOT stored here; it enters
    only when building k-vectors (multiply e_r by k) at propagation time.
    """
    def __init__(self, parent_focal_length, aperture, R=None, t=None,
                 incident_direction=(0.0, 0.0, -1.0)):
        self.f = float(parent_focal_length)
        self.aperture = float(aperture)          # circular clear aperture diameter
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.t = np.zeros(3) if t is None else np.asarray(t, float)
        e = np.asarray(incident_direction, float)
        self.e_i = e / np.linalg.norm(e)

    # ideal parent focus, transformed into the lab frame
    def focus_lab(self):
        return rigid_transform(np.array([0.0, 0.0, self.f]), self.R, self.t)

    def prepare_geometry(self, xp, yp, output_basis):
        """
        Precompute the wavelength-independent geometry on a parent (xp,yp) grid.
        `output_basis` = (u_hat, v_hat, w_hat). Returns a dict of cached arrays.
        k-vectors here are returned as DIRECTION components (e_r); multiply by k
        at propagation time. Aperture mask is circular on the parent grid.
        """
        u_hat, v_hat, w_hat = (np.asarray(a, float) for a in output_basis)

        r_par = paraboloid_point(xp, yp, self.f)
        n_par = paraboloid_normal(xp, yp, self.f)
        r_lab = rigid_transform(r_par, self.R, self.t)
        n_lab = transform_normal(n_par, self.R)
        e_r = reflect_direction(self.e_i, n_lab)

        # direction components on the output basis
        eu = e_r @ u_hat
        ev = e_r @ v_hat
        ew = e_r @ w_hat
        cos_theta = eu                      # cos(theta) w.r.t. u_hat, since |e_r|=1
        mask = (xp**2 + yp**2) <= (self.aperture / 2.0) ** 2

        return dict(r_lab=r_lab, n_lab=n_lab, e_r=e_r,
                    eu=eu, ev=ev, ew=ew, cos_theta=cos_theta,
                    apod=physical_apodisation(cos_theta), mask=mask)
