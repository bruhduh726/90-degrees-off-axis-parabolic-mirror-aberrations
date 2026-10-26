"""
oap_tilt.py -- forward, exact-nonuniform OAP propagation that correctly handles
rigid-body tilt/decenter (the pull-back in oap_prop is only valid for the
aligned parent).

Method:  each pupil point emits one plane wave in its ACTUAL reflected direction
e_r (from the rotated normal), carrying the reflected vector field, the intrinsic
parabola apodisation, and the ACTUAL optical-path phase (constant for the aligned
parent, an aberration when tilted). The focal field is the exact Debye sum

    E(v,w) = sum_pupil  a(pupil) * exp(+i (kv v + kw w))

evaluated as two dense matmuls (a nonuniform DFT), so there is no scattered
interpolation and the sign is fixed by construction. O(N_pupil * N_out).

Output basis: by default tracked to the tilted CHIEF ray, so the ~2*alpha beam
walk-off (millimetres) is removed and the residual aberration is visible in a
micron window. The bulk pointing shift is real but is pure tip/tilt; recentering
is the standard way to inspect the aberration it leaves behind.
"""
from __future__ import annotations
import numpy as np
import oap_core as core
import oap_prop as prop


def _orthobasis(u):
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    ref = np.array([0., 0., 1.]) if abs(u[2]) < 0.9 else np.array([1., 0., 0.])
    v = ref - np.dot(ref, u) * u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return u, v, w


def OAP_propagation_direct(field, oap, output_size, output_shape,
                           polarization=(0., 1., 0.), recenter=True,
                           output_basis=None, z_ref=0.05):
    lam = field.wavelength
    k = 2 * np.pi / lam
    f = oap.f
    e_i = oap.e_i

    x, y, X, Y, dx, dy = field.xy_grid()
    r_max = oap.aperture / 2.0
    rho2 = X**2 + Y**2
    m = rho2 <= r_max**2
    xp, yp = X[m], Y[m]
    E_i = field.field[m]
    rho2m = rho2[m]
    dA = dx * dy

    # 3-D geometry of every illuminated pupil point
    r_par = core.paraboloid_point(xp, yp, f)
    n_par = core.paraboloid_normal(xp, yp, f)
    r_m = core.rigid_transform(r_par, oap.R, oap.t)
    n_lab = core.transform_normal(n_par, oap.R)
    e_r = core.reflect_direction(e_i, n_lab)                    # (Np, 3)

    # chief ray (pupil centre)
    n_ch = core.transform_normal(core.paraboloid_normal(0., 0., f), oap.R)
    e_r_ch = core.reflect_direction(e_i, n_ch)

    # ACTUAL focus of the tilted-mirror + fixed-input system: the point of
    # closest approach to all reflected ray lines (least squares). Tilting the
    # mirror is NOT a rigid rotation of the focus -- an off-axis-illuminated
    # parabola has no stigmatic focus, so we take the best geometric crossing.
    #   minimise sum_i || (I - e e^T)(P - r_m) ||^2
    P = np.eye(3) - e_r[:, :, None] * e_r[:, None, :]          # (Np,3,3)
    A = P.sum(axis=0)
    b = np.einsum('nij,nj->i', P, r_m)
    F = np.linalg.solve(A, b)

    if output_basis is not None:
        u_hat, v_hat, w_hat = (np.asarray(a, float) for a in output_basis)
    elif recenter:
        u_hat, v_hat, w_hat = _orthobasis(e_r.mean(axis=0))    # bundle mean dir
    else:
        u_hat = np.array([0., 0., 1.]); v_hat = np.array([1., 0., 0.]); w_hat = np.array([0., 1., 0.])

    kv = k * (e_r @ v_hat)
    kw = k * (e_r @ w_hat)

    # reflected vector field
    pol = np.asarray(polarization, float)
    pol = pol / np.linalg.norm(pol)
    E_r_vec = core.reflect_vector_field(E_i[:, None] * pol, n_lab)

    # intrinsic parabola apodisation cos^2(theta_p/2) = (1+cos theta_p)/2,
    # cos theta_p depends only on pupil radius (rotation-invariant).
    cos_tp = (4 * f**2 - rho2m) / (4 * f**2 + rho2m)
    Wt = 0.5 * (1.0 + cos_tp) * dA

    # actual optical-path phase (aberration). L = L_in + e_r.(F - r_m).
    # Constant for the aligned parent -> global phase; varies when tilted -> coma.
    L_in = np.sum((r_m - np.array([0., 0., z_ref])) * e_i, axis=1)
    L_out = np.sum(e_r * (F - r_m), axis=1)
    L = L_in + L_out
    phi = k * (L - L.mean())

    amp = Wt * np.exp(1j * phi)
    a = E_r_vec * amp[:, None]                                 # (Np, 3)

    Lv, Lw = prop._as_size_tuple(output_size)
    Nv, Nw = prop._as_pix_tuple(output_shape)
    v_ax = (np.arange(Nv) - Nv / 2) * (Lv / Nv)
    w_ax = (np.arange(Nw) - Nw / 2) * (Lw / Nw)

    Mv = np.exp(1j * np.outer(v_ax, kv))                       # (Nv, Np)
    Mw = np.exp(1j * np.outer(w_ax, kw))                       # (Nw, Np)

    def trans(col):
        return ((Mv * col[None, :]) @ Mw.T).T                 # (Nw, Nv)

    return dict(Ex=trans(a[:, 0]), Ey=trans(a[:, 1]), Ez=trans(a[:, 2]),
                kv=kv, kw=kw, u_hat=u_hat, v_hat=v_hat, w_hat=w_hat,
                aberr_ptv_waves=(phi.max() - phi.min()) / (2 * np.pi),
                v_ax=v_ax, w_ax=w_ax)
