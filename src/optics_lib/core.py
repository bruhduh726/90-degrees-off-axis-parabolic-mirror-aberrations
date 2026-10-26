import math
import numpy as np
import matplotlib.pyplot as plt

from numba import njit, prange
from scipy.ndimage import zoom
from scipy.special import assoc_laguerre
from diffractio.scalar_fields_XY import Scalar_field_XY
import matplotlib.gridspec as gridspec
plt.rcParams.update({'font.size': 14})

# =============================================================================
# Helpers
# =============================================================================

def _as_size_tuple(size):
    """
    Convert size to (size_x, size_y).
    Accepts scalar or iterable of length 2.
    """
    if np.isscalar(size):
        s = float(size)
        return (s, s)

    if len(size) != 2:
        raise ValueError("size must be a scalar or a tuple/list (size_x, size_y)")
    return (float(size[0]), float(size[1]))


def _as_pix_tuple(pix):
    """
    Convert pix to (pix_x, pix_y).
    Accepts scalar or iterable of length 2.
    """
    if np.isscalar(pix):
        p = int(pix)
        return (p, p)

    if len(pix) != 2:
        raise ValueError("pix must be a scalar or a tuple/list (pix_x, pix_y)")
    return (int(pix[0]), int(pix[1]))


def _shape_from_pix(pix):
    """
    Convert (pix_x, pix_y) -> array shape (pix_y, pix_x).
    """
    pix_x, pix_y = _as_pix_tuple(pix)
    return (pix_y, pix_x)


def _xy_grid_from_size_pix(size, pix):
    """
    Returns x, y, X, Y, dx, dy for rectangular field.

    size = (size_x, size_y)
    pix  = (pix_x, pix_y)

    Output:
        x.shape = (pix_x,)
        y.shape = (pix_y,)
        X.shape = Y.shape = (pix_y, pix_x)
    """
    size_x, size_y = _as_size_tuple(size)
    pix_x, pix_y = _as_pix_tuple(pix)

    dx = size_x / pix_x
    dy = size_y / pix_y

    x = (np.arange(pix_x) - pix_x / 2) * dx
    y = (np.arange(pix_y) - pix_y / 2) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y, dx, dy


def _crop_or_pad_1d(U, axis, n_target):
    """
    Center crop or zero-pad along one axis.
    """
    n = U.shape[axis]
    if n_target == n:
        return U

    if n_target < n:
        s = (n - n_target) // 2
        sl = [slice(None)] * U.ndim
        sl[axis] = slice(s, s + n_target)
        return U[tuple(sl)]

    p = n_target - n
    p0, p1 = p // 2, p - p // 2
    pad_width = [(0, 0)] * U.ndim
    pad_width[axis] = (p0, p1)
    return np.pad(U, pad_width, mode="constant")


def _resample_complex(U, zoom_y, zoom_x, order=1):
    """
    Interpolate complex array by resampling real/imag separately.
    """
    Ur = zoom(U.real, (zoom_y, zoom_x), order=order)
    Ui = zoom(U.imag, (zoom_y, zoom_x), order=order)
    return Ur + 1j * Ui


def _grid_1d(size: float, pix: int) -> np.ndarray:
    return np.linspace(-size / 2, size / 2, pix, endpoint=True)


def _normalize_rect_args(N=None, pix=None):
    """
    Support either:
        N=scalar or N=(Nx,Ny)
    or:
        pix=scalar or pix=(Nx,Ny)

    Returns (pix_x, pix_y).
    """
    arg = pix if pix is not None else N
    if arg is None:
        raise ValueError("Provide N or pix.")
    return _as_pix_tuple(arg)


# =============================================================================
# Main class
# =============================================================================

class optical_field:
    """
    Rectangular optical field.

    Conventions
    -----------
    size = (size_x, size_y) in meters
    pix  = (pix_x, pix_y)
    field.shape = (pix_y, pix_x)

    Backward compatibility
    ----------------------
    Scalars are still accepted:
        size = L     -> (L, L)
        pix  = N     -> (N, N)
    """

    def __init__(self, size, pix, wavelength):
        self.size = _as_size_tuple(size)
        self.pix = _as_pix_tuple(pix)
        self.wavelength = float(wavelength)

        pix_x, pix_y = self.pix
        self.field = np.zeros((pix_y, pix_x), dtype=np.complex128)

    def xy_grid(self):
        return _xy_grid_from_size_pix(self.size, self.pix)

    def rescale_field(self, new_size, new_pix):
        self.field = rescale(
            self.field,
            old_size=self.size,
            new_size=_as_size_tuple(new_size),
            new_pix=_as_pix_tuple(new_pix),
        )
        self.size = _as_size_tuple(new_size)
        self.pix = _as_pix_tuple(new_pix)
        return self

    def gaussian_beam(self, E0, sigma, x_shift=0, y_shift=0, tilt_x_deg=0, tilt_y_deg=0):
        x, y, X, Y, dx, dy = self.xy_grid()
        field = E0 * np.exp(-((X - x_shift) ** 2 + (Y - y_shift) ** 2) / (2 * sigma ** 2)) + 0j

        if tilt_x_deg != 0 or tilt_y_deg != 0:
            k = 2 * np.pi / self.wavelength
            tx = np.deg2rad(tilt_x_deg)
            ty = np.deg2rad(tilt_y_deg)
            field *= np.exp(1j * k * (X * np.sin(tx) + Y * np.sin(ty)))

        self.field = field.astype(np.complex128)
        return self

    def top_hat(self, E0, radius):
        x, y, X, Y, dx, dy = self.xy_grid()
        R = np.sqrt(X**2 + Y**2)
        self.field = (E0 * (R <= radius)).astype(np.complex128)
        return self

    def mountain_range_beam(self, w0, m=2, n=8):
        x, y, X, Y, dx, dy = self.xy_grid()
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)
        z = np.exp(-(R / w0) ** n) * (
            1 + np.cos(m * PHI) * np.exp(-((R - w0 / 2) / (w0 / 3)) ** 4)
        ) + 0j + 0.1
        self.field = np.sqrt(z).astype(np.complex128)
        return self

    def paraboloid_beam(self, w0, n):
        x, y, X, Y, dx, dy = self.xy_grid()
        R = np.sqrt(X**2 + Y**2)
        z = R**2 * np.exp(-(R / w0) ** n)
        self.field = np.sqrt(z).astype(np.complex128)
        return self

    def apply_phase_mask(self, phase):
        phase = np.asarray(phase)
        if phase.shape != self.field.shape:
            raise ValueError(f"phase shape {phase.shape} must match field shape {self.field.shape}")
        self.field *= np.exp(1j * phase)
        return self

    def lens_phase(self, f):
        k = 2 * np.pi / self.wavelength
        x, y, X, Y, dx, dy = self.xy_grid()
        self.field *= np.exp(-1j * k * (X**2 + Y**2) / (2 * f))
        return self

    def grating(self, theta_x, theta_y):
        k = 2 * np.pi / self.wavelength
        x, y, X, Y, dx, dy = self.xy_grid()
        tx = np.deg2rad(theta_x)
        ty = np.deg2rad(theta_y)
        phase = k * (X * np.sin(tx) + Y * np.sin(ty))
        self.field *= np.exp(1j * phase)
        return self

    def circ_aperture(self, radius):
        """
        Circular aperture with radius in meters.
        """
        x, y, X, Y, dx, dy = self.xy_grid()
        R = np.sqrt(X**2 + Y**2)
        self.field *= (R < radius)
        return self

    def oap_phase(self, f, remove_focus=False):
        """
        Phase mask for a 90-degree off-axis parabolic (OAP) mirror.

        The scalar propagators in this class propagate along the local output
        axis. In that exit-pupil coordinate system a 90-degree OAP with parent
        focal length f has reflected focal length RFL = 2*f. The leading term
        is therefore the thin-lens phase for RFL, with the next terms carrying
        the OAP's off-axis aberration.

        Parameters
        ----------
        f : float
            Parent focal length [m]. Reflected focal length RFL = 2*f.
        remove_focus : bool
            If True, subtract the ideal RFL quadratic term and apply only the
            residual OAP aberration.
        """
        k = 2.0 * np.pi / self.wavelength
        _, _, X, Y, _, _ = self.xy_grid()

        # Intersection depth of the local output-axis ray with the 90-degree
        # OAP surface. The optical path change on reflection is 2*cos(45 deg)*Z.
        b = 2.0 * X - 8.0 * np.sqrt(2.0) * f
        c = X**2 + Y**2
        discriminant = b**2 - 4.0 * c
        if np.any(discriminant < 0):
            raise ValueError("field extends outside the valid 90-degree OAP projection")

        Z = (-b - np.sqrt(discriminant)) / 2.0
        OPD = 2.0 * np.sqrt(2.0) * Z

        if remove_focus:
            OPD -= (X**2 + Y**2) / (4.0 * f)

        self.field *= np.exp(-1j * k * OPD)
        return self
    
    def oap_phase2(self, f, remove_tilt=True, remove_focus=False):
        """
        Exact phase mask for a 90-degree OAP mirror.

        OPD is computed relative to a flat 45-degree reference mirror,
        which correctly captures focusing AND aberrations.

        The reference is the optical path a flat mirror at z=0 would
        produce: each ray at (xi, eta) would travel straight to the
        focus from (xi, eta, 0), with path length sqrt((xi+2f)^2+eta^2).

        Parameters
        ----------
        f : float
            Parent focal length [m]. Reflected focal length = 2*f.
        remove_tilt : bool
            Remove linear -xi tilt carrier. Default True.
        remove_focus : bool
            Remove the -eta^2/(4f) astigmatic focus term, leaving
            only coma and higher aberrations. Default False.
        """
        k = 2.0 * np.pi / self.wavelength
        _, _, X, Y, _, _ = self.xy_grid()

        xi  = X
        eta = Y

        # Exact mirror surface height
        z_mirror = xi + (xi**2 + eta**2) / (4.0 * f)

        # Leg 1: path from z=0 reference plane to mirror
        leg1 = -z_mirror

        # Leg 2 OAP: mirror hit point to focus at origin
        leg2_oap = np.sqrt((xi + 2.0*f)**2 + eta**2 + z_mirror**2)

        # Leg 2 flat reference: flat mirror at z=0 to focus at origin
        leg2_flat = np.sqrt((xi + 2.0*f)**2 + eta**2)

        # OPD relative to flat mirror reference
        OPD = leg2_oap

        if remove_tilt:
            OPD = OPD - xi          # cancel the -xi tilt term

        if remove_focus:
            OPD = OPD + eta**2 / (4.0 * f)   # cancel -eta^2/(4f)

        self.field *= np.exp(-1j * k * OPD)
        return self

    def vortex_phase(self, l):
        x, y, X, Y, dx, dy = self.xy_grid()
        phi = np.arctan2(Y, X)
        self.field *= np.exp(-1j * l * phi)
        return self

    @staticmethod
    def lg_phase_mask(l, p, theta, r, w0):
        field = assoc_laguerre(2 * (r / w0) ** 2, p, abs(l)) * np.exp(1j * l * theta)
        return np.mod(np.angle(field), 2 * np.pi)

    def fft_propagation(self, f):
        """
        Lens-Fourier propagation scaling for rectangular grids.

        Output pitches:
            dx_f = lambda * f / size_x
            dy_f = lambda * f / size_y

        Output sizes:
            size_x_f = pix_x * dx_f
            size_y_f = pix_y * dy_f
        """
        pix_x, pix_y = self.pix
        size_x, size_y = self.size
        lam = self.wavelength

        self.field = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(self.field))
        ).astype(np.complex128)

        dx = size_x / pix_x
        dy = size_y / pix_y

        size_x_f = lam * f / dx
        size_y_f = lam * f / dy
        self.size = (size_x_f, size_y_f)
        return self

    def ifft_backprop_from_focal(self, f):
        pix_x, pix_y = self.pix
        size_x_f, size_y_f = self.size
        lam = self.wavelength

        self.field = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(self.field))
        ).astype(np.complex128)

        size_x = lam * f * pix_x / size_x_f
        size_y = lam * f * pix_y / size_y_f
        self.size = (size_x, size_y)
        return self

    def ASM_propagation(self, z_prop):
        pix_x, pix_y = self.pix
        size_x, size_y = self.size

        dx = size_x / pix_x
        dy = size_y / pix_y
        k = 2 * np.pi / self.wavelength

        fx = np.fft.fftfreq(pix_x, d=dx)
        fy = np.fft.fftfreq(pix_y, d=dy)
        FX, FY = np.meshgrid(fx, fy, indexing="xy")

        kx = 2 * np.pi * FX
        ky = 2 * np.pi * FY

        kz = np.sqrt(k**2 - kx**2 - ky**2 + 0j)
        H = np.exp(1j * kz * z_prop)

        self.field = np.fft.ifft2(np.fft.fft2(self.field) * H).astype(np.complex128)
        return self

    def CZT_propagation(self, z, size_out, pix_out):
        """
        Rectangular CZT propagation.

        size_out = (size_x_out, size_y_out)
        pix_out  = (pix_x_out, pix_y_out)
        """
        size_out = _as_size_tuple(size_out)
        pix_out = _as_pix_tuple(pix_out)

        field = np.asarray(self.field, dtype=np.complex128)
        k = 2 * np.pi / self.wavelength

        size_x_in, size_y_in = self.size
        pix_x_in, pix_y_in = self.pix
        dx_in = size_x_in / pix_x_in
        dy_in = size_y_in / pix_y_in

        size_x_out, size_y_out = size_out
        pix_x_out, pix_y_out = pix_out
        dx_out = size_x_out / pix_x_out
        dy_out = size_y_out / pix_y_out

        x0 = (np.arange(pix_x_in) - pix_x_in / 2) * dx_in
        y0 = (np.arange(pix_y_in) - pix_y_in / 2) * dy_in
        X0, Y0 = np.meshgrid(x0, y0, indexing="xy")

        x1 = (np.arange(pix_x_out) - pix_x_out / 2) * dx_out
        y1 = (np.arange(pix_y_out) - pix_y_out / 2) * dy_out
        X1, Y1 = np.meshgrid(x1, y1, indexing="xy")

        F0 = np.exp(1j * k * z) / (1j * self.wavelength * z) * np.exp(
            1j * k / (2 * z) * (X1**2 + Y1**2)
        )
        F = np.exp(1j * k / (2 * z) * (X0**2 + Y0**2))
        gout = field * F

        # y pass
        fs_y = self.wavelength * z / dy_in
        ystart = -size_y_out / 2
        yend = size_y_out / 2
        fy1 = ystart + fs_y / 2
        fy2 = yend + fs_y / 2
        gout = _bluestein_dft_axis(gout, fy1, fy2, fs_y, pix_y_out, axis=0)

        # x pass
        fs_x = self.wavelength * z / dx_in
        xstart = -size_x_out / 2
        xend = size_x_out / 2
        fx1 = xstart + fs_x / 2
        fx2 = xend + fs_x / 2
        gout = _bluestein_dft_axis(gout, fx1, fx2, fs_x, pix_x_out, axis=1)

        gout = (dx_in * dy_in) * F0 * gout

        self.field = gout.astype(np.complex128)
        self.pix = pix_out
        self.size = size_out
        return self

    def CZT_back_propagation(self, z, size_out, pix_out):
        """
        Adjoint / conjugate back-propagation for rectangular grids.
        """
        size_out = _as_size_tuple(size_out)
        pix_out = _as_pix_tuple(pix_out)

        field = np.asarray(self.field, dtype=np.complex128)
        if field.shape != (self.pix[1], self.pix[0]):
            raise ValueError(f"field must have shape {(self.pix[1], self.pix[0])}, got {field.shape}")

        k = 2 * np.pi / self.wavelength

        # current grid = output plane of forward propagation
        size_x_cur, size_y_cur = self.size
        pix_x_cur, pix_y_cur = self.pix
        dx_cur = size_x_cur / pix_x_cur
        dy_cur = size_y_cur / pix_y_cur

        x_cur = (np.arange(pix_x_cur) - pix_x_cur / 2) * dx_cur
        y_cur = (np.arange(pix_y_cur) - pix_y_cur / 2) * dy_cur
        Xcur, Ycur = np.meshgrid(x_cur, y_cur, indexing="xy")

        # target grid = input plane of forward propagation
        size_x_tgt, size_y_tgt = size_out
        pix_x_tgt, pix_y_tgt = pix_out
        dx_tgt = size_x_tgt / pix_x_tgt
        dy_tgt = size_y_tgt / pix_y_tgt

        x_tgt = (np.arange(pix_x_tgt) - pix_x_tgt / 2) * dx_tgt
        y_tgt = (np.arange(pix_y_tgt) - pix_y_tgt / 2) * dy_tgt
        Xtgt, Ytgt = np.meshgrid(x_tgt, y_tgt, indexing="xy")

        F0 = np.exp(1j * k * z) / (1j * self.wavelength * z) * np.exp(
            1j * k / (2 * z) * (Xcur**2 + Ycur**2)
        )
        F = np.exp(1j * k / (2 * z) * (Xtgt**2 + Ytgt**2))

        gin = field * np.conj((dx_tgt * dy_tgt) * F0)

        # reverse x
        fs_x = self.wavelength * z / dx_tgt
        xstart = -size_x_cur / 2
        xend = size_x_cur / 2
        fx1 = xstart + fs_x / 2
        fx2 = xend + fs_x / 2
        gin = _bluestein_dft_adjoint_axis(gin, fx1, fx2, fs_x, pix_x_tgt, axis=1)

        # reverse y
        fs_y = self.wavelength * z / dy_tgt
        ystart = -size_y_cur / 2
        yend = size_y_cur / 2
        fy1 = ystart + fs_y / 2
        fy2 = yend + fs_y / 2
        gin = _bluestein_dft_adjoint_axis(gin, fy1, fy2, fs_y, pix_y_tgt, axis=0)

        gin = gin * np.conj(F)

        self.field = gin.astype(np.complex128)
        self.pix = pix_out
        self.size = size_out
        return self

    def CZT_propagation_diffractio(self, z, size_out, pix_out):
        size_out = _as_size_tuple(size_out)
        pix_out = _as_pix_tuple(pix_out)

        x, y, X, Y, dx, dy = self.xy_grid()
        field_diffractio = Scalar_field_XY(x, y, self.wavelength, n_background=1)
        field_diffractio.u = self.field

        self.size = size_out
        self.pix = pix_out
        x_out, y_out, X, Y, dx, dy = self.xy_grid()

        field_out = field_diffractio.CZT(z, x_out, y_out)
        self.field = np.array(field_out.u, dtype=np.complex128)
        return self

    def RaySom_propagation(self, size_f, pix_f, z_prop):
        size_f = _as_size_tuple(size_f)
        pix_f = _as_pix_tuple(pix_f)

        self.field = RaySom_propagation(
            self.field,
            self.size,
            self.pix,
            size_f,
            pix_f,
            z_prop,
            self.wavelength,
        )
        self.size = size_f
        self.pix = pix_f
        return self

    def plot(self, zoom=1):
        """
        Display intensity with clickable cross-sections.

        zoom may be:
            scalar        -> same in x and y
            (zoom_x, zoom_y)
        """
        if np.isscalar(zoom):
            zoom = (zoom, zoom)

        zoom_x, zoom_y = zoom
        size_x, size_y = self.size
        pix_x, pix_y = self.pix

        size_x_mm = size_x * 1e3 / zoom_x
        size_y_mm = size_y * 1e3 / zoom_y

        crop_x = max(1, int(round(pix_x / zoom_x)))
        crop_y = max(1, int(round(pix_y / zoom_y)))

        mid_x = pix_x // 2
        mid_y = pix_y // 2

        x0 = max(0, mid_x - crop_x // 2)
        x1 = min(pix_x, x0 + crop_x)
        y0 = max(0, mid_y - crop_y // 2)
        y1 = min(pix_y, y0 + crop_y)

        field_abs = np.abs(self.field[y0:y1, x0:x1])

        fig = plt.figure(figsize=(6, 5))
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            wspace=0.05,
            hspace=0.05
        )

        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_bottom = fig.add_subplot(gs[0, 0], sharex=ax_main)

        ax_main.imshow(
            field_abs**2,
            extent=[-size_x_mm / 2, size_x_mm / 2, -size_y_mm / 2, size_y_mm / 2],
            origin="lower",
            aspect="equal"
        )

        ax_main.set_xlabel("x (mm)")
        ax_main.set_ylabel("y (mm)")

        plt.setp(ax_right.get_yticklabels(), visible=False)
        plt.setp(ax_bottom.get_xticklabels(), visible=False)

        x = np.linspace(-size_x_mm / 2, size_x_mm / 2, field_abs.shape[1])
        y = np.linspace(-size_y_mm / 2, size_y_mm / 2, field_abs.shape[0])

        vline = ax_main.axvline(0, color="w", lw=1)
        hline = ax_main.axhline(0, color="w", lw=1)

        right_plot, = ax_right.plot(field_abs[:, field_abs.shape[1] // 2]**2, y)
        bottom_plot, = ax_bottom.plot(x, field_abs[field_abs.shape[0] // 2, :]**2)

        maxv = np.max(field_abs**2) if field_abs.size else 1.0
        ax_right.set_xlim(0, maxv * 1.05)
        ax_bottom.set_ylim(0, maxv * 1.05)

        def onclick(event):
            if event.inaxes != ax_main or event.xdata is None or event.ydata is None:
                return

            ix = np.argmin(np.abs(x - event.xdata))
            iy = np.argmin(np.abs(y - event.ydata))

            vline.set_xdata([x[ix], x[ix]])
            hline.set_ydata([y[iy], y[iy]])

            right_plot.set_xdata(field_abs[:, ix]**2)
            right_plot.set_ydata(y)

            bottom_plot.set_xdata(x)
            bottom_plot.set_ydata(field_abs[iy, :]**2)

            ax_right.relim()
            ax_right.autoscale_view(scalex=True, scaley=False)
            ax_bottom.relim()
            ax_bottom.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.tight_layout()
        plt.show()

    def focus_richards_wolf(self, f, NA, n=1.0, output_size=None, output_pix=None):
        """
        Vectorial focusing via Richards–Wolf integral (Novotny 3.43, 3.47, 3.51).
        Returns Ex, Ey, Ez in the focal plane.
        """

        lam = self.wavelength
        k   = 2*np.pi*n/lam

        sx, sy = self.size
        Nx, Ny = self.pix
        dx, dy = sx/Nx, sy/Ny

        # Pupil grid
        x = (np.arange(Nx) - Nx/2) * dx
        y = (np.arange(Ny) - Ny/2) * dy
        X, Y = np.meshgrid(x, y, indexing="xy")
        R = np.hypot(X, Y)

        # Sine condition
        pupil = R <= f*(NA/n)
        sin_t = np.where(pupil, R/f, 0.0)
        cos_t = np.where(pupil, np.sqrt(1 - sin_t**2), 1.0)
        phi   = np.arctan2(Y, X)

        # Input field
        E = self.field.astype(np.complex128) * pupil
        ap = 1.0 / np.sqrt(cos_t)   # cos_t = 1 outside pupil, correct Jacobian 1/cosθ

        # Eq. (3.51): projected far field
        Einf_x = E * ap * 0.5*((1+cos_t) - (1-cos_t)*np.cos(2*phi))
        Einf_y = E * ap * 0.5*(       - (1-cos_t)*np.sin(2*phi))
        Einf_z = E * ap * (-np.cos(phi)*sin_t)

        # Output grid
        if output_pix is None:
            Mx, My = Nx, Ny
        else:
            Mx, My = (int(output_pix), int(output_pix)) if np.isscalar(output_pix) else map(int, output_pix)

        if output_size is None:
            sx_out = lam*f/sx * Mx
            sy_out = lam*f/sy * My
        else:
            sx_out, sy_out = (output_size, output_size) if np.isscalar(output_size) else output_size

        # CZT-based Fourier transform
        def _czt2(U):
            nat_sx, nat_sy = lam*f/dx, lam*f/dy
            fx1, fx2 = -sx_out/(2*nat_sx)+0.5, sx_out/(2*nat_sx)+0.5
            fy1, fy2 = -sy_out/(2*nat_sy)+0.5, sy_out/(2*nat_sy)+0.5

            V = _bluestein_dft(U,    fy1, fy2, 1.0, My)
            V = _bluestein_dft(V.T,  fx1, fx2, 1.0, Mx).T
            V *= dx*dy

            # Phase correction (Bluestein shift)
            x_out = np.linspace(-sx_out/2, sx_out/2, Mx, endpoint=False) + sx_out/(2*Mx)
            y_out = np.linspace(-sy_out/2, sy_out/2, My, endpoint=False) + sy_out/(2*My)
            Xo, Yo = np.meshgrid(x_out, y_out, indexing="xy")

            return V * np.exp(1j*np.pi*(1 + Xo*dx/nat_sx + Yo*dy/nat_sy))

        Ex = _czt2(Einf_x)
        Ey = _czt2(Einf_y)
        Ez = _czt2(Einf_z)

        # RW prefactor
        pref = 1j * k / (2 * np.pi * f) * np.exp(-1j * k * f)
        Ex *= pref; Ey *= pref; Ez *= pref

        # Wrap into optical_field objects
        def _wrap(F):
            obj = optical_field(size=(sx_out, sy_out), pix=(Mx, My), wavelength=lam)
            obj.field = F
            return obj

        Ex_obj, Ey_obj, Ez_obj = _wrap(Ex), _wrap(Ey), _wrap(Ez)

        # Plot (kept as requested)
        _plot_focal_components(Ex, Ey, Ez, sx_out, sy_out, lam, f, NA, n)

        return Ex_obj, Ey_obj, Ez_obj
    
    def oap_phase_offaxis_exact(self, f, alpha):
        """
        Exact phase mask for 90-degree OAP with tilted input beam.

        Parameters
        ----------
        f : float
            Parent focal length [m]. Reflected focal length RFL = 2*f.
        alpha : float
            Input tilt angle [rad] in the xz-plane (plane of deflection).
            alpha=0 gives the perfect on-axis case.
        """
        k = 2.0 * np.pi / self.wavelength
        _, _, X, Y, _, _ = self.xy_grid()

        xi  = X
        eta = Y

        # --- alpha=0 guard: perfect lens, no aberrations ---
        if alpha == 0.0:
            phase = -k * (xi**2 + eta**2) / (4.0 * f)
            self.field *= np.exp(1j * phase)
            return self

        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)

        # Ray start point on input wavefront plane through operating point (2f,0,0)
        # Pupil basis vectors perpendicular to k_in = (-sin_a, 0, -cos_a):
        #   e_xi  = (cos_a, 0, -sin_a)
        #   e_eta = (0, 1, 0)
        rx0 = 2*f + xi*cos_a
        ry0 = eta
        rz0 = -xi*sin_a

        # Find t where ray r(t) = r0 + t*k_in hits the mirror z=(x^2+y^2)/(4f)-f
        # k_in = (-sin_a, 0, -cos_a)
        # x(t) = rx0 - t*sin_a
        # z(t) = rz0 - t*cos_a
        # Mirror condition: z(t) = (x(t)^2 + ry0^2)/(4f) - f
        # Rearranging gives quadratic: A*t^2 + B*t + C = 0
        A =  sin_a**2 / (4*f)
        B = -(2*rx0*sin_a / (4*f) + cos_a)
        C =  (rx0**2 + ry0**2) / (4*f) - f - rz0

        disc = B**2 - 4*A*C

        if np.any(disc < 0):
            raise ValueError(
                "Some rays miss the mirror surface entirely. "
                "Reduce beam size or check geometry."
            )

        # Take the physical solution (smaller t = first intersection)
        t_hit = (-B - np.sqrt(disc)) / (2*A)

        # Actual hit point on mirror
        x_hit = rx0 - t_hit * sin_a
        y_hit = ry0
        z_hit = rz0 - t_hit * cos_a

        # OPL from input wavefront to mirror hit point = t (distance along ray)
        OPL_in = t_hit

        # Reflected direction of chief ray (xi=eta=0) off operating point (2f,0,0)
        # Surface normal at operating point: n0 = (-1, 0, 1)/sqrt(2)
        n0       = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
        k_in_vec = np.array([-sin_a, 0.0, -cos_a])
        k_out    = k_in_vec - 2.0 * np.dot(k_in_vec, n0) * n0
        k_out    = k_out / np.linalg.norm(k_out)

        # Image point: 2f along reflected chief ray from operating point
        r_image        = np.array([2*f, 0.0, 0.0]) + 2*f * k_out
        x_im, y_im, z_im = r_image

        # OPL from mirror hit point to image point
        leg2 = np.sqrt((x_hit - x_im)**2 + (y_hit - y_im)**2 + (z_hit - z_im)**2)

        # Reference OPL: chief ray (xi=eta=0)
        # Chief ray starts exactly on the mirror surface (t_hit=0),
        # then travels 2f to the image point
        OPL_ref = 2.0 * f

        # Wavefront aberration
        OPD = (OPL_in + leg2) - OPL_ref

        self.field *= np.exp(1j * k * OPD)
        return self
    
 
# --------------------------------------------------------------------------- #
#  Plotting helper
# --------------------------------------------------------------------------- #
 
def _plot_focal_components(Ex, Ey, Ez, size_x, size_y, lam, f, NA, n):
    """
    4-panel figure: |Ex|^2, |Ey|^2, |Ez|^2, and total |E|^2 in focal plane.
    Each panel normalised to its own peak; percentage of global peak shown.
    """
    Ix    = np.abs(Ex)**2
    Iy    = np.abs(Ey)**2
    Iz    = np.abs(Ez)**2
    I_tot = Ix + Iy + Iz
 
    global_peak = max(Ix.max(), Iy.max(), Iz.max(), I_tot.max())
    if global_peak == 0.0:
        global_peak = 1.0
 
    panels = [
        (Ix,    r"$|E_x|^2$",          "magma"  ),
        (Iy,    r"$|E_y|^2$",          "cividis"),
        (Iz,    r"$|E_z|^2$",          "viridis"),
        (I_tot, r"$|E_{\rm tot}|^2$",  "inferno"),
    ]
 
    scale = 1e6   # m -> µm
    ext   = [-size_x/2*scale, size_x/2*scale,
             -size_y/2*scale, size_y/2*scale]
 
    bg  = "#111111"
    fig = plt.figure(figsize=(15, 4.8), facecolor=bg)
    gs  = gridspec.GridSpec(1, 4, figure=fig,
                            wspace=0.38,
                            left=0.06, right=0.97,
                            top=0.82,  bottom=0.14)
 
    for col, (data, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(bg)
 
        vmax = data.max() if data.max() > 0 else 1.0
        im = ax.imshow(data, extent=ext, origin="lower",
                       aspect="equal", cmap=cmap,
                       vmin=0, vmax=vmax, interpolation="lanczos")
 
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="#888888", labelsize=12)
        cb.outline.set_edgecolor("#444444")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#aaaaaa")
 
        rel = data.max() / global_peak * 100.0
        ax.set_title(f"{title}\n{rel:.1f} % of peak",
                     color="white", fontsize=14, pad=5)
        ax.tick_params(colors="#888888", labelsize=12)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
 
        ax.set_xlabel("x  (µm)", color="#888888", fontsize=14)
        if col == 0:
            ax.set_ylabel("y  (µm)", color="#888888", fontsize=14)
 
    fig.suptitle(
        f"Richards–Wolf focal decomposition  |  "
        f"NA = {NA:.2f},  f = {f*1e3:.1f} mm,  "
        f"λ = {lam*1e9:.0f} nm,  n = {n:.2f}",
        color="white", fontsize=14, y=0.95,
    )
    plt.show()

# =============================================================================
# Rayleigh-Sommerfeld
# =============================================================================

@njit(parallel=True)
def RaySom_propagation(field, size_i, pix_i, size_f, pix_f, z_prop, wavelength):
    """
    Rectangular Rayleigh-Sommerfeld propagation.

    field.shape = (pix_i_y, pix_i_x)
    size_i = (size_x_i, size_y_i)
    pix_i  = (pix_x_i, pix_y_i)
    """
    size_ix, size_iy = size_i[0], size_i[1]
    pix_ix, pix_iy = pix_i[0], pix_i[1]

    size_fx, size_fy = size_f[0], size_f[1]
    pix_fx, pix_fy = pix_f[0], pix_f[1]

    k0 = 2 * np.pi / wavelength
    E_final = np.zeros((pix_fy, pix_fx), dtype=np.complex128)

    x_i = np.linspace(-size_ix / 2, size_ix / 2, pix_ix)
    y_i = np.linspace(-size_iy / 2, size_iy / 2, pix_iy)
    x_f = np.linspace(-size_fx / 2, size_fx / 2, pix_fx)
    y_f = np.linspace(-size_fy / 2, size_fy / 2, pix_fy)

    dx_i = x_i[1] - x_i[0] if pix_ix > 1 else size_ix
    dy_i = y_i[1] - y_i[0] if pix_iy > 1 else size_iy
    dS_i = dx_i * dy_i

    for iy_f in prange(pix_fy):
        yf = y_f[iy_f]
        for ix_f in range(pix_fx):
            xf = x_f[ix_f]
            acc = 0.0 + 0.0j
            for iy_i in range(pix_iy):
                yi = y_i[iy_i]
                for ix_i in range(pix_ix):
                    xi = x_i[ix_i]
                    s = np.sqrt(z_prop**2 + (xf - xi)**2 + (yf - yi)**2)
                    acc += field[iy_i, ix_i] * np.exp(1j * k0 * s) * z_prop / s**2
            E_final[iy_f, ix_f] = acc

    return -1j / wavelength * dS_i * E_final


# =============================================================================
# Rectangular mask generators
# =============================================================================

def zernike(n, m, N=None, pix=None, mask=True):
    """
    Zernike polynomial on a rectangular grid.

    Parameters
    ----------
    n, m : int
        Zernike indices.
    N or pix : scalar or (Nx, Ny)
        Output grid size. Scalar keeps square backward compatibility.
    mask : bool
        Mask outside unit disk.

    Returns
    -------
    Z : ndarray, shape (Ny, Nx)
    """
    pix_x, pix_y = _normalize_rect_args(N=N, pix=pix)

    tx = np.linspace(-1.0, 1.0, pix_x)
    ty = np.linspace(-1.0, 1.0, pix_y)
    x, y = np.meshgrid(tx, ty, indexing="xy")
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)

    mabs = abs(m)
    if (mabs > n) or ((n - mabs) & 1):
        return np.zeros((pix_y, pix_x), dtype=float)

    smax = (n - mabs) // 2
    R = np.zeros_like(r)
    for s in range(smax + 1):
        R += (
            (-1) ** s
            * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial((n + mabs) // 2 - s)
                * math.factorial((n - mabs) // 2 - s)
            )
            * r ** (n - 2 * s)
        )

    if m == 0:
        Z = R
    elif m > 0:
        Z = R * np.cos(m * th)
    else:
        Z = R * np.sin(mabs * th)

    return np.where(r <= 1.0, Z, np.nan) if mask else Z


def lg_phase(p, l, N=None, pix=None, w=0.5, mask=True):
    """
    LG phase mask on a rectangular normalized grid.

    Parameters
    ----------
    p, l : int
        LG indices.
    N or pix : scalar or (Nx, Ny)
        Output grid size.
    w : float
        Waist in normalized coordinates.
    mask : bool
        Mask outside unit disk.
    """
    pix_x, pix_y = _normalize_rect_args(N=N, pix=pix)

    a = abs(l)
    tx = np.linspace(-1.0, 1.0, pix_x)
    ty = np.linspace(-1.0, 1.0, pix_y)
    x, y = np.meshgrid(tx, ty, indexing="xy")
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)

    xarg = 2.0 * (r / w) ** 2
    L = np.zeros_like(r, dtype=float)
    for mm in range(p + 1):
        L += (
            (-1) ** mm
            * math.factorial(p + a)
            / (
                math.factorial(p - mm)
                * math.factorial(a + mm)
                * math.factorial(mm)
            )
            * (xarg ** mm)
        )

    radial_jump = np.where(L >= 0.0, 0.0, np.pi)
    phase = (l * th + radial_jump) % (2.0 * np.pi)

    if mask:
        phase = np.where(r <= 1.0, phase, 0.0)

    return phase.astype(np.float32)


def hg_phase(n, m, N=None, pix=None, w=0.5, mask=True):
    """
    HG phase mask on a rectangular normalized grid.

    Parameters
    ----------
    n, m : int
        HG indices.
    N or pix : scalar or (Nx, Ny)
        Output grid size.
    w : float
        Waist in normalized coordinates.
    mask : bool
        Mask outside unit disk.
    """
    pix_x, pix_y = _normalize_rect_args(N=N, pix=pix)

    tx = np.linspace(-1.0, 1.0, pix_x)
    ty = np.linspace(-1.0, 1.0, pix_y)
    x, y = np.meshgrid(tx, ty, indexing="xy")

    sx = np.sqrt(2.0) * x / w
    sy = np.sqrt(2.0) * y / w

    def H(k, z):
        if k == 0:
            return np.ones_like(z)
        if k == 1:
            return 2.0 * z
        Hkm2 = np.ones_like(z)
        Hkm1 = 2.0 * z
        for i in range(2, k + 1):
            Hk = 2.0 * z * Hkm1 - 2.0 * (i - 1) * Hkm2
            Hkm2, Hkm1 = Hkm1, Hk
        return Hkm1

    Hn = H(n, sx)
    Hm = H(m, sy)

    phase = np.where((Hn * Hm) >= 0.0, 0.0, np.pi).astype(np.float32)

    if mask:
        r = np.sqrt(x * x + y * y)
        phase = np.where(r <= 1.0, phase, 0.0).astype(np.float32)

    return phase


# =============================================================================
# Zoom / bandlimit / rescale
# =============================================================================

def zoom_field(U, zoom_factor, order=3, mode="constant", cval=0.0):
    """
    Real-space zoom of a complex field, then center-crop/pad back to original shape.

    Parameters
    ----------
    U : ndarray, complex
        2D field.
    zoom_factor : scalar or (zoom_x, zoom_y)
        zoom > 1 : zoom in
        zoom < 1 : zoom out
    """
    from scipy.ndimage import zoom as ndi_zoom

    if np.isscalar(zoom_factor):
        zoom_x = zoom_y = float(zoom_factor)
    else:
        zoom_x, zoom_y = float(zoom_factor[0]), float(zoom_factor[1])

    if zoom_x <= 0 or zoom_y <= 0:
        raise ValueError("zoom_factor entries must be > 0.")

    Ny, Nx = U.shape

    Ur = ndi_zoom(U.real, zoom=(zoom_y, zoom_x), order=order, mode=mode, cval=float(np.real(cval)))
    Ui = ndi_zoom(U.imag, zoom=(zoom_y, zoom_x), order=order, mode=mode, cval=float(np.imag(cval)))
    Uz = Ur + 1j * Ui

    My, Mx = Uz.shape

    if Mx > Nx:
        sx = (Mx - Nx) // 2
        Uz = Uz[:, sx:sx + Nx]
    elif Mx < Nx:
        p0 = (Nx - Mx) // 2
        p1 = Nx - Mx - p0
        Uz = np.pad(Uz, ((0, 0), (p0, p1)), mode="constant", constant_values=0)

    My, Mx = Uz.shape
    if My > Ny:
        sy = (My - Ny) // 2
        Uz = Uz[sy:sy + Ny, :]
    elif My < Ny:
        p0 = (Ny - My) // 2
        p1 = Ny - My - p0
        Uz = np.pad(Uz, ((p0, p1), (0, 0)), mode="constant", constant_values=0)

    return Uz


def bandlimit_field(U, size, wavelength, NA, soft_edge=0.0):
    """
    Circular frequency-domain bandlimit on rectangular spatial grid.
    """
    size_x, size_y = _as_size_tuple(size)
    Ny, Nx = U.shape

    fc = NA / wavelength

    dx = size_x / Nx
    dy = size_y / Ny

    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    R = np.sqrt(FX**2 + FY**2)

    if soft_edge <= 0:
        P = (R <= fc).astype(float)
    else:
        r0 = fc * (1.0 - soft_edge)
        P = np.ones_like(R, dtype=float)
        P[R >= fc] = 0.0
        band = (R >= r0) & (R < fc)
        P[band] = 0.5 * (1 + np.cos(np.pi * (R[band] - r0) / (fc - r0)))

    return np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(U)) * P))


def rescale(U, old_size, new_size, new_pix):
    """
    Rectangular rescale:
      1) crop/pad independently in x and y to preserve old pixel pitch
      2) interpolate to new pixel count

    Parameters
    ----------
    U : ndarray, shape (old_pix_y, old_pix_x)
    old_size : scalar or (size_x, size_y)
    new_size : scalar or (size_x, size_y)
    new_pix  : scalar or (pix_x, pix_y)
    """
    if U.ndim != 2:
        raise ValueError("U must be 2D")

    old_size = _as_size_tuple(old_size)
    new_size = _as_size_tuple(new_size)
    new_pix = _as_pix_tuple(new_pix)

    old_size_x, old_size_y = old_size
    new_size_x, new_size_y = new_size
    new_pix_x, new_pix_y = new_pix

    old_pix_y, old_pix_x = U.shape

    target_x = max(1, int(round(old_pix_x * (new_size_x / old_size_x))))
    target_y = max(1, int(round(old_pix_y * (new_size_y / old_size_y))))

    U = _crop_or_pad_1d(U, axis=1, n_target=target_x)
    U = _crop_or_pad_1d(U, axis=0, n_target=target_y)

    if U.shape != (new_pix_y, new_pix_x):
        zoom_y = new_pix_y / U.shape[0]
        zoom_x = new_pix_x / U.shape[1]
        U = _resample_complex(U, zoom_y=zoom_y, zoom_x=zoom_x, order=1)

    return U.astype(np.complex128, copy=False)


# =============================================================================
# Bluestein / CZT helpers
# =============================================================================

def _bluestein_dft(x, f1, f2, fs, mout):
    """
    1D forward Bluestein DFT along axis 0.

    Input:
        x.shape = (m, n)
    Output:
        out.shape = (mout, n)
    """
    x = np.asarray(x, dtype=np.complex128)
    m, n = x.shape

    f11 = f1 + (mout * fs + f2 - f1) / (2 * mout)
    f22 = f2 + (mout * fs + f2 - f1) / (2 * mout)

    a = np.exp(1j * 2 * np.pi * f11 / fs)
    w = np.exp(-1j * 2 * np.pi * (f22 - f11) / (mout * fs))

    h_idx = np.arange(-m + 1, max(mout - 1, m - 1) + 1)
    mp = m + mout - 1
    h = w ** ((h_idx ** 2) / 2.0)

    nfft = 1 << int(np.ceil(np.log2(mp)))
    ft = np.fft.fft(1.0 / h[:mp], nfft)

    b_pre = (a ** (-np.arange(m))) * h[m - 1:2 * m - 1]
    b = np.fft.fft(x * b_pre[:, None], nfft, axis=0)
    b = np.fft.ifft(b * ft[:, None], axis=0)

    h_seg = h[m - 1:mp]
    b = b[m - 1:mp, :] * h_seg[:, None]

    l = np.linspace(0, mout - 1, mout)
    l = l / mout * (f22 - f11) + f11

    Mshift = -m / 2.0
    phase_shift = np.exp(-1j * 2 * np.pi * l * (Mshift + 0.5) / fs)

    b = b * phase_shift[:, None]
    return b


def _bluestein_dft_adjoint(y, f1, f2, fs, m_in):
    """
    Adjoint of _bluestein_dft under standard complex inner product.

    Input:
        y.shape = (mout, n)
    Output:
        x_adj.shape = (m_in, n)
    """
    y = np.asarray(y, dtype=np.complex128)
    mout, n = y.shape
    m = m_in

    f11 = f1 + (mout * fs + f2 - f1) / (2 * mout)
    f22 = f2 + (mout * fs + f2 - f1) / (2 * mout)

    a = np.exp(1j * 2 * np.pi * f11 / fs)
    w = np.exp(-1j * 2 * np.pi * (f22 - f11) / (mout * fs))

    h_idx = np.arange(-m + 1, max(mout - 1, m - 1) + 1)
    mp = m + mout - 1
    h = w ** ((h_idx ** 2) / 2.0)
    h_seg = h[m - 1:mp]

    nfft = 1 << int(np.ceil(np.log2(mp)))
    ft = np.fft.fft(1.0 / h[:mp], nfft)

    l = np.linspace(0, mout - 1, mout)
    l = l / mout * (f22 - f11) + f11

    Mshift = -m / 2.0
    phase_shift = np.exp(-1j * 2 * np.pi * l * (Mshift + 0.5) / fs)

    g = y * np.conj(phase_shift)[:, None]
    c_grad = g * np.conj(h_seg)[:, None]

    c_full = np.zeros((nfft, n), dtype=np.complex128)
    c_full[m - 1:mp, :] = c_grad

    tmp = np.fft.fft(c_full, axis=0)
    tmp = tmp * np.conj(ft)[:, None]
    tmp = np.fft.ifft(tmp, axis=0)

    b_pre = (a ** (-np.arange(m))) * h[m - 1:2 * m - 1]
    x_adj = tmp[:m, :] * np.conj(b_pre)[:, None]
    return x_adj


def _bluestein_dft_axis(U, f1, f2, fs, mout, axis):
    """
    Apply 1D Bluestein DFT to selected axis of 2D array.
    """
    if axis == 0:
        return _bluestein_dft(U, f1, f2, fs, mout)
    elif axis == 1:
        return _bluestein_dft(U.T, f1, f2, fs, mout).T
    else:
        raise ValueError("axis must be 0 or 1")


def _bluestein_dft_adjoint_axis(U, f1, f2, fs, m_in, axis):
    """
    Apply adjoint 1D Bluestein DFT to selected axis of 2D array.
    """
    if axis == 0:
        return _bluestein_dft_adjoint(U, f1, f2, fs, m_in)
    elif axis == 1:
        return _bluestein_dft_adjoint(U.T, f1, f2, fs, m_in).T
    else:
        raise ValueError("axis must be 0 or 1")
