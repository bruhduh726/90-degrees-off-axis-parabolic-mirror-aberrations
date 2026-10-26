import time

import numpy as np
import matplotlib.pyplot as plt

import optics_lib as opt

plt.rcParams.update({'font.size': 13})

size = 10e-3
pix = 2**11
f = 20e-3
field = opt.optical_field(size, pix, wavelength=1064e-9)
field.gaussian_beam(1, 3e-3, x_shift=0e-3)
#field.oap_phase_offaxis_exact(f=f, alpha=1 * np.pi / 180)
field.oap_phase(f=f)


tick = time.perf_counter()
field.CZT_propagation(2*f, size/100, 1024)
tock = time.perf_counter()
print(f"computation time = {tock-tick:e}s")

field.plot(zoom=1)
