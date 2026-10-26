# 90° off-axis parabolic mirror – aberrations

Vectorial simulation of the focal field of a 90° off-axis parabolic (OAP)
mirror, used to study its aberrations: what the focus looks like for an ideal
mirror, how a real gold coating changes it, and how much tilt the mirror can
take before the focus stops being diffraction-limited.

The mirror is modelled as a segment of a parent paraboloid in 3-D. The input
field is reflected vectorially off the real surface (including rigid-body tilt
and decenter), and the field near focus is evaluated with a Debye–Wolf
integral. The full derivation is in [`notes/theory.md`](notes/theory.md).

## Contents

```text
parabolic_mirror/
|-- OAP_method_stepbystep.ipynb   # the method step by step: paraboloid -> aperture ->
|                                 # input field -> reflection -> Debye-Wolf focus
|                                 # (example: polarised Gaussian beam with a vortex)
|-- OAP_perfect_vs_gold.ipynb     # perfect conductor vs gold coating, vectorial
|                                 # Fresnel s/p reflection
|-- OAP_tilt_tolerancing.ipynb    # how much tilt the off-axis segment tolerates;
|                                 # exact ray/surface intersection, Strehl scan
|-- notes/
|   `-- theory.md                 # derivations, conventions, approximations
|-- src/
|   |-- oap_core.py               # geometry & reflection: surface, normals, rigid
|   |                             # transforms, optical path, apodisation, k-mapping
|   |-- oap_prop.py               # aligned mirror: pull-back to k-space + exact DFT
|   |-- oap_tilt.py               # tilted/decentred mirror: forward nonuniform DFT
|   `-- optics_lib/               # general scalar-optics library (optical_field
|                                 # class, FFT/ASM/CZT/Rayleigh-Sommerfeld propagation,
|                                 # Richards-Wolf focusing, Zernike/LG/HG modes)
|-- parmir.py                     # small scalar example using optics_lib
|-- pyproject.toml
`-- uv.lock
```

The notebooks are self-contained and need only `numpy` and `matplotlib`. They are
the reference implementation: all three model the same off-axis segment with the
same clear aperture (a circle in the mirror-local y-z plane) and the same
wavefront statistics (RMS and Strehl with piston, tilt and focus shift removed;
see `notes/theory.md` §9.4). All parameters are in the first code cell of each
notebook.

The `src/oap_*.py` modules are an earlier version: on-axis parent paraboloid
only, ideal-conductor reflection, parent-pupil sampling in the tilt path, and
they need `scipy`.

The input field is the field just in front of the mirror, collimated along the
incident direction; a strongly non-collimated beam (e.g. after an SLM and a
lens) must be propagated to the mirror first. See `notes/theory.md` §10.

## Setup

The project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv` and installs `optics_lib` into it, so scripts at the
project root can `import optics_lib as opt`.

## Usage

Open the notebooks in Jupyter, or run the scalar example:

```bash
uv run python parmir.py
```

The `oap_*.py` modules import each other by name (`import oap_core`), so run
code that uses them from `src/` or add `src/` to `PYTHONPATH`.

## License

[MIT](LICENSE)
