# Vectorial OAP propagation — theory and step-by-step derivations

**Audience.** A physics student who has had ~3 semesters of maths (multivariable
calculus, Jacobians, linear algebra with rotation matrices, dot/cross products)
and introductory optics (plane waves, Fourier transforms, the idea of an angular
spectrum). Every non-obvious step is derived. Symbols are defined the first time
they appear.

**Where things live.** The three notebooks are the reference implementation: off-axis
segment, exact ray/surface intersection, Fresnel reflection, the wavefront statistics of
§9.4. The `src/` modules (§8–9) are an earlier version for the on-axis parent only
(ideal conductor, parent-pupil sampling in the tilt path); they need `scipy`.

Throughout, a **carat** denotes a unit vector ($\hat{\mathbf u}$), boldface a
3-vector ($\mathbf r$), and $k = 2\pi/\lambda$ is the wavenumber.

---

## 0. What the code computes, in one picture

We send a collimated (or arbitrary) beam onto a **parabolic mirror**, and we want
the electric field in the tiny region near its **focus** — including the vector
character of the light (a tightly focused beam is *not* purely transverse).

```
 input field E_in(x,y)            reflected vector field            focal field
 on a pupil plane        --->     on the mirror surface     --->   E(v,w) = (Ex,Ey,Ez)
 (amplitude + phase)              (one ray per pupil point)         in a small window
```

The central modelling decision: we do **not** replace the mirror by a thin lens
with a quadratic phase $\exp(-ik r^2/2f)$. Instead we use the mirror's real 3-D
geometry. Why this matters is explained in §5.3.

The mathematical engine is the **angular-spectrum (Debye–Wolf) representation of
a focus**: near the focus the field is a coherent superposition of plane waves,
one for each point of the pupil, and that superposition is a 2-D Fourier
transform. Building that correctly is the whole game.

---

## 1. Coordinate systems

We use one fixed **laboratory frame** $\mathbf r = (x,y,z)$ and never rotate it,
even though the beam is folded by ~90°. We keep the fold as a choice of *output
axes*, not as a rotation of the world (§6.3).

Three conceptual frames:

1. **Lab frame** $\mathbf r_{\text{lab}}=(x,y,z)$ — fixed; describes the experiment.
2. **Parent-paraboloid frame** $\mathbf r_p=(x_p,y_p,z_p)$ — the natural frame of
   the infinite "parent" paraboloid the commercial optic is cut from.
3. A **rigid transform** relating them:
$$
\mathbf r_{\text{lab}} = R\,\mathbf r_p + \mathbf t ,
$$
with $R$ a $3\times3$ rotation matrix and $\mathbf t$ a translation. This single
transform is the **only** mechanism for tilt/decenter — we never bolt on
hand-fitted coma/astigmatism.

The **output basis** $(\hat{\mathbf u},\hat{\mathbf v},\hat{\mathbf w})$ is an
orthonormal triad with $\hat{\mathbf u}$ the nominal outgoing direction and
$\hat{\mathbf v},\hat{\mathbf w}$ transverse to it. For the aligned parent used in
`src/`, $\hat{\mathbf u}=\hat{\mathbf z}$, $\hat{\mathbf v}=\hat{\mathbf x}$,
$\hat{\mathbf w}=\hat{\mathbf y}$; for the 90°-folded lab layout it is
$\hat{\mathbf u}=\hat{\mathbf x}$ etc. The equations below are written in the
$(u,v,w)$ basis so they are reusable for any orientation.

---

## 2. The parent paraboloid: surface and normal

### 2.1 Surface

A paraboloid of revolution with **parent focal length** $f$ (here $f=7.5$ mm) is
$$
\boxed{\,z_p = \dfrac{x_p^2 + y_p^2}{4f}\,}\qquad\text{equivalently } F(\mathbf r_p)\equiv z_p-\frac{x_p^2+y_p^2}{4f}=0 .
$$
Its focus sits on the axis at $\mathbf r_{\text{focus}}=(0,0,f)$. (Careful: the
commercial "reflected focal length" $\approx 15$ mm is a *different* number — the
distance from the off-axis piece to the focus, $f+\rho_0^2/4f=2f$ for a 90° OAP — and
must **not** be put into the surface equation.)

### 2.2 Normal from the gradient

For any surface written as a level set $F(\mathbf r)=0$, the normal is parallel to
$\nabla F$ (the gradient points perpendicular to level surfaces — a standard
multivariable-calculus fact). Here
$$
\nabla F = \Big(-\tfrac{x_p}{2f},\,-\tfrac{y_p}{2f},\,1\Big).
$$
Normalising,
$$
\boxed{\;\hat{\mathbf n}_p=\dfrac{1}{\sqrt{1+\dfrac{x_p^2+y_p^2}{4f^2}}}\Big(-\tfrac{x_p}{2f},\,-\tfrac{y_p}{2f},\,1\Big).\;}
$$
Under the rigid transform the normal rotates the same way the surface does (a
rotation preserves angles and lengths), and because $R$ is orthogonal the result
stays a unit vector:
$$
\hat{\mathbf n}_{\text{lab}} = R\,\hat{\mathbf n}_p , \qquad \lVert\hat{\mathbf n}_{\text{lab}}\rVert=1 .
$$
(The translation $\mathbf t$ does not affect directions, only positions.)

---

## 3. Reflection

### 3.1 Ray direction (a Householder reflection)

Let $\hat{\mathbf e}_i$ be the incident **propagation direction** (a unit vector;
here $\hat{\mathbf e}_i=-\hat{\mathbf z}$). Reflection off a surface with normal
$\hat{\mathbf n}$ flips the component of $\hat{\mathbf e}_i$ *along* the normal and
keeps the tangential component. Decompose
$\hat{\mathbf e}_i=\hat{\mathbf e}_i^{\parallel}+\hat{\mathbf e}_i^{\perp}$ with
$\hat{\mathbf e}_i^{\parallel}=(\hat{\mathbf e}_i\!\cdot\!\hat{\mathbf n})\hat{\mathbf n}$.
Flipping the normal part ($\hat{\mathbf e}_i^{\parallel}\to-\hat{\mathbf e}_i^{\parallel}$):
$$
\boxed{\;\hat{\mathbf e}_r=\hat{\mathbf e}_i-2(\hat{\mathbf e}_i\!\cdot\!\hat{\mathbf n})\,\hat{\mathbf n}.\;}
$$
This is a **Householder reflection**; it preserves length, so
$\lVert\hat{\mathbf e}_r\rVert=1$ automatically.

### 3.2 The electric field at a mirror

For an ideal mirror the reflected field vector is
$$
\boxed{\;\mathbf E_r=-\mathbf E_i+2(\mathbf E_i\!\cdot\!\hat{\mathbf n})\,\hat{\mathbf n}.\;}
$$
Check it against normal incidence ($\hat{\mathbf n}=\hat{\mathbf z}$,
$\mathbf E_i=\hat{\mathbf x}$): $\mathbf E_r=-\hat{\mathbf x}$ — the tangential
field flips sign on reflection off a conductor, as it must ($\pi$ phase shift).
This is an **ideal-conductor** model: it treats $s$- and $p$-polarised light
identically. A real metal coating at ~45° gives a small $s$–$p$ phase difference;
the notebooks replace it by the Fresnel $s/p$ form (step-by-step and perfect-vs-gold).

### 3.3 The reflected field is transverse to the reflected ray

We must check the reflected field is perpendicular to the reflected propagation
direction (otherwise it would not be a valid transverse wave). Compute
$\hat{\mathbf e}_r\!\cdot\!\mathbf E_r$ with the two boxed formulas. Expanding and
using $\hat{\mathbf n}\!\cdot\!\hat{\mathbf n}=1$, all the cross terms collapse to
$$
\hat{\mathbf e}_r\!\cdot\!\mathbf E_r=-(\hat{\mathbf e}_i\!\cdot\!\mathbf E_i).
$$
So **if the incident field is transverse** ($\hat{\mathbf e}_i\!\cdot\!\mathbf E_i=0$),
the reflected field is transverse too. Good — the model is self-consistent. (The
notebooks check it numerically to machine precision.)

---

## 4. The focusing property of the parabola

### 4.1 A parabola turns an axial plane wave into a perfect converging wave

Take the aligned parent and an axial plane wave $\hat{\mathbf e}_i=-\hat{\mathbf z}$.
Two facts make it focus perfectly.

**(a) Every reflected ray passes through the focus.** Insert the paraboloid normal
into the reflection formula. With $\rho^2\equiv x_p^2+y_p^2$ one finds
$$
\hat{\mathbf e}_r=\frac{1}{4f^2+\rho^2}\big(-4f\,x_p,\;-4f\,y_p,\;4f^2-\rho^2\big).
$$
Starting from the mirror point $\mathbf r_m=(x_p,y_p,\rho^2/4f)$ and travelling a
distance $s=(4f^2+\rho^2)/(4f)$ along $\hat{\mathbf e}_r$ lands you at exactly
$(0,0,f)$ — the focus — for **every** pupil point.

**(b) Equal optical path (constant OPD).** Put the input reference plane at
$z=z_0$. The input leg (travelling $-\hat{\mathbf z}$) to the mirror has length
$L_{\text{in}}=z_0-z_p=z_0-\rho^2/4f$. The distance from the mirror to the focus is
$$
R=\lVert(0,0,f)-\mathbf r_m\rVert=\sqrt{\rho^2+\big(f-\tfrac{\rho^2}{4f}\big)^2}
 = f+\frac{\rho^2}{4f}.
$$
(The square root is a perfect square — expand it and see.) Therefore
$$
L_{\text{total}}=L_{\text{in}}+R=z_0+f=\text{constant, independent of }(x_p,y_p).
$$
A constant path means every part of the beam arrives at the focus **in phase** —
that is the definition of an aberration-free (stigmatic) focus. This is why an
*on-axis* parabola has no spherical aberration, unlike a thin lens.

### 4.2 Pupil radius ↔ ray angle: $\rho = 2f\tan(\theta/2)$

Let $\theta$ be the angle a reflected ray makes with the outgoing axis, so
$\cos\theta=\hat{\mathbf e}_r\!\cdot\!\hat{\mathbf u}$. For the aligned parent
$\hat{\mathbf u}=\hat{\mathbf z}$ and from the boxed $\hat{\mathbf e}_r$ above,
$$
\cos\theta=\frac{4f^2-\rho^2}{4f^2+\rho^2}.
$$
Now use the half-angle identities $1+\cos\theta=\tfrac{8f^2}{4f^2+\rho^2}$,
$1-\cos\theta=\tfrac{2\rho^2}{4f^2+\rho^2}$, and
$\sin\theta=\sqrt{(1-\cos\theta)(1+\cos\theta)}=\tfrac{4f\rho}{4f^2+\rho^2}$. Then
$$
\tan\frac{\theta}{2}=\frac{\sin\theta}{1+\cos\theta}=\frac{\rho}{2f}
\;\;\Longrightarrow\;\;
\boxed{\;\rho = 2f\tan\!\tfrac{\theta}{2}\;}
$$
This **pupil-to-angle map** is the parabola's analogue of the lens "sine
condition" $\rho=f\sin\theta$. It is the reason a parabola cannot be Fourier-
transformed with a plain FFT of the pupil (see §9): $\rho$ is *not* linear in
$\sin\theta$, hence not linear in transverse $k$.

### 4.3 Why not a quadratic lens phase?

A thin-lens phase $\exp(-ikr^2/2f)$ is the *paraxial* (small-angle) approximation
of a focusing element. This optic is fast ($\text{RFL}/D=2f_{\text{parent}}/D\approx1.2$), so paraxial is badly
violated. (Its NA depends on the definition: the catalogue value
$\sin\arctan(D/2\,\text{RFL})\approx0.39$; the real cone of the segment is asymmetric
and reaches ≈40° from the mean ray; a disc of the same diameter centred on the parent
axis has $\sin(2\arctan(D/4f))\approx0.72$.)

* the quadratic phase would inject spurious **spherical aberration** the real
  parabola does not have;
* it cannot represent the **90° fold**, the **longitudinal** field component
  $E_z$ that appears at high NA, or the aberrations produced by misalignment.

All three are visible in the figures and come out *for free* from the 3-D geometry.

---

## 5. From the mirror to the focal field: the Debye–Wolf picture

### 5.1 One plane wave per pupil point

Near a focus, a converging beam is well described as a bundle of plane waves — one
for each pupil point — all heading toward the focus. This is the **Debye
approximation**; it is accurate when the Fresnel number is large (aperture $\gg\lambda$,
focus many wavelengths away), which is overwhelmingly true here. Each pupil point
contributes a plane wave $\propto \exp(i\mathbf k_r\!\cdot\!\mathbf r)$ with

* **direction** $\hat{\mathbf e}_r$ (the reflected ray), so
  $\mathbf k_r = k\,\hat{\mathbf e}_r$, and
* **complex amplitude** = the (apodised, §6) reflected field on a reference sphere
  centred at the focus.

### 5.2 The focal field is a 2-D Fourier transform

Write $\mathbf k_r=(k_u,k_v,k_w)$ in the output basis, with
$k_u^2+k_v^2+k_w^2=k^2$ (it is a unit direction times $k$). A field point in the
focal plane is $\mathbf r=(0,v,w)$ in that basis (we set $u=0$ at the focus), so
$\mathbf k_r\!\cdot\!\mathbf r=k_v v + k_w w$. Summing the plane waves over the
cone of directions gives the **Debye–Wolf integral**
$$
\boxed{\;\mathbf E(v,w)=\iint P(k_v,k_w)\,e^{\,i(k_v v + k_w w)}\,dk_v\,dk_w\;}
$$
i.e. a 2-D inverse Fourier transform of an *angular spectrum* $P(k_v,k_w)$. All the
physics is in building $P$ correctly. The kernel sign is $+i$ (a **converging**
wave); this is fixed by physics, and the code checks it against a brute-force sum
to machine precision (§9).

For the aligned lab layout ($\hat{\mathbf u}=\hat{\mathbf x}$) this is exactly the
$E(Y,Z)=\iint P(k_y,k_z)e^{i(k_yY+k_zZ)}dk_y\,dk_z$ requested in the spec.

---

## 6. The two weights — keep them separate

$P$ is the reflected field mapped onto the reference sphere, times **two distinct
factors** that are easy to confuse. They must be kept apart (the code does).

### 6.1 Physical apodisation $q(\theta)$ from energy conservation

A flat input beam has intensity spread over a *flat* pupil; the converging beam
spreads the same power over a *cone* of angles. Energy conservation fixes how the
amplitude must scale. Power in a pupil annulus of radius $\rho$ equals power in the
corresponding angular ring:
$$
|a(\theta)|^2\,\underbrace{\sin\theta\,d\theta\,d\varphi}_{d\Omega}
=|E_{\text{in}}|^2\,\underbrace{\rho\,d\rho\,d\varphi}_{\text{pupil area}} .
$$
Here $a(\theta)$ is the amplitude on the reference sphere. Solve for it using
$\rho=2f\tan(\theta/2)$, hence $d\rho/d\theta=f\sec^2(\theta/2)$:
$$
|a|^2=|E_{\text{in}}|^2\,\frac{\rho}{\sin\theta}\frac{d\rho}{d\theta}
     =|E_{\text{in}}|^2\,f^2\sec^4\!\tfrac{\theta}{2}
\;\;\Longrightarrow\;\;
a = f\,\sec^2\!\tfrac{\theta}{2}\;E_{\text{in}} .
$$
(The algebra: $\rho\,d\rho/d\theta=2f^2\sin(\tfrac\theta2)/\cos^3(\tfrac\theta2)$ and
$\sin\theta=2\sin(\tfrac\theta2)\cos(\tfrac\theta2)$; the ratio is $f^2\sec^4(\tfrac\theta2)$.)
Dropping the global constant $f$, the **amplitude apodisation** is
$$
\boxed{\;q(\theta)=\sec^2\!\tfrac{\theta}{2}=\dfrac{2}{1+\cos\theta}\;}
$$
Sanity check: the *same* derivation with a lens ($\rho=f\sin\theta$) gives the
textbook aplanatic factor $\sqrt{\cos\theta}$ — so the method is trustworthy and
the parabola's factor is genuinely different.

### 6.2 The k-space Jacobian $d\Omega = dk_v\,dk_w/(k\,k_u)$

The Debye integral naturally uses the solid-angle measure $d\Omega=\sin\theta\,d\theta\,d\varphi$,
but the transform in §5.2 integrates over $dk_v\,dk_w$. Convert. With
$k_v=k\sin\theta\cos\varphi$, $k_w=k\sin\theta\sin\varphi$, the Jacobian is
$$
dk_v\,dk_w=\left|\frac{\partial(k_v,k_w)}{\partial(\theta,\varphi)}\right|d\theta\,d\varphi
=k^2\sin\theta\cos\theta\,d\theta\,d\varphi .
$$
(The determinant is $k^2\sin\theta\cos\theta$ — a short $2\times2$ computation.)
Hence, using $k_u=k\cos\theta$,
$$
\boxed{\;d\Omega=\sin\theta\,d\theta\,d\varphi=\dfrac{dk_v\,dk_w}{k^2\cos\theta}=\dfrac{dk_v\,dk_w}{k\,k_u}.\;}
$$

### 6.3 Putting them together

The Debye integral is $\mathbf E=\int a\,e^{i\mathbf k_r\cdot\mathbf r}\,d\Omega$
with $a=q(\theta)\,\mathbf E_r$ (apodised reflected field). Substituting the
Jacobian gives the transform of §5.2 with
$$
\boxed{\;P(k_v,k_w)=\dfrac{q(\theta)\,\mathbf E_r}{k\,k_u}\,e^{\,i\varphi_{\text{OPD}}}\;}
$$
where $\varphi_{\text{OPD}}$ is the optical-path phase (constant, hence droppable,
for the aligned case; the aberration for a tilted mirror — §10). **$q(\theta)$ is
physics (energy conservation); $1/k_u$ is bookkeeping (change of measure).** They
are different animals and the code stores them separately, exactly as the spec
demands.

Equivalent pupil-space form (used by the tilt path). Change the Debye integral
back to pupil variables. Using $\rho=2f\tan(\theta/2)$ one finds the neat result
$d\Omega=\cos^4(\theta/2)\,dx_p\,dy_p/f^2$, and since $q=\sec^2(\theta/2)$,
$$
q(\theta)\,d\Omega=\frac{\cos^2(\theta/2)}{f^2}\,dx_p\,dy_p ,
$$
so the whole angular-spectrum weight collapses to a **single factor
$\cos^2(\theta/2)$** per pupil sample:
$$
\boxed{\;\mathbf E(v,w)=\frac{1}{f^2}\iint \mathbf E_r(x_p,y_p)\,\cos^2\!\tfrac{\theta}{2}\;e^{i(k_v v+k_w w)}\,dx_p\,dy_p.\;}
$$

---

## 7. The vector field and the intensity

At each pupil point we build the incident vector field from a scalar amplitude and
a chosen polarisation $\hat{\mathbf p}$ (default $\hat{\mathbf y}$):
$\mathbf E_i=E_{\text{in}}(x_p,y_p)\,\hat{\mathbf p}$, reflect it with the mirror
formula (§3.2) to get $\mathbf E_r=(E_{r,x},E_{r,y},E_{r,z})$, weight it, and
Fourier-transform **each Cartesian component independently** — the kernel
$e^{i\mathbf k_r\cdot\mathbf r}$ is a scalar, so the three components do not mix in
the transform:
$$
E_x=\mathcal F\{P_x\},\quad E_y=\mathcal F\{P_y\},\quad E_z=\mathcal F\{P_z\}.
$$
The intensity keeps **all three**:
$$
\boxed{\;I(v,w)=|E_x|^2+|E_y|^2+|E_z|^2.\;}
$$
Even when the focal plane is the $y$–$z$ plane, the $x$ (longitudinal) component is
nonzero at high NA and contributes real intensity — this is the two-lobed $|E_z|$
pattern in the Gaussian figure and the on-axis longitudinal spot in the vortex
figure. A scalar or quadratic-phase model misses it entirely.

---

## 8. Numerical route A — pull-back (aligned parent) — `oap_prop.py`

Because $\rho$ is not linear in transverse $k$ (§4.2), we cannot FFT the pupil
arrays directly. The **well-conditioned** fix is a *pull-back*:

1. Lay down a **regular** $(k_v,k_w)$ grid over the captured cone.
2. Invert analytically to the pupil: from $q=\sqrt{(k_v/k)^2+(k_w/k)^2}=\sin\theta$,
   get $\rho=2f\,q/(1+\sqrt{1-q^2})=2f\tan(\theta/2)$, and the azimuth. **Azimuth
   subtlety:** the reflected transverse-$k$ of an axial beam points radially
   *inward*, so the pupil azimuth is $\varphi_{\text{pupil}}=\operatorname{atan2}(-k_w,-k_v)$,
   not $\operatorname{atan2}(k_w,k_v)$. Getting this wrong point-inverts the field
   and silently flips a vortex's topological charge.
3. Interpolate the **smooth input field** at those pupil points (interpolating the
   input on its own regular grid is stable; interpolating a phase-carrying field on
   scattered points is not).
4. Build $P_x,P_y,P_z$ with the weights of §6, then transform.

The transform $\mathbf E(v,w)=\sum P\,e^{i(k_v v+k_w w)}\,\Delta k_v\Delta k_w$ is
implemented as an **exact separable matrix-DFT** (two dense matrix products). It is
$O(N^3)$ — fine to ~$512^2$ — and its sign is $e^{+i\mathbf k\cdot\mathbf r}$ by
construction (the kernel is written out explicitly).
Replacing it with the host library's Bluestein CZT would give $O(N^2\log N)$ but
requires matching that routine's internal sign to this reference first.

---

## 9. Numerical route B — forward exact sum (tilt/decenter) — `oap_tilt.py`

When the mirror is tilted, the analytic inverse map of route A is no longer valid
(the pupil→$k$ mapping and the optical path both change). We switch to the
**forward** picture and the pupil-space weight $\cos^2(\theta/2)$ of §6.3:
$$
\mathbf E(v,w)=\sum_{\text{pupil}} \mathbf E_r\,\cos^2\!\tfrac{\theta}{2}\,e^{i\varphi_{\text{OPD}}}\;e^{i(k_v v+k_w w)}\,\Delta A .
$$
The scattered directions $(k_v,k_w)$ come straight from the real reflected rays.
This is a **non-uniform DFT**; it is evaluated exactly as two dense matrix products
(no scattered interpolation, no sign ambiguity), at cost $O(N_{\text{pupil}}N_{\text{out}})$.
On the aligned vortex it agrees with route A to a relative $L_2$ of order
$10^{-4}$–$10^{-3}$ after matching one global complex constant (the two routes drop
different constant prefactors); the residual depends on sampling, since route A
interpolates the input field. Not exact, but a sanity check worth keeping.

### 9.1 The optical-path (aberration) phase

For a general configuration the OPD is **not** constant, and its variation across
the pupil *is* the wavefront aberration. Along the ray from the mirror point
$\mathbf r_m$ in direction $\hat{\mathbf e}_r$,
$$
L=\underbrace{(\mathbf r_m-\mathbf p_{\text{ref}})\!\cdot\!\hat{\mathbf e}_i}_{\text{input leg }L_{\text{in}}}
 +\underbrace{\hat{\mathbf e}_r\!\cdot\!(\mathbf F-\mathbf r_m)}_{\text{leg to reference plane through }\mathbf F},
\qquad
\varphi_{\text{OPD}}=k\,(L-\langle L\rangle).
$$
The second term is the path from the mirror to the plane through the focus $\mathbf F$
perpendicular to the ray (the standard Debye reference). For the aligned parent
$L$ is constant (§4.1) and $\varphi_{\text{OPD}}=0$; when tilted, its variation across
the pupil is the aberration. Which terms dominate depends on the geometry and the
illumination, so read it off the computed wavefront rather than assuming it. Subtracting the mean $\langle L\rangle$ removes only the
irrelevant piston.

### 9.2 Where is the focus? Least-squares ray crossing

A tilted-mirror + fixed-input system has **no** exact stigmatic focus, so we take
the point $\mathbf F$ of closest approach to all reflected ray lines. The squared
distance from $\mathbf P$ to the line through $\mathbf r_m$ with direction
$\hat{\mathbf e}_r$ is
$(\mathbf P-\mathbf r_m)^\top(\mathbb 1-\hat{\mathbf e}_r\hat{\mathbf e}_r^\top)(\mathbf P-\mathbf r_m)$.
Summing and setting the gradient to zero gives a $3\times3$ linear system
$$
\Big[\sum_i(\mathbb 1-\hat{\mathbf e}_{r,i}\hat{\mathbf e}_{r,i}^\top)\Big]\mathbf F
=\sum_i(\mathbb 1-\hat{\mathbf e}_{r,i}\hat{\mathbf e}_{r,i}^\top)\,\mathbf r_{m,i} .
$$
We also set $\hat{\mathbf u}$ to the mean reflected direction, so the ~$2\alpha$
beam walk-off is removed from the display.

### 9.3 Why *not* the rigidly-transformed focus $R\,(0,0,f)+\mathbf t$

Tempting, but wrong. Rotating the *mirror* while the input beam stays along
$-\hat{\mathbf z}$ is **not** a rigid rotation of the whole system — it is
equivalent to illuminating the parabola off-axis, which has no stigmatic focus.
The chief ray reflected off the tilted vertex does *not* pass through
$R(0,0,f)+\mathbf t$. Using that point as the reference injects a huge spurious
tilt into $\varphi_{\text{OPD}}$ (hundreds of waves), which is a bug, not physics.
The least-squares focus removes most of it. It is a geometric estimate, though, so
$\varphi_{\text{OPD}}$ still contains some tilt and focus shift; §9.4 removes them
exactly before computing any figure of merit.

### 9.4 Wavefront statistics: RMS and Strehl

All notebooks use one function, `wavefront_stats`. Moving the observation point by
$(\delta v,\delta w,\delta u)$ multiplies each plane wave by
$e^{i(k_v\delta v+k_w\delta w+k_u\delta u)}$, so piston and the parts of
$\varphi_{\text{OPD}}$ linear in $(k_v,k_w,k_u)$ only *move* the spot (pointing and focus
shift); they do not blur it. They are removed by a least-squares fit weighted by $a^2$,
where $a=|E_{\text{in}}|\sqrt J$ is the ray amplitude in the Debye sum. From the residual
$\varphi_{\text{res}}$:
$$
\sigma=\frac{1}{2\pi}\sqrt{\frac{\sum a^2\varphi_{\text{res}}^2}{\sum a^2}},\qquad
S=\frac{\big|\sum a\,e^{i\varphi_{\text{res}}}\big|^2}{\big(\sum a\big)^2}.
$$
$S$ is exactly (in the scalar Debye picture) the intensity at the best point over the
peak of the unaberrated focus with the same input. It weights by amplitude and $\sigma$
by intensity, so $S\approx e^{-(2\pi\sigma)^2}$ holds only roughly for non-uniform
beams. The fitted $(\delta v,\delta w,\delta u)$ is returned too: a fixed detector sees
the spot move by that much in addition to the loss in $S$.

---

## 10. Sign and normalisation conventions

* **Forward propagation** in the host library uses $\exp(+ikz)$ (its
  `ASM_propagation` kernel), and its Fresnel `CZT_propagation` carries an $\exp(-i)$
  *transverse* kernel. The OAP output transform uses $\exp(+i(k_vv+k_ww))$ — the
  **converging-wave** sign, dictated by the physics of §5.2 and validated
  numerically. These are different operators; do not copy signs blindly between them.
* **Absolute scale.** With the energy-conserving weight $\sqrt{J}$, $J=d\Omega/dA$, the
  Debye–Wolf field is
  $\mathbf E=\frac{-i}{\lambda}\sum\mathbf E_r\sqrt J\,e^{i\varphi}e^{i(k_vv+k_ww)}\,\Delta A$
  (up to the constant phase $e^{ikf}$), so $|\mathbf E|^2$ is in the units of
  $|E_{\text{in}}|^2$. The step-by-step notebook includes the $-i/\lambda$; the other
  notebooks and `src/` drop it (and `src/` other constants), so their intensities are
  relative. Note: at high NA the power is **not** $\int|E|^2$ over the focal plane —
  compare in $k$-space with the $d\Omega$ measure, or use the axial Poynting component.
* **Input field.** $E_{\text{in}}(x,y)$ is the field in a plane just in front of the
  mirror, carried to the surface along the fixed direction $\hat{\mathbf e}_i$: each
  ray is sampled at its own $(x,y)$. Phase gradients of $E_{\text{in}}$ enter only as
  phase, not as ray directions, so the beam must be locally close to collimated along
  $\hat{\mathbf e}_i$. A field with strong phase gradients (e.g. after an SLM and a
  lens) must first be propagated to the mirror (e.g. angular-spectrum method).

---

## 11. Approximations, and what to make rigorous next

1. **Reflection model.** `src/` and the tilt notebook use the ideal conductor (§3.2);
   the step-by-step and perfect-vs-gold notebooks use Fresnel $r_s,r_p$ for a complex
   index.
2. **Parent-pupil sampling** in `src/oap_tilt.py`: the input is sampled at the parent
   coordinate of each mirror point instead of at the ray's own transverse coordinate.
   The notebooks trace each ray to the surface exactly (closed-form, cancellation-free
   quadratic root), so they do not make this approximation.
3. **Geometry.** `src/` models the on-axis parent only; the notebooks model the
   off-axis segment with a body-fixed circular aperture in the mirror-local $y_p$–$z_p$
   plane and rigid tilt/shift about a mount point.
4. **Input.** Collimated along $\hat{\mathbf e}_i$ (§10).
5. **Speed**: swap the exact $O(N^3)$ / $O(N_{\text{pupil}}N_{\text{out}})$
   transforms for the host Bluestein CZT (route A) or a NUFFT (route B).

---

## 12. Symbol table

| symbol | meaning |
|---|---|
| $f$ | parent focal length (7.5 mm), used in the surface equation |
| $\lambda,\;k=2\pi/\lambda$ | wavelength, wavenumber |
| $\mathbf r_p,\ \mathbf r_{\text{lab}}$ | parent-frame / lab-frame position |
| $R,\ \mathbf t$ | rigid-body rotation, translation ($\mathbf r_{\text{lab}}=R\mathbf r_p+\mathbf t$) |
| $\hat{\mathbf n}$ | unit surface normal |
| $\hat{\mathbf e}_i,\ \hat{\mathbf e}_r$ | incident / reflected ray unit directions |
| $\mathbf E_i,\ \mathbf E_r$ | incident / reflected vector field |
| $\rho=\sqrt{x_p^2+y_p^2}$ | pupil radius on the parent |
| $\theta,\varphi$ | polar/azimuth of a reflected ray about the output axis |
| $(\hat{\mathbf u},\hat{\mathbf v},\hat{\mathbf w})$ | output basis (axial, two transverse) |
| $k_u,k_v,k_w$ | components of $\mathbf k_r=k\hat{\mathbf e}_r$ on that basis |
| $q(\theta)=\sec^2(\theta/2)$ | physical amplitude apodisation |
| $d\Omega=dk_v dk_w/(k k_u)$ | solid-angle ↔ $k$-space measure |
| $P(k_v,k_w)$ | vector angular spectrum |
| $\varphi_{\text{OPD}}$ | optical-path (aberration) phase |

## References

* L. Novotny & B. Hecht, *Principles of Nano-Optics*, Ch. 3 (the focusing-field /
  angular-spectrum formalism your library's `focus_richards_wolf` follows).
* B. Richards & E. Wolf, *Proc. R. Soc. A* **253**, 358 (1959) — the original
  high-NA vectorial focus integral.
* M. Born & E. Wolf, *Principles of Optics* — Debye approximation, aberrations.