"""
Generated with Claude Opus 5 via AI Tennessee
Link: https://drive.google.com/open?id=1ynQJUX-0APiMBXyMmXZF8hryd-Vj9RRO&usp=drive_fs

Blind ptychographic phase retrieval.

Model:   d_j = |F (w .* S_j z)|,  j = 1..J
  z  in C^{n x n}   object
  w  in C^{m x m}   probe (localized illumination)
  S_j in R^{m^2 x n^2}  binary crop: picks the m x m window at corner (r_j, c_j)
  F   2D DFT (unitary, norm="ortho")

S_j is NEVER formed as a matrix: S_j z == z[r_j:r_j+m, c_j:c_j+m]  (a slice),
and S_j^T == scatter-add of an m x m patch back into an n x n grid.
The whole scan geometry is just the J x 2 array of corner positions.

Coverage condition (every pixel seen at least once):
    for each i there is a j with ||S_j e_i||_1 = 1
    <=>  sum_j diag(S_j^T S_j) >= 1 entrywise   <=>  coverage(...).min() >= 1
A raster scan with stride <= m satisfies it; overlap ratio 1 - stride/m should
be >= ~0.6 for the blind problem to be well posed in practice.
"""

import numpy as np

# ----------------------------------------------------------------------
# scan geometry  (this *is* the definition of the S_j's)
# ----------------------------------------------------------------------

def scan_positions(n, m, stride, periodic=True, jitter=0, seed=0):
    """J x 2 array of top-left corners of the scanning windows.

    periodic=True : windows wrap around (torus). Clean adjoint, uniform
                    coverage, no boundary pixels with weight 1.
    periodic=False: windows stay inside the array; the last row/col is
                    snapped to n-m so the border is still covered.
    jitter        : random +-jitter pixel perturbation. Breaks the "raster
                    grid pathology" (an affine-phase ambiguity that appears
                    for perfectly periodic raster scans).
    """
    if periodic:
        g = np.arange(0, n, stride)
    else:
        if stride > m:
            raise ValueError("stride > m leaves uncovered pixels")
        g = np.unique(np.append(np.arange(0, n - m + 1, stride), n - m))
    pos = np.array([(r, c) for r in g for c in g], dtype=int)
    if jitter:
        rng = np.random.default_rng(seed)
        pos = pos + rng.integers(-jitter, jitter + 1, pos.shape)
        pos = np.mod(pos, n) if periodic else np.clip(pos, 0, n - m)
    return pos


def _idx(r, c, m, n, periodic):
    if periodic:
        return np.ix_(np.mod(np.arange(r, r + m), n),
                      np.mod(np.arange(c, c + m), n))
    return np.ix_(np.arange(r, r + m), np.arange(c, c + m))


def coverage(n, m, pos, periodic=True):
    """diag(sum_j S_j^T S_j): how many windows see each pixel. Must be >= 1."""
    cov = np.zeros((n, n), dtype=int)
    for r, c in pos:
        cov[_idx(r, c, m, n, periodic)] += 1
    return cov


def build_S(n, m, r, c, periodic=True):
    """Explicit m^2 x n^2 binary matrix. ONLY for unit tests / tiny n."""
    S = np.zeros((m * m, n * n))
    for a in range(m):
        for b in range(m):
            i = (r + a) % n if periodic else r + a
            j = (c + b) % n if periodic else c + b
            S[a * m + b, i * n + j] = 1.0
    return S


# ----------------------------------------------------------------------
# linear operators
# ----------------------------------------------------------------------

def extract(z, pos, m, periodic=True):
    """stack of S_j z,  shape (J, m, m)"""
    n = z.shape[0]
    return np.stack([z[_idx(r, c, m, n, periodic)] for r, c in pos])


def scatter(patches, pos, n, periodic=True):
    """sum_j S_j^T p_j  -- adjoint of extract. np.add.at handles overlaps."""
    m = patches.shape[-1]
    out = np.zeros((n, n), dtype=complex)
    for j, (r, c) in enumerate(pos):
        np.add.at(out, _idx(r, c, m, n, periodic), patches[j])
    return out


def forward(z, w, pos, periodic=True):
    """complex exit waves F(w .* S_j z), shape (J, m, m)"""
    return np.fft.fft2(extract(z, pos, w.shape[0], periodic) * w[None],
                       norm="ortho")


def measure(z, w, pos, periodic=True, photons=None, seed=0):
    """Phaseless data (magnitudes). photons=None -> noiseless;
    otherwise Poisson shot noise with ~`photons` counts per diffraction pattern.
    This is the ptychography analogue of the `percentage` noise in radon."""
    I = np.abs(forward(z, w, pos, periodic)) ** 2
    if photons is None:
        return np.sqrt(I)
    rng = np.random.default_rng(seed)
    s = photons / I.sum(axis=(1, 2)).mean()
    return np.sqrt(rng.poisson(I * s) / s)


def residual(z, w, d, pos, periodic=True):
    return (np.linalg.norm(np.abs(forward(z, w, pos, periodic)) - d)
            / np.linalg.norm(d))


# ----------------------------------------------------------------------
# probe model
# ----------------------------------------------------------------------

def gaussian_probe(m, sigma_frac=0.25, curvature=1.5):
    """Gaussian amplitude x quadratic (defocus) phase -- a realistic w."""
    g = (np.arange(m) - (m - 1) / 2) / m
    X, Y = np.meshgrid(g, g, indexing="ij")
    R2 = X ** 2 + Y ** 2
    return np.exp(-R2 / (2 * sigma_frac ** 2)) * np.exp(1j * curvature * np.pi * R2)


# ----------------------------------------------------------------------
# solver: ePIE (stochastic alternating projection)
# ----------------------------------------------------------------------

def epie(d, pos, n, w0, periodic=True, iters=300, az=0.9, aw=0.3,
         update_probe_after=20, seed=0, callback=None):
    """Extended ptychographic iterative engine.

    Per window j:
        psi   = (S_j z) .* w
        Psi   = F psi ;  Psi <- d_j * Psi/|Psi|          (magnitude projection)
        dpsi  = F^H Psi - psi
        z[win] += az * conj(w)/max|w|^2 * dpsi
        w      += aw * conj(S_j z)/max|S_j z|^2 * dpsi

    `update_probe_after` keeps w frozen at w0 early on; releasing it
    immediately from a bad guess usually stalls.
    """
    rng = np.random.default_rng(seed)
    m = d.shape[-1]
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    w = w0.astype(complex).copy()
    hist = []
    for it in range(iters):
        for j in rng.permutation(len(d)):
            sl = _idx(pos[j, 0], pos[j, 1], m, n, periodic)
            zp = z[sl]
            psi = zp * w
            Psi = np.fft.fft2(psi, norm="ortho")
            Psi = d[j] * np.exp(1j * np.angle(Psi))
            dpsi = np.fft.ifft2(Psi, norm="ortho") - psi
            z[sl] = zp + az * np.conj(w) / (np.abs(w).max() ** 2 + 1e-12) * dpsi
            if it >= update_probe_after:
                w = w + aw * np.conj(zp) / (np.abs(zp).max() ** 2 + 1e-12) * dpsi
        hist.append(residual(z, w, d, pos, periodic))
        if callback:
            callback(it, z, w, hist[-1])
    return z, w, np.array(hist)


def align(x, x_ref):
    """Remove the (z,w) -> (c z, w/c) scale + global phase ambiguity before
    comparing to ground truth. Least-squares best complex scalar."""
    a = np.vdot(x.ravel(), x_ref.ravel()) / np.vdot(x.ravel(), x.ravel())
    return a * x