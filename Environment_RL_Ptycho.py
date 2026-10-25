#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generated with Claude Opus 5 via AI Tennessee
Link: https://drive.google.com/open?id=1ynQJUX-0APiMBXyMmXZF8hryd-Vj9RRO&usp=drive_fs

RL environment for *blind ptychographic phase retrieval*.

Drop-in analogue of Environment_RL.py (CT / radon):

      CT                                  ptychography
      ----------------------------------  --------------------------------------
      action  = projection angle theta    action  = scanning window S_j (a corner)
      data    = sinogram row              data    = d_j = |F(w .* S_j z)|^2
      solver  = iradon_sart (linear)      solver  = scipy.optimize on
                                                    min_{w,z} sum_j c(|F(w.*S_j z)|^2, d_j)
      state   = reconstructed image       state   = reconstructed complex object
      reward  = -||forward(state) - d||   same, in the phaseless metric

The agent sequentially picks N windows out of `action_size` candidate windows;
after every pick the blind problem is re-solved (warm-started) and the
reconstruction is handed back as the belief state.

Everything about S_j lives in `ptychography.py`: S_j z is a slice, S_j^T is a
scatter-add, and a scan is just a (J x 2) integer array of window corners.
"""

import math
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize

from ptychography import (
    _idx, coverage, extract, scatter, forward, measure, gaussian_probe, align,
)

# ----------------------------------------------------------------------
# optional project imports (fall back to local definitions when running alone)
# ----------------------------------------------------------------------
try:
    from utils import ImageType, RewardType
except Exception:                                            # pragma: no cover
    from enum import Enum

    class ImageType(Enum):
        MIXED = 1; TRIANGLE = 2; TRIANGLE_THIRTY = 3; TRIANGLE_FIXED = 4
        LINE = 5; LINE_THIRTY = 6; LINE_FIXED = 7

    class RewardType(Enum):
        PNSR_E2E = 1; PNSR_INCREMENTAL = 2; PNSR_ADD = 3
        FWD_E2E = 4; FWD_INCREMENTAL = 5
        FWD_RND_1 = 6; FWD_RND_10 = 7
        FWD_RND_COMPLETE_10 = 8; FWD_RND_COMPLETE_FULL = 9
        FWD_INC_RND_1 = 10; FWD_INC_RND_10 = 11
        FWD_RND_1_REG = 12

try:
    from PhantomGenerator import PhantomGenerator
except Exception:                                            # pragma: no cover
    PhantomGenerator = None


# ======================================================================
# 1. penalty functions c(a, b),  a = |F(w .* S_j z)|^2  (model intensity)
#                                b = d_j                 (measured intensity)
#    each returns (value, dc/da).  dc/da is all the solver needs.
# ======================================================================

_EPS = 1e-12


def _c_intensity(a, b):
    """Gaussian / least-squares on intensities: (a - b)^2."""
    r = a - b
    return r ** 2, 2.0 * r


def _c_amplitude(a, b):
    """Amplitude least squares: (sqrt(a) - sqrt(b))^2.  Best conditioned."""
    sa, sb = np.sqrt(a + _EPS), np.sqrt(b)
    return (sa - sb) ** 2, 1.0 - sb / sa


def _c_poisson(a, b):
    """Poisson / shot-noise MLE (Kullback-Leibler): a - b log a."""
    aa = a + _EPS
    return aa - b * np.log(aa), 1.0 - b / aa


def _c_huber(a, b, delta=1.0):
    """Huber on the amplitude residual -- robust to outlier pixels."""
    r = np.sqrt(a + _EPS) - np.sqrt(b)
    big = np.abs(r) > delta
    val = np.where(big, delta * (2 * np.abs(r) - delta), r ** 2)
    dcdr = np.where(big, 2 * delta * np.sign(r), 2 * r)
    return val, dcdr * 0.5 / np.sqrt(a + _EPS)


PENALTIES = {
    "intensity": _c_intensity,
    "amplitude": _c_amplitude,
    "poisson":   _c_poisson,
    "huber":     _c_huber,
}


# ======================================================================
# 2. objective + analytic gradient, packed into a real vector for scipy
# ======================================================================

def _pack(z, w):
    return np.concatenate([z.real.ravel(), z.imag.ravel(),
                           w.real.ravel(), w.imag.ravel()])


def _unpack(x, n, m):
    k = n * n
    z = (x[:k] + 1j * x[k:2 * k]).reshape(n, n)
    w = (x[2 * k:2 * k + m * m] + 1j * x[2 * k + m * m:]).reshape(m, m)
    return z, w


def objective(x, d2, pos, n, m, penalty="amplitude", periodic=True,
              fix_probe=None, reg=0.0, normalize=True):
    """f(z, w) = sum_j c(|F(w .* S_j z)|^2, d_j)   and its real gradient.

    Wirtinger calculus:  with y = F(w .* S_j z) and g_j = F^H( c_a(|y|^2) * y ),
        df/d(conj z) = sum_j S_j^T ( conj(w) * g_j )          <- scatter
        df/d(conj w) = sum_j conj(S_j z) * g_j                <- sum over j
    and the gradient w.r.t. the stacked real vector is 2 * df/d(conj .).
    """
    z, w = _unpack(x, n, m)
    if fix_probe is not None:
        w = fix_probe

    psi = extract(z, pos, m, periodic) * w[None]           # w .* S_j z
    y = np.fft.fft2(psi, norm="ortho")
    a = np.abs(y) ** 2

    cval, ca = PENALTIES[penalty](a, d2)
    f = float(cval.sum())

    g = np.fft.ifft2(ca * y, norm="ortho")                 # F^H (c_a * y)
    gz = scatter(np.conj(w)[None] * g, pos, n, periodic)
    gw = np.sum(np.conj(extract(z, pos, m, periodic)) * g, axis=0)

    if reg:                                                # Tikhonov, keeps the
        f += reg * (np.vdot(z, z).real + np.vdot(w, w).real)   # never-scanned
        gz = gz + reg * z                                      # pixels bounded
        gw = gw + reg * w

    if fix_probe is not None:
        gw = np.zeros_like(gw)

    scale = 1.0 / max(d2.size, 1) if normalize else 1.0
    grad = 2.0 * _pack(gz, gw) * scale
    return f * scale, grad


# ======================================================================
# 3. generic scipy solver wrapper -- swap `method` freely
# ======================================================================

# methods that accept an analytic jacobian and scale to 1e4+ variables
SCIPY_METHODS = ("L-BFGS-B", "CG", "BFGS", "TNC", "Newton-CG", "trust-constr", "Powell")


def solve_blind(d, pos, n, m, z0=None, w0=None, penalty="amplitude",
                method="L-BFGS-B", maxiter=200, periodic=True, reg=1e-8,
                fix_probe=False, options=None, callback=None):
    """min_{w,z} sum_j c(|F(w .* S_j z)|^2, d_j) with scipy.optimize.minimize.

    d : (J, m, m) measured *magnitudes* (as returned by ptychography.measure)
    returns (z_hat, w_hat, scipy OptimizeResult)
    """
    d2 = d.astype(float) ** 2                              # intensities
    if z0 is None:
        z0 = np.ones((n, n), dtype=complex)
    if w0 is None:
        w0 = gaussian_probe(m, 0.35, 0.0).astype(complex)

    probe = w0.copy() if fix_probe else None
    args = (d2, np.asarray(pos, int), n, m, penalty, periodic, probe, reg)

    # scipy's default ftol stops L-BFGS-B absurdly early on this objective
    opt = {"maxiter": maxiter}
    if method in ("L-BFGS-B", "TNC"):
        opt.update(maxfun=10 * maxiter, ftol=1e-16, gtol=1e-14)
    elif method in ("CG", "BFGS", "Newton-CG"):
        opt["gtol"] = 1e-12
    if options:
        opt.update(options)

    res = minimize(objective, _pack(z0, w0), args=args, jac=True,
                   method=method, options=opt, callback=callback)

    z, w = _unpack(res.x, n, m)
    if fix_probe:
        w = w0
    else:                                                  # fix the (cz, w/c)
        s = la.norm(w0) / (la.norm(w) + _EPS)              # scale ambiguity
        w, z = w * s, z / s
    return z, w, res


# ======================================================================
# 4. action space: the candidate scanning windows S_j
# ======================================================================

def candidate_windows(n, m, action_size, periodic=True, jitter=0, seed=0):
    """`action_size` candidate window corners on a (roughly) square grid.

    Action a  <->  S_{pos[a]}  <->  the slice z[r:r+m, c:c+m].
    A small jitter breaks the perfectly-periodic raster degeneracy that makes
    the *blind* problem ill-posed.
    """
    k = int(round(math.sqrt(action_size)))
    if k * k != action_size:
        raise ValueError("action_size must be a perfect square (grid of windows)")
    hi = n if periodic else n - m + 1
    g = np.unique(np.linspace(0, hi - 1, k).round().astype(int))
    pos = np.array([(r, c) for r in g for c in g], dtype=int)
    if jitter:
        rng = np.random.default_rng(seed)
        pos += rng.integers(-jitter, jitter + 1, pos.shape)
        pos = np.mod(pos, n) if periodic else np.clip(pos, 0, n - m)
    return pos


def psnr(target, ref):
    diff = (ref - target).ravel()
    rmse = math.sqrt(np.mean(np.abs(diff) ** 2))
    return 20 * math.log10(1.0 / max(rmse, _EPS))


# ======================================================================
# 5. the environment
# ======================================================================

class env():
    """Same API as the CT env: reset() -> state, step(action) -> tuple."""

    def __init__(
        self, seed, n_images, num_scans, image_size, action_size,
        probe_size, corruption_photons, pnsr_reg,
        image_type, reward_type,
        penalty, solver, solver_iters,
        periodic, jitter, allow_repeats,
        state_mode, warm_start, blind,
    ):
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)

        self.image_size = image_size            # n
        self.probe_size = probe_size            # m
        self.num_scans = num_scans              # N = episode length
        self.action_size = action_size
        self.periodic = periodic
        self.penalty = penalty
        self.solver = solver
        self.solver_iters = solver_iters
        self.reward_type = reward_type
        self.pnsr_reg = pnsr_reg
        self.allow_repeats = allow_repeats
        self.state_mode = state_mode
        self.warm_start = warm_start
        self.blind = blind                      # False -> probe known (easier)
        self.corruption_photons = corruption_photons
        self.init_start = 0
        self.img_channels = 1 + (self.state_mode != "amp") # only amp has 1 channels

        # --- objects of interest (complex: amplitude + phase from a phantom)
        self.P_all = self._make_objects(seed, n_images, image_size, image_type)
        self.w_true = gaussian_probe(probe_size)

        # --- the action space: one S_j per action
        self.positions = candidate_windows(image_size, probe_size, action_size,
                                           periodic, jitter, seed)

        self.n = self.rng.integers(0, len(self.P_all))
        self.reset()

    # ---------------- objects ----------------
    def _make_objects(self, seed, n_images, n, image_type):
        if PhantomGenerator is not None:
            gen = PhantomGenerator(seed=seed, image_size=n)
            if image_type == ImageType.MIXED:
                P = gen.generate_mixed(n_samples=n_images)
            elif image_type == ImageType.TRIANGLE:
                P = gen.generate_triangle(n_samples=n_images)
            elif image_type == ImageType.TRIANGLE_THIRTY:
                P = gen.generate_triangle(n_samples=n_images, max_angle=30)
            elif image_type == ImageType.TRIANGLE_FIXED:
                P = gen.generate_triangle(n_samples=n_images, max_angle=0)
            elif image_type == ImageType.LINE:
                P = gen.generate_line(n_samples=n_images)
            elif image_type == ImageType.LINE_THIRTY:
                P = gen.generate_line(n_samples=n_images, max_angle=30)
            elif image_type == ImageType.LINE_FIXED:
                P = gen.generate_line(n_samples=n_images, max_angle=0)
            elif image_type == ImageType.SKIMAGE:
                P = gen.generate_skimage(n_samples=n_images)
            else:
                raise Exception("Unknown ImageType %s" % image_type)
            P = np.asarray(P, dtype=float)
        else:                                   # standalone fallback
            P = np.stack([self._smooth_field(n) for _ in range(n_images)])

        # phantom -> complex transmission function:  A e^{i phi}
        return (1.0 - 0.5 * P) * np.exp(1j * np.pi * (P - 0.5))

    def _smooth_field(self, n, bw=0.08):
        A = np.fft.fft2(self.rng.normal(size=(n, n)))
        f = np.fft.fftfreq(n)
        g = np.exp(-(f[:, None] ** 2 + f[None, :] ** 2) / (2 * bw ** 2))
        o = np.real(np.fft.ifft2(A * g))
        return (o - o.min()) / (o.max() - o.min() + _EPS)

    # ---------------- gym-ish API ----------------
    def reset(self):
        self.n = int(self.rng.integers(0, len(self.P_all)))
        self.a_start = self.init_start
        self.curr_iteration = 0
        self.pos_seq = []                      # list of chosen S_j corners
        self.action_seq = []
        self.d_seq = []                        # list of (m,m) magnitude frames
        self.previous_reward = 0.0
        self.reward = 0.0
        self.done = False
        self.z_hat = np.ones((self.image_size,) * 2, dtype=complex)
        self.w_hat = gaussian_probe(self.probe_size, 0.35, 0.0).astype(complex)
        self.state = self._state_from(self.z_hat * 0)
        return self.state

    def valid_action_mask(self):
        mask = np.ones(len(self.positions), dtype=bool)
        if not self.allow_repeats:
            mask[self.action_seq] = False
        return mask

    def step(self, action):
        action = int(action)
        self.a_start += 1
        self.action_seq.append(action)
        self.window_action = self.positions[action]
        self.pos_seq.append(self.window_action)

        # --- acquire the phaseless frame d_j for the chosen window
        self.d_seq.append(self._acquire(self.window_action))
        d = np.stack(self.d_seq)
        pos = np.asarray(self.pos_seq, int)

        # --- re-solve the blind problem on everything measured so far
        z0 = self.z_hat if self.warm_start else None
        w0 = self.w_hat if self.warm_start else None
        self.z_hat, self.w_hat, self.opt_res = solve_blind(
            d, pos, self.image_size, self.probe_size, z0=z0, w0=w0,
            penalty=self.penalty, method=self.solver, maxiter=self.solver_iters,
            periodic=self.periodic, fix_probe=not self.blind,
        )
        self.state = self._state_from(self.z_hat)

        self.reward = self._get_reward(d, pos)

        self._final_pos = pos            # kept for info() after the reset below

        if self.a_start > self.num_scans:
            self.a_start = self.init_start
            self.pos_seq, self.action_seq, self.d_seq = [], [], []
            self.done = True

        return (np.array(self.state), self.reward, self.done,
                self.window_action, self.pos_seq, self.n)

    # ---------------- measurement ----------------
    def _acquire(self, corner):
        """d_j: one phaseless diffraction pattern (magnitudes), with shot noise."""
        return measure(self.P_all[self.n], self.w_true, np.array([corner]),
                       self.periodic, photons=self.corruption_photons,
                       seed=int(self.rng.integers(1 << 31)))[0]

    def _state_from(self, z):
        if self.state_mode == "amp":
            return np.abs(z).astype(np.float32)
        if self.state_mode == "ampphase":
            return np.stack([np.abs(z), np.angle(z) / np.pi]).astype(np.float32)
        return np.stack([z.real, z.imag]).astype(np.float32)   # "reim"

    # ---------------- reward ----------------
    def _fwd_error(self, pos, d):
        """-|| |F(w_hat .* S_j z_hat)| - d || / ||d||   (phaseless data misfit)."""
        err = la.norm(np.abs(forward(self.z_hat, self.w_hat, np.asarray(pos, int),
                                     self.periodic)) - d)
        return -err / max(la.norm(d), _EPS)

    def _rnd_positions(self, n_scans):
        idx = self.rng.choice(len(self.positions),
                              size=min(n_scans, len(self.positions)), replace=False)
        return self.positions[idx]

    def _get_reward(self, d, pos):
        reward = 0.0
        rt, R = self.reward_type, RewardType

        if rt in (R.PNSR_E2E, R.PNSR_INCREMENTAL, R.PNSR_ADD):
            z_ref = self.P_all[self.n]
            current_reward = psnr(z_ref, align(self.z_hat, z_ref))
            if rt == R.PNSR_E2E and self.a_start > self.num_scans:
                reward = current_reward
            elif rt == R.PNSR_INCREMENTAL:
                reward = current_reward - self.previous_reward
                self.previous_reward = current_reward
            elif rt == R.PNSR_ADD:
                reward = current_reward / float(self.num_scans)

        elif rt in (R.FWD_E2E, R.FWD_INCREMENTAL):
            current_reward = self._fwd_error(pos, d)
            if rt == R.FWD_E2E and self.a_start > self.num_scans:
                reward = current_reward
            elif rt == R.FWD_INCREMENTAL:
                reward = current_reward - self.previous_reward
                self.previous_reward = current_reward

        elif rt in (R.FWD_RND_1, R.FWD_RND_10, R.FWD_RND_COMPLETE_10,
                    R.FWD_RND_COMPLETE_FULL, R.FWD_INC_RND_1, R.FWD_INC_RND_10):
            # "does the reconstruction predict windows we never scanned?"
            if rt in (R.FWD_RND_COMPLETE_10, R.FWD_RND_COMPLETE_FULL) \
                    and self.a_start < self.num_scans:
                return reward
            n_scans = 1 if rt in (R.FWD_RND_1, R.FWD_INC_RND_1) else 10
            rnd_pos = (self.positions if rt == R.FWD_RND_COMPLETE_FULL
                       else self._rnd_positions(n_scans))
            rnd_d = np.stack([self._acquire(p) for p in rnd_pos])
            current_reward = self._fwd_error(rnd_pos, rnd_d)

            if rt in (R.FWD_RND_1, R.FWD_RND_10):
                reward = current_reward / float(self.num_scans)
            elif rt in (R.FWD_INC_RND_1, R.FWD_INC_RND_10):
                reward = current_reward - self.previous_reward
                self.previous_reward = current_reward
            else:
                reward = current_reward

        elif rt == R.FWD_RND_1_REG:
            rnd_pos = self._rnd_positions(1)
            rnd_d = np.stack([self._acquire(p) for p in rnd_pos])
            z_ref = self.P_all[self.n]
            reward = (self._fwd_error(rnd_pos, rnd_d)
                      + self.pnsr_reg * psnr(z_ref, align(self.z_hat, z_ref)))
            reward /= float(self.num_scans)

        else:
            raise Exception("Unknown reward_type %s" % rt)

        return reward

    # ---------------- diagnostics ----------------
    def info(self):
        pos = (np.asarray(self.pos_seq, int) if self.pos_seq
               else getattr(self, "_final_pos", np.zeros((0, 2), int)))
        cov = (coverage(self.image_size, self.probe_size, pos, self.periodic)
               if len(pos) else np.zeros((self.image_size,) * 2, int))
        z_ref = self.P_all[self.n]
        seen = cov > 0
        err = (la.norm((align(self.z_hat, z_ref) - z_ref)[seen])
               / max(la.norm(z_ref[seen]), _EPS)) if seen.any() else np.nan
        return {"n_scans": len(pos), "min_coverage": int(cov.min()),
                "covered_frac": float(seen.mean()), "rel_err_z_seen": float(err)}


# ======================================================================
# 6. self-tests / demo
# ======================================================================

def _grad_check():
    rng = np.random.default_rng(0)
    n, m = 12, 6
    pos = candidate_windows(n, m, 9, True, 1, 0)
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    w = gaussian_probe(m)
    d2 = measure(z, w, pos) ** 2
    for p in PENALTIES:
        x = _pack(z + 0.3 * rng.normal(size=(n, n)), w * 1.1)
        f0, g = objective(x, d2, pos, n, m, penalty=p)
        num = np.zeros_like(g)
        for i in rng.choice(len(x), 40, replace=False):
            e = np.zeros_like(x); e[i] = 1e-6
            num[i] = (objective(x + e, d2, pos, n, m, penalty=p)[0]
                      - objective(x - e, d2, pos, n, m, penalty=p)[0]) / 2e-6
        k = num != 0
        print(f"  grad check [{p:>9}]  rel err = "
              f"{la.norm(num[k] - g[k]) / la.norm(num[k]):.2e}")


if __name__ == "__main__":
    print("gradient checks")
    _grad_check()

    print("\nenvironment rollout (random policy)")
    e = env(seed=0, n_images=3, num_scans=24, image_size=32, action_size=64,
            probe_size=12, corruption_photons=1e8, pnsr_reg=0.0,
            image_type=ImageType.MIXED,
            reward_type=RewardType.FWD_INCREMENTAL,
            penalty="amplitude", solver="L-BFGS-B", solver_iters=80, 
            periodic=True, jitter=1, allow_repeats=False,
            state_mode="reim", warm_start=True, blind=True)

    s = e.reset()
    print(f"  state {s.shape}   actions {len(e.positions)}   episode {e.num_scans}")
    done, t = False, 0
    while not done:
        a = int(np.random.choice(np.flatnonzero(e.valid_action_mask())))
        s, r, done, win, seq, idx = e.step(a)
        t += 1
        if t % 6 == 0 or done:
            i = e.info()
            print(f"  t={t:3d}  S_j=({win[0]:2d},{win[1]:2d})  r={r:+.4f}  "
                  f"cov={i['covered_frac']:.2f}  relerr_z={i['rel_err_z_seen']:.3f}")

    print("\nsolver comparison on a fixed good scan (non-blind vs blind)")
    from ptychography import scan_positions
    n, m = 32, 12
    pos = scan_positions(n, m, 4, True, jitter=1)
    z_true = e.P_all[0]
    d = measure(z_true, e.w_true, pos, photons=None)
    for method, blind in (("L-BFGS-B", False), ("L-BFGS-B", True), ("CG", True)):
        zh, wh, res = solve_blind(d, pos, n, m, penalty="amplitude",
                                  method=method, maxiter=400,
                                  w0=e.w_true if not blind else None,
                                  fix_probe=not blind)
        ez = la.norm(align(zh, z_true) - z_true) / la.norm(z_true)
        print(f"  {method:>9}  blind={blind!s:>5}  f={res.fun:.2e}  "
              f"nit={res.nit:4d}  rel.err z={ez:.2e}")
