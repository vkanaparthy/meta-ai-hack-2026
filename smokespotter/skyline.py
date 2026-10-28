"""Fit a camera's orientation, focal length and distortion by aligning the
terrain skyline predicted from a DEM with the sky/terrain edge in the image."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares, minimize

from .camera import CameraModel

WORK_WIDTH = 1536          # images are analysed at this width


def edge_map(images: list[Image.Image]) -> tuple[np.ndarray, float]:
    """Sky-to-terrain edge strength (darker below = positive), on a median of
    several frames. Returns (map, scale) where scale = original / working width."""
    arrs = []
    for im in images:
        im = im.convert("RGB")
        s = im.width / WORK_WIDTH
        im = im.resize((WORK_WIDTH, round(im.height / s)), Image.BILINEAR)
        a = np.asarray(im, np.float32)
        lum = 0.3 * a[..., 0] + 0.5 * a[..., 1] + 0.2 * a[..., 2]
        blue = a[..., 2] - a[..., 0]                 # sky is much bluer than terrain
        arrs.append(lum + 0.5 * blue)
    a = gaussian_filter(np.median(np.stack(arrs), axis=0), 1.0)
    g = np.zeros_like(a)
    g[2:-2] = a[:-4] - a[4:]
    g = np.clip(g, 0, None)
    g /= np.percentile(g, 99.5) + 1e-6
    return np.tanh(g), s


@dataclass
class SkylineSamples:
    az: np.ndarray            # degrees
    el: np.ndarray            # degrees
    dist: np.ndarray          # metres


def predicted_rows(cam: CameraModel, sky: SkylineSamples, xs: np.ndarray) -> np.ndarray:
    x, y = cam.project(sky.az, sky.el)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return np.full(xs.shape, np.nan)
    order = np.argsort(x)
    x, y = x[order], y[order]
    yp = np.interp(xs, x, y, left=np.nan, right=np.nan)
    return np.where((yp >= 0) & (yp < cam.height), yp, np.nan)


def score(cam: CameraModel, sky: SkylineSamples, emap: np.ndarray) -> float:
    xs = np.arange(0, cam.width, 2, dtype=float)
    yp = predicted_rows(cam, sky, xs)
    ok = np.isfinite(yp)
    if ok.sum() < 0.5 * len(xs):
        return 0.0
    vals = map_coordinates(emap, [yp[ok], xs[ok]], order=1, mode="nearest")
    return float(vals.sum() / len(xs))


def _with(cam: CameraModel, p) -> CameraModel:
    return replace(cam, heading_deg=p[0] % 360, pitch_deg=p[1], roll_deg=p[2],
                   f_px=p[3], k1=p[4])


def fit(emap: np.ndarray, sky: SkylineSamples, base: CameraModel,
        heading_span: float = 25.0, pitch_span: float = 8.0,
        fix_lens: bool = False) -> tuple[CameraModel, dict]:
    """``base`` gives position, image size (working resolution) and starting guesses.
    fix_lens=True keeps base.f_px and base.k1 and fits orientation only."""
    maps = {s: gaussian_filter(emap, s) for s in (6.0, 3.0, 1.5, 0.75)}
    for m in maps.values():
        m /= m.max() + 1e-9

    # 1) coarse grid over heading and pitch
    cands = []
    for h in np.arange(base.heading_deg - heading_span, base.heading_deg + heading_span + 1e-9, 0.25):
        for p in np.arange(-pitch_span, pitch_span + 1e-9, 0.25):
            c = replace(base, heading_deg=h % 360, pitch_deg=p)
            cands.append((score(c, sky, maps[6.0]), h, p))
    cands.sort(reverse=True)
    starts = []
    for s_, h, p in cands:
        if all(abs(h - h2) > 1.5 or abs(p - p2) > 1.0 for _, h2, p2 in starts):
            starts.append((s_, h, p))
        if len(starts) == 5:
            break

    f0 = base.f_px
    lo = np.array([-np.inf, -15, -8, 0.75 * f0, -0.35])
    hi = np.array([np.inf, 15, 8, 1.35 * f0, 0.35])
    if fix_lens:
        lo[3], hi[3] = f0 - 1e-6, f0 + 1e-6
        lo[4], hi[4] = base.k1 - 1e-6, base.k1 + 1e-6

    def obj(p, m):
        if np.any(p < lo) or np.any(p > hi):
            return 1e3
        return -score(_with(base, p), sky, m)

    # 2) refine every start with all five parameters, coarse-to-fine
    best = None
    for _, h, p in starts:
        x = np.array([h, p, base.roll_deg, f0, base.k1])
        for s in (6.0, 3.0, 1.5, 0.75):
            steps = (np.array([1.0, 0.5, 0.5, 0.03 * f0, 0.05]) if s >= 3.0
                     else np.array([0.3, 0.15, 0.3, 0.015 * f0, 0.03]))
            if fix_lens:
                steps[3:] = 1e-9
            prev = np.inf
            for _ in range(4):           # restart until Nelder-Mead stops improving
                simplex = np.vstack([x, x + np.diag(steps)])
                r = minimize(obj, x, args=(maps[s],), method="Nelder-Mead",
                             options={"xatol": 0.002, "fatol": 1e-7, "maxiter": 3000,
                                      "initial_simplex": simplex})
                x = r.x
                if prev - r.fun < 1e-5:
                    break
                prev = r.fun
                steps = steps * 0.5
        val = -obj(x, maps[0.75])
        if best is None or val > best[0]:
            best = (val, x)

    cam = refine_lsq(_with(base, best[1]), sky, emap, lo, hi)
    return cam, quality(cam, sky, emap) | {"score": round(score(cam, sky, maps[0.75]), 4)}


def observed_rows(emap: np.ndarray, xs: np.ndarray, yp: np.ndarray, window: int,
                  min_edge: float = 0.3) -> np.ndarray:
    """Sub-pixel row of the strongest edge within +-window of each predicted row."""
    out = np.full(xs.shape, np.nan)
    for i, (x, y) in enumerate(zip(xs, yp)):
        if not np.isfinite(y):
            continue
        y0, y1 = int(max(1, y - window)), int(min(emap.shape[0] - 2, y + window))
        col = emap[y0:y1 + 1, int(x)]
        if col.size < 3 or col.max() < min_edge:
            continue
        k = int(np.argmax(col))
        if 0 < k < col.size - 1:            # parabola through the peak
            a, b, c = col[k - 1], col[k], col[k + 1]
            den = a - 2 * b + c
            k = k + (0.5 * (a - c) / den if den < 0 else 0.0)
        out[i] = y0 + k
    return out


def refine_lsq(cam: CameraModel, sky: SkylineSamples, emap: np.ndarray, lo, hi,
               windows=(16, 8, 4, 3)) -> CameraModel:
    """Iterated robust least squares on (predicted - observed) skyline rows."""
    xs = np.arange(0, cam.width, 2, dtype=float)
    p = np.array([cam.heading_deg, cam.pitch_deg, cam.roll_deg, cam.f_px, cam.k1])
    lo = np.where(np.isfinite(lo), lo, p - 90)
    hi = np.where(np.isfinite(hi), hi, p + 90)
    base = cam
    for w in windows:
        obs = observed_rows(emap, xs, predicted_rows(_with(base, p), sky, xs), w)
        ok = np.isfinite(obs)
        if ok.sum() < 20:
            break

        def resid(q):
            r = predicted_rows(_with(base, q), sky, xs[ok]) - obs[ok]
            return np.nan_to_num(r, nan=3.0 * w)

        r = least_squares(resid, np.clip(p, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), loss="soft_l1",
                          f_scale=1.5, x_scale=[0.1, 0.1, 0.1, 0.005 * cam.f_px, 0.01],
                          diff_step=[1e-4, 1e-3, 1e-3, 1e-4, 1e-2])
        p = r.x
    return _with(base, p)


def quality(cam: CameraModel, sky: SkylineSamples, emap: np.ndarray, window: int = 12) -> dict:
    """Pixel residual between the predicted skyline and the strongest nearby edge."""
    xs = np.arange(0, cam.width, 2, dtype=float)
    yp = predicted_rows(cam, sky, xs)
    res = []
    for x, y in zip(xs, yp):
        if not np.isfinite(y):
            continue
        y0, y1 = int(max(0, y - window)), int(min(emap.shape[0] - 1, y + window))
        col = emap[y0:y1 + 1, int(x)]
        if col.size and col.max() > 0.3:
            res.append(y0 + int(np.argmax(col)) - y)
    res = np.array(res)
    if res.size == 0:
        return {"median_abs_px": None, "edge_coverage": 0.0}
    return {"median_abs_px": round(float(np.median(np.abs(res))), 2),
            "edge_coverage": round(res.size / len(xs), 3)}


def fit_shared_lens(items: list[tuple[np.ndarray, SkylineSamples, CameraModel]],
                    windows=(12, 6, 4, 3), step: int = 4) -> tuple[dict, list[CameraModel]]:
    """Jointly fit one lens (focal length as a fraction of width, k1, k2) shared by
    all sequences, plus each sequence's heading/pitch/roll, by iterated robust
    least squares on skyline rows. ``items`` = (edge map, skyline, initial camera
    at working resolution)."""
    from scipy.sparse import lil_matrix

    n = len(items)
    cams = [c for _, _, c in items]
    fn0 = float(np.median([c.f_px / c.width for c in cams]))
    p = np.concatenate([[fn0, float(np.median([c.k1 for c in cams])), 0.0],
                        np.concatenate([[c.heading_deg, c.pitch_deg, c.roll_deg] for c in cams])])
    lo = np.concatenate([[0.6 * fn0, -0.6, -0.5], np.concatenate([[c.heading_deg - 20, -15, -8] for c in cams])])
    hi = np.concatenate([[1.6 * fn0, 0.6, 0.5], np.concatenate([[c.heading_deg + 20, 15, 8] for c in cams])])

    def cam_i(q, i):
        c = cams[i]
        return replace(c, f_px=q[0] * c.width, k1=q[1], k2=q[2],
                       heading_deg=q[3 + 3 * i] % 360, pitch_deg=q[4 + 3 * i], roll_deg=q[5 + 3 * i])

    xs = [np.arange(0, c.width, step, dtype=float) for c in cams]
    for w in windows:
        obs = []
        for i, (emap, sky, _) in enumerate(items):
            o = observed_rows(emap, xs[i], predicted_rows(cam_i(p, i), sky, xs[i]), w)
            obs.append(np.where(np.isfinite(o), o, np.nan))
        masks = [np.isfinite(o) for o in obs]
        sizes = [int(m.sum()) for m in masks]
        if sum(sizes) < 50:
            break

        def resid(q):
            out = []
            for i, (emap, sky, _) in enumerate(items):
                r = predicted_rows(cam_i(q, i), sky, xs[i][masks[i]]) - obs[i][masks[i]]
                out.append(np.nan_to_num(r, nan=3.0 * w))
            return np.concatenate(out)

        sparsity = lil_matrix((sum(sizes), len(p)), dtype=int)
        row = 0
        for i, k in enumerate(sizes):
            sparsity[row:row + k, 0:3] = 1
            sparsity[row:row + k, 3 + 3 * i:6 + 3 * i] = 1
            row += k
        r = least_squares(resid, np.clip(p, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), loss="soft_l1",
                          f_scale=1.5, jac_sparsity=sparsity,
                          x_scale=np.concatenate([[0.002, 0.01, 0.01], np.tile([0.1, 0.1, 0.1], n)]),
                          diff_step=np.concatenate([[1e-5, 1e-3, 1e-3], np.tile([1e-5, 1e-3, 1e-3], n)]))
        p = r.x
    lens = {"f_norm": float(p[0]), "k1": float(p[1]), "k2": float(p[2]),
            "hfov_deg": float(np.degrees(2 * np.arctan(0.5 / p[0])))}
    return lens, [cam_i(p, i) for i in range(n)]
