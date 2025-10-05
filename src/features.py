from typing import Dict, Optional

import numpy as np


def nanpercentile(a: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(a, q))


def robust_scale(a: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Robustly scale array to roughly [-1, 1] using percentiles.

    Values are clipped to [p_low, p_high] percentiles, then centered at median and scaled by IQR-like span.
    """
    a = np.asarray(a, dtype=float)
    lo = nanpercentile(a, p_low)
    hi = nanpercentile(a, p_high)
    med = nanpercentile(a, 50)
    span = max(hi - lo, 1e-6)
    x = np.clip(a, lo, hi)
    return (x - med) / (0.5 * span)


def gradient_magnitude(field: np.ndarray, lat: Optional[np.ndarray] = None, lon: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute gradient magnitude. If lat/lon spacing available, weight by approximate meters per degree.
    """
    f = np.asarray(field, dtype=float)
    if f.ndim != 2:
        # fall back to simple grad on last two dims
        f = f.reshape(f.shape[-2], f.shape[-1])

    if lat is None or lon is None:
        gy, gx = np.gradient(f)
        g = np.hypot(gx, gy)
        return g

    # Approx meters per degree (latitude ~ 111e3 m/deg; longitude scales with cos(lat))
    # Use mean lat for scaling
    try:
        lat_mean = float(np.nanmean(lat))
    except Exception:
        lat_mean = 0.0
    meters_per_deg_lat = 111_000.0
    meters_per_deg_lon = meters_per_deg_lat * max(np.cos(np.deg2rad(lat_mean)), 1e-3)

    # Compute gradients assuming uniform degree spacing if 1D lat/lon
    if lat.ndim == 1 and lon.ndim == 1:
        dlat = np.gradient(lat) * meters_per_deg_lat
        dlon = np.gradient(lon) * meters_per_deg_lon
        gy, gx = np.gradient(f, dlat, dlon)
    else:
        gy, gx = np.gradient(f)

    return np.hypot(gx, gy)


def bloom_mask(chl: np.ndarray, perc: float = 85.0) -> np.ndarray:
    thr = nanpercentile(chl, perc)
    return (chl >= thr).astype(float)


def compute_hsi(fields: Dict[str, Optional[np.ndarray]], weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """Compute a simple Habitat Suitability Index (HSI) from available fields.

    Features:
    - chl_scaled: robust_scale(chlor)
    - kd_scaled: inverse of Kd490 (clearer water favored): robust_scale(-kd490)
    - bbp_scaled: robust_scale(bbp) as a proxy for particles/zooplankton
    - fronts: gradient magnitude of chlor (scaled)
    - bloom: high chlor mask (0/1)

    HSI = normalized weighted sum of available features.
    """
    chl = fields.get("chlor")
    kd = fields.get("kd490")
    bbp = fields.get("bbp")
    lat = fields.get("lat")
    lon = fields.get("lon")

    # Determine base grid shape from first available 2D field
    base_shape = None
    for cand in (chl, kd, bbp):
        if isinstance(cand, np.ndarray) and cand.ndim == 2:
            base_shape = cand.shape
            break

    feats = {}
    if chl is not None and chl.ndim == 2:
        feats["chl_scaled"] = robust_scale(chl)
        fronts_raw = gradient_magnitude(
            chl,
            lat=lat if isinstance(lat, np.ndarray) else None,
            lon=lon if isinstance(lon, np.ndarray) else None,
        )
        feats["fronts"] = robust_scale(fronts_raw)
        feats["bloom"] = bloom_mask(chl)
    if kd is not None and kd.ndim == 2 and (base_shape is None or kd.shape == base_shape):
        feats["kd_scaled"] = robust_scale(-kd)
    if bbp is not None and bbp.ndim == 2 and (base_shape is None or bbp.shape == base_shape):
        feats["bbp_scaled"] = robust_scale(bbp)

    # default weights
    default_w = {
        "chl_scaled": 0.35,
        "fronts": 0.25,
        "bloom": 0.15,
        "kd_scaled": 0.15,
        "bbp_scaled": 0.10,
    }
    if weights:
        default_w.update(weights)

    # normalize weights for available features
    avail = {k: v for k, v in default_w.items() if k in feats}
    s = sum(avail.values()) or 1.0
    for k in avail:
        avail[k] /= s

    # weighted sum; rescale to [0,1]
    hsi = None
    for k, w in avail.items():
        v = feats[k]
        hsi = v * w if hsi is None else hsi + v * w

    # min-max to [0,1]
    if hsi is None:
        raise ValueError("No features available to compute HSI. Check input variables.")
    hmin = np.nanpercentile(hsi, 2)
    hmax = np.nanpercentile(hsi, 98)
    hsi01 = np.clip((hsi - hmin) / (hmax - hmin + 1e-6), 0, 1)

    feats["HSI"] = hsi01
    return feats
