import os
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from netCDF4 import Dataset as NC


def open_dataset(path: str) -> xr.Dataset:
    """Open a NetCDF file with xarray.

    Uses netCDF4 engine and avoids decoding times (PACE L2 often uses integer times).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # Many PACE/SeaDAS L2 products use HDF5 groups (e.g., Geophysical_Data, Navigation_Data).
    # Enumerate all groups via netCDF4 and merge them into a single xarray Dataset.

    def _enumerate_groups(nc_group, prefix=""):
        paths = []
        # netCDF4 group path: root has '/', child paths like '/Geophysical_Data'
        # xarray expects group paths without leading '/', nested as 'A/B'.
        for name, grp in nc_group.groups.items():
            path = f"{prefix}/{name}" if prefix else name
            paths.append(path)
            paths.extend(_enumerate_groups(grp, path))
        return paths

    group_paths = []
    try:
        with NC(path, mode="r") as nc:
            group_paths = [""]  # include root
            group_paths += _enumerate_groups(nc, prefix="")
    except Exception:
        # If we cannot enumerate, fall back to simple open
        return xr.open_dataset(path, engine="netcdf4", decode_times=False)

    opened = []
    for gp in group_paths:
        group_arg = None if gp == "" else gp
        try:
            ds_g = xr.open_dataset(path, engine="netcdf4", group=group_arg, decode_times=False)
            if len(ds_g.data_vars) == 0 and len(ds_g.coords) == 0:
                continue
            opened.append(ds_g)
        except Exception:
            continue

    def _merge(ds_list):
        if not ds_list:
            return None
        if len(ds_list) == 1:
            return ds_list[0]
        return xr.merge(ds_list, compat="no_conflicts", join="outer")

    merged = _merge(opened)
    if merged is not None and (len(merged.data_vars) > 0 or len(merged.coords) > 0):
        return merged

    # Retry with h5netcdf engine
    opened_h5 = []
    for gp in group_paths:
        group_arg = None if gp == "" else gp
        try:
            ds_g = xr.open_dataset(
                path, engine="h5netcdf", group=group_arg, decode_times=False, phony_dims="access"
            )
            if len(ds_g.data_vars) == 0 and len(ds_g.coords) == 0:
                continue
            opened_h5.append(ds_g)
        except Exception:
            continue
    merged_h5 = _merge(opened_h5)
    if merged_h5 is not None and (len(merged_h5.data_vars) > 0 or len(merged_h5.coords) > 0):
        return merged_h5

    # Final fallback to simple open attempts
    try:
        return xr.open_dataset(path, engine="h5netcdf", decode_times=False, phony_dims="access")
    except Exception:
        return xr.open_dataset(path, engine="netcdf4", decode_times=False)


def _find_by_candidates(ds: xr.Dataset, candidates: Tuple[str, ...]) -> Optional[str]:
    keys = list(ds.data_vars.keys())
    lower_map = {k.lower(): k for k in keys}
    for cand in candidates:
        if cand in ds.data_vars:
            return cand
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def _find_by_contains(ds: xr.Dataset, substrings: Tuple[str, ...]) -> Optional[str]:
    for k in ds.data_vars.keys():
        kl = k.lower()
        if all(s in kl for s in substrings):
            return k
    return None


def _find_lat_lon(ds: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
    lat_cands = ("lat", "latitude", "nav_lat", "y")
    lon_cands = ("lon", "longitude", "nav_lon", "x")

    lat = _find_by_candidates(ds, lat_cands)
    lon = _find_by_candidates(ds, lon_cands)

    # If not present as data_vars, check coordinates
    if lat is None:
        for c in lat_cands:
            if c in ds.coords:
                lat = c
                break
    if lon is None:
        for c in lon_cands:
            if c in ds.coords:
                lon = c
                break

    # Fallback by CF units attribute
    if (lat is None) or (lon is None):
        for k in list(ds.variables):
            v = ds.variables[k]
            units = getattr(v, "units", "").lower()
            sn = getattr(v, "standard_name", "").lower()
            if lat is None and (units in ("degrees_north",) or "latitude" in sn):
                lat = k
            if lon is None and (units in ("degrees_east",) or "longitude" in sn):
                lon = k
            if lat is not None and lon is not None:
                break

    return lat, lon


def detect_core_variables(ds: xr.Dataset) -> Dict[str, Optional[str]]:
    """Heuristically detect core ocean color variables in a PACE L2 OC/BGC file.

    Returns mapping for keys: 'chlor', 'kd490', 'bbp', 'lat', 'lon'.
    'bbp' will try to select a commonly used wavelength if available (e.g., 443 or 532 nm).
    """
    lat, lon = _find_lat_lon(ds)

    # Chlorophyll-a candidates
    chlor = _find_by_candidates(
        ds,
        (
            "chlor_a",
            "chlor_a_total",
            "chlor",
            "chlora",
            "chl_ocx",
            "chl",
            "CHL",  # some products use uppercase
        ),
    )
    if chlor is None:
        # CF attribute-based search
        for k in ds.data_vars:
            v = ds[k]
            sn = getattr(v, "standard_name", "").lower()
            ln = getattr(v, "long_name", "").lower()
            if "chlorophyll" in sn or "chlorophyll" in ln:
                chlor = k
                break
    if chlor is None:
        chlor = _find_by_contains(ds, ("chlor",))

    # Diffuse attenuation at 490 nm
    kd490 = _find_by_candidates(ds, ("Kd_490", "kd_490", "kd490", "KD490"))
    if kd490 is None:
        # CF/name attribute based search
        for k in ds.data_vars:
            v = ds[k]
            sn = getattr(v, "standard_name", "").lower()
            ln = getattr(v, "long_name", "").lower()
            if ("diffuse" in ln or "attenuation" in ln or "kd" in k.lower()) and ("490" in k.lower() or "490" in ln):
                kd490 = k
                break
    if kd490 is None:
        kd490 = _find_by_contains(ds, ("kd", "490"))

    # Particulate backscatter (prefer ~443 nm, else any bbp_###)
    bbp = None
    preferred = ["bbp_443", "bbp_440", "bbp_532", "bbp_555"]
    for p in preferred:
        if p in ds.data_vars:
            bbp = p
            break
    if bbp is None:
        # fall back to any variable that looks like bbp_* or contains 'bbp'
        for k in ds.data_vars.keys():
            if k.lower().startswith("bbp_") or "bbp" in k.lower():
                bbp = k
                break
    if bbp is None:
        # try CF attribute search for backscattering
        for k in ds.data_vars:
            v = ds[k]
            sn = getattr(v, "standard_name", "").lower()
            ln = getattr(v, "long_name", "").lower()
            if "backscatt" in sn or "backscatt" in ln:
                bbp = k
                break

    return {
        "lat": lat,
        "lon": lon,
        "chlor": chlor,
        "kd490": kd490,
        "bbp": bbp,
    }


def summarize_dataset(ds: xr.Dataset, max_vars: int = 200) -> str:
    lines = []
    lines.append("Coords:")
    for c in ds.coords:
        v = ds[c]
        lines.append(f" - {c}: dims={v.dims}, shape={getattr(v,'shape',None)}, units={getattr(v,'units','')}")
    lines.append("")
    lines.append("Data vars (limited):")
    for i, k in enumerate(list(ds.data_vars)[:max_vars]):
        v = ds[k]
        ln = getattr(v, 'long_name', '')
        sn = getattr(v, 'standard_name', '')
        un = getattr(v, 'units', '')
        lines.append(f"[{i:03d}] {k}: dims={v.dims}, shape={getattr(v,'shape',None)}, units='{un}', long_name='{ln}', standard_name='{sn}'")
    lines.append("")
    lines.append(f"Total data vars: {len(list(ds.data_vars))}")
    return "\n".join(lines)


def load_core_fields(path: str, overrides: Optional[Dict[str, str]] = None) -> Dict[str, np.ndarray]:
    """Load arrays for lat, lon, chlor, kd490, bbp. Missing variables return as None.

    Returns dict of numpy arrays with keys: 'lat', 'lon', 'chlor', 'kd490', 'bbp'.
    """
    ds = open_dataset(path)
    names = detect_core_variables(ds)

    # Apply user overrides if provided
    if overrides:
        for k, v in overrides.items():
            if v and v in ds.variables:
                names[k] = v

    out: Dict[str, Optional[np.ndarray]] = {k: None for k in ("lat", "lon", "chlor", "kd490", "bbp")}

    lat_name, lon_name = names["lat"], names["lon"]
    if lat_name is not None:
        out["lat"] = np.asarray(ds[lat_name].values)
    if lon_name is not None:
        out["lon"] = np.asarray(ds[lon_name].values)

    for key in ("chlor", "kd490", "bbp"):
        v = names[key]
        if v is not None:
            arr = np.asarray(ds[v].values)
            # squeeze singleton dims if present
            arr = np.squeeze(arr)
            # Ensure float type for mathematical operations
            if not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float64)
            out[key] = arr
        else:
            out[key] = None

    # Basic sanity: try to align shapes by broadcasting where possible
    # This keeps first 2 dims (y, x) if present
    def _2d(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if a is None:
            return None
        if a.ndim == 2:
            return a
        if a.ndim > 2:
            # take first time or band if needed
            return a.reshape(a.shape[-2], a.shape[-1])
        if a.ndim == 1:
            # 1D arrays are likely spectral data or coordinates
            # Only keep if they're coordinate arrays (lat/lon)
            return None  # Filter out 1D spectral data
        return a

    out["chlor"] = _2d(out["chlor"])
    out["kd490"] = _2d(out["kd490"])
    out["bbp"] = _2d(out["bbp"])
    
    # Additional validation: ensure spatial fields have compatible shapes
    chlor_shape = out["chlor"].shape if out["chlor"] is not None else None
    
    # Filter out fields that don't match chlorophyll spatial dimensions
    for key in ("kd490", "bbp"):
        if out[key] is not None and chlor_shape is not None:
            if out[key].shape != chlor_shape:
                print(f"Warning: {key} shape {out[key].shape} doesn't match chlorophyll {chlor_shape}, setting to None")
                out[key] = None

    return out


def dump_group_tree(nc_path: str, out_path: str, max_show: int = 10000) -> None:
    """Write a readable tree of groups, dimensions, and variables to a text file.

    Useful for products that store variables inside nested groups.
    """
    def _recurse(grp, prefix: str, lines: list, count: int) -> int:
        lines.append(f"{prefix}Group: {grp.path if hasattr(grp,'path') else '/'}")
        # Dims
        if getattr(grp, 'dimensions', None):
            for name, dim in grp.dimensions.items():
                lines.append(f"{prefix}  dim {name}: size={len(dim)}")
        # Vars
        if getattr(grp, 'variables', None):
            for name, var in grp.variables.items():
                try:
                    shape = var.shape
                except Exception:
                    shape = None
                units = getattr(var, 'units', '')
                long_name = getattr(var, 'long_name', '')
                standard_name = getattr(var, 'standard_name', '')
                lines.append(
                    f"{prefix}  var {name}: shape={shape}, units='{units}', long_name='{long_name}', standard_name='{standard_name}'"
                )
                count += 1
                if count >= max_show:
                    return count
        # Children
        for child_name, child in grp.groups.items():
            count = _recurse(child, prefix + "  ", lines, count)
            if count >= max_show:
                return count
        return count

    lines: list[str] = []
    try:
        with NC(nc_path, mode="r") as nc:
            _recurse(nc, prefix="", lines=lines, count=0)
    except Exception as e:
        lines.append(f"Failed to read groups: {e}")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass
