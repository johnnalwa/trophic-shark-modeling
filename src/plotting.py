from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _meshgrid_lonlat(lon: Optional[np.ndarray], lat: Optional[np.ndarray], ny: int, nx: int):
    if lon is None or lat is None:
        return None, None
    if lon.ndim == 1 and lat.ndim == 1:
        LON, LAT = np.meshgrid(lon, lat)
        return LON, LAT
    if lon.shape == (ny, nx) and lat.shape == (ny, nx):
        return lon, lat
    return None, None


def plot_field(field: np.ndarray, lon: Optional[np.ndarray], lat: Optional[np.ndarray], title: str, out_path: str, cmap: str = "viridis") -> None:
    plt.figure(figsize=(8, 5))
    ny, nx = field.shape[-2], field.shape[-1]
    LON, LAT = _meshgrid_lonlat(lon, lat, ny, nx)
    if LON is None or LAT is None:
        im = plt.imshow(field, origin="lower", cmap=cmap)
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
    else:
        im = plt.pcolormesh(LON, LAT, field, shading="auto", cmap=cmap)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
    plt.colorbar(im, label=title)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
