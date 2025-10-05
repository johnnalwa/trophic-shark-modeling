"""
Trophic Lag Mathematical Model for Shark Habitat Prediction

This module implements a scientifically-grounded trophic cascade model that accounts
for time delays between phytoplankton blooms and shark foraging activity.

Mathematical Framework:
- Phytoplankton → Zooplankton: 3-7 day lag (Gaussian kernel)
- Zooplankton → Small Fish: 7-14 day lag (Exponential decay)
- Small Fish → Sharks: 14-28 day lag (Gamma distribution)
- Total system lag: ~1-2 months from bloom to peak shark activity
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import ndimage, signal
from scipy.stats import gamma


def gaussian_kernel(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Create 1D Gaussian kernel for convolution."""
    size = int(2 * sigma * truncate + 1)
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def exponential_kernel(tau: float, length: int) -> np.ndarray:
    """Create exponential decay kernel."""
    t = np.arange(length)
    kernel = np.exp(-t / tau)
    return kernel / kernel.sum()


def gamma_kernel(shape: float, scale: float, length: int) -> np.ndarray:
    """Create gamma distribution kernel for predator response."""
    t = np.arange(length)
    kernel = gamma.pdf(t, a=shape, scale=scale)
    return kernel / kernel.sum()


class TrophicLagModel:
    """
    Mathematical model for trophic cascades in marine ecosystems.
    
    Models the time-delayed response of predators to primary productivity
    through multiple trophic levels.
    """
    
    def __init__(self, 
                 phyto_to_zoo_lag: float = 5.0,  # days
                 zoo_to_fish_lag: float = 10.0,  # days
                 fish_to_shark_lag: float = 21.0,  # days
                 efficiency_decay: float = 0.8):  # energy transfer efficiency
        """
        Initialize trophic lag model parameters.
        
        Args:
            phyto_to_zoo_lag: Days for phytoplankton → zooplankton response
            zoo_to_fish_lag: Days for zooplankton → small fish response  
            fish_to_shark_lag: Days for small fish → shark response
            efficiency_decay: Energy transfer efficiency between levels (0-1)
        """
        self.phyto_to_zoo_lag = phyto_to_zoo_lag
        self.zoo_to_fish_lag = zoo_to_fish_lag
        self.fish_to_shark_lag = fish_to_shark_lag
        self.efficiency_decay = efficiency_decay
        
        # Create convolution kernels for each trophic transition
        self.zoo_kernel = gaussian_kernel(phyto_to_zoo_lag / 2.355)  # FWHM to sigma
        self.fish_kernel = exponential_kernel(zoo_to_fish_lag, int(zoo_to_fish_lag * 3))
        self.shark_kernel = gamma_kernel(2.0, fish_to_shark_lag / 2.0, int(fish_to_shark_lag * 3))
    
    def compute_trophic_cascade(self, phytoplankton_time_series: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute trophic cascade from phytoplankton to sharks.
        
        Args:
            phytoplankton_time_series: Time series of phytoplankton biomass
            
        Returns:
            Dictionary with time series for each trophic level
        """
        # Ensure input is 1D time series
        phyto = np.asarray(phytoplankton_time_series).flatten()
        
        # Level 1: Phytoplankton → Zooplankton
        zooplankton = signal.convolve(phyto, self.zoo_kernel, mode='same')
        zooplankton *= self.efficiency_decay
        
        # Level 2: Zooplankton → Small Fish
        small_fish = signal.convolve(zooplankton, self.fish_kernel, mode='same')
        small_fish *= self.efficiency_decay
        
        # Level 3: Small Fish → Sharks
        sharks = signal.convolve(small_fish, self.shark_kernel, mode='same')
        sharks *= self.efficiency_decay
        
        return {
            'phytoplankton': phyto,
            'zooplankton': zooplankton,
            'small_fish': small_fish,
            'sharks': sharks
        }
    
    def compute_spatial_trophic_response(self, 
                                       chlorophyll_field: np.ndarray,
                                       time_axis: int = 0) -> np.ndarray:
        """
        Apply trophic lag model to spatial chlorophyll data.
        
        Args:
            chlorophyll_field: 3D array (time, lat, lon) or (time, y, x)
            time_axis: Which axis represents time
            
        Returns:
            Shark foraging potential field with same spatial dimensions
        """
        chl = np.asarray(chlorophyll_field)
        
        # Move time axis to first position
        if time_axis != 0:
            chl = np.moveaxis(chl, time_axis, 0)
        
        # Apply trophic model along time axis for each spatial location
        shark_response = np.zeros_like(chl)
        
        for i in range(chl.shape[1]):
            for j in range(chl.shape[2]):
                time_series = chl[:, i, j]
                if not np.all(np.isnan(time_series)):
                    trophic_result = self.compute_trophic_cascade(time_series)
                    shark_response[:, i, j] = trophic_result['sharks']
        
        # Move time axis back to original position
        if time_axis != 0:
            shark_response = np.moveaxis(shark_response, 0, time_axis)
            
        return shark_response
    
    def get_total_lag_days(self) -> float:
        """Get total system lag from phytoplankton to sharks."""
        return self.phyto_to_zoo_lag + self.zoo_to_fish_lag + self.fish_to_shark_lag


def create_synthetic_bloom_time_series(days: int = 90, 
                                     bloom_start: int = 20,
                                     bloom_duration: int = 15,
                                     baseline: float = 0.5,
                                     bloom_intensity: float = 3.0) -> np.ndarray:
    """
    Create synthetic phytoplankton bloom time series for testing.
    
    Args:
        days: Total number of days
        bloom_start: Day when bloom begins
        bloom_duration: Duration of bloom in days
        baseline: Baseline chlorophyll level
        bloom_intensity: Peak bloom intensity
        
    Returns:
        Time series of phytoplankton biomass
    """
    time_series = np.full(days, baseline)
    
    # Add Gaussian bloom
    bloom_center = bloom_start + bloom_duration // 2
    bloom_sigma = bloom_duration / 4.0
    
    for day in range(days):
        if bloom_start <= day <= bloom_start + bloom_duration:
            bloom_factor = np.exp(-0.5 * ((day - bloom_center) / bloom_sigma) ** 2)
            time_series[day] += bloom_intensity * bloom_factor
    
    # Add noise
    noise = np.random.normal(0, 0.1, days)
    time_series += noise
    
    return np.maximum(time_series, 0)  # Ensure non-negative


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt
    
    # Create synthetic data
    days = 90
    phyto_data = create_synthetic_bloom_time_series(days)
    
    # Initialize model
    model = TrophicLagModel()
    
    # Compute trophic cascade
    results = model.compute_trophic_cascade(phyto_data)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    time_days = np.arange(days)
    
    plt.subplot(2, 2, 1)
    plt.plot(time_days, results['phytoplankton'], 'g-', label='Phytoplankton')
    plt.title('Primary Producers')
    plt.ylabel('Biomass')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(time_days, results['zooplankton'], 'b-', label='Zooplankton')
    plt.title('Primary Consumers')
    plt.ylabel('Biomass')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(time_days, results['small_fish'], 'orange', label='Small Fish')
    plt.title('Secondary Consumers')
    plt.ylabel('Biomass')
    plt.xlabel('Days')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(time_days, results['sharks'], 'r-', label='Sharks', linewidth=2)
    plt.title('Apex Predators')
    plt.ylabel('Foraging Potential')
    plt.xlabel('Days')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('trophic_cascade_example.png', dpi=150)
    plt.show()
    
    print(f"Total system lag: {model.get_total_lag_days():.1f} days")
    print(f"Peak shark response occurs ~{np.argmax(results['sharks'])} days after bloom start")
