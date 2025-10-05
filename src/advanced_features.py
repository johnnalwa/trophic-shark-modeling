"""
Advanced Mathematical Framework for Shark Habitat Prediction

This module implements sophisticated algorithms for:
1. Eddy detection and retention zones
2. Thermal habitat modeling
3. Prey aggregation indices
4. Multi-scale habitat suitability
5. Uncertainty quantification
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy import ndimage, signal
from scipy.spatial.distance import cdist
import warnings

# Optional sklearn import
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available. Some advanced features will be disabled.")


class EddyDetector:
    """
    Mathematical framework for detecting mesoscale eddies and retention zones
    that concentrate prey and attract sharks.
    """
    
    def __init__(self, min_radius_km: float = 20.0, max_radius_km: float = 200.0):
        self.min_radius_km = min_radius_km
        self.max_radius_km = max_radius_km
    
    def detect_eddies_from_ssh(self, ssh: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect eddies from sea surface height anomaly using Okubo-Weiss parameter.
        
        Args:
            ssh: Sea surface height anomaly field
            lat, lon: Coordinate arrays
            
        Returns:
            Dictionary with eddy detection results
        """
        # Compute geostrophic velocity components
        g = 9.81  # gravity
        f = 2 * 7.2921e-5 * np.sin(np.deg2rad(lat))  # Coriolis parameter
        
        # Compute gradients (simplified - assumes regular grid)
        dh_dy, dh_dx = np.gradient(ssh)
        
        # Geostrophic velocities
        u = -(g / f[..., np.newaxis]) * dh_dy  # eastward velocity
        v = (g / f[..., np.newaxis]) * dh_dx   # northward velocity
        
        # Compute velocity gradients
        du_dx, du_dy = np.gradient(u)
        dv_dx, dv_dy = np.gradient(v)
        
        # Okubo-Weiss parameter
        shear = (du_dy + dv_dx) ** 2
        strain = (du_dx - dv_dy) ** 2
        vorticity = (dv_dx - du_dy) ** 2
        
        okubo_weiss = strain + shear - vorticity
        
        # Eddy cores where OW < threshold (typically -0.2 * std)
        ow_threshold = -0.2 * np.nanstd(okubo_weiss)
        eddy_cores = okubo_weiss < ow_threshold
        
        return {
            'okubo_weiss': okubo_weiss,
            'eddy_cores': eddy_cores.astype(float),
            'vorticity': vorticity,
            'u_velocity': u,
            'v_velocity': v
        }
    
    def detect_fronts_from_gradient(self, field: np.ndarray, threshold_percentile: float = 90) -> np.ndarray:
        """
        Detect oceanic fronts using gradient magnitude.
        
        Args:
            field: Ocean property field (SST, chlorophyll, etc.)
            threshold_percentile: Percentile threshold for front detection
            
        Returns:
            Binary front mask
        """
        # Compute gradient magnitude
        gy, gx = np.gradient(field)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Threshold based on percentile
        threshold = np.nanpercentile(grad_mag, threshold_percentile)
        fronts = grad_mag > threshold
        
        return fronts.astype(float)


class ThermalHabitatModel:
    """
    Mathematical model for thermal habitat preferences of sharks.
    """
    
    def __init__(self, 
                 optimal_temp_range: Tuple[float, float] = (18.0, 26.0),
                 thermal_tolerance: float = 5.0):
        """
        Initialize thermal habitat model.
        
        Args:
            optimal_temp_range: (min, max) optimal temperature range in Celsius
            thermal_tolerance: Temperature tolerance range in Celsius
        """
        self.temp_min, self.temp_max = optimal_temp_range
        self.thermal_tolerance = thermal_tolerance
    
    def compute_thermal_suitability(self, sst: np.ndarray) -> np.ndarray:
        """
        Compute thermal habitat suitability using Gaussian preference curve.
        
        Args:
            sst: Sea surface temperature field
            
        Returns:
            Thermal suitability index (0-1)
        """
        temp_optimal = (self.temp_min + self.temp_max) / 2
        temp_range = self.temp_max - self.temp_min
        
        # Gaussian suitability curve
        thermal_suit = np.exp(-0.5 * ((sst - temp_optimal) / (temp_range / 2)) ** 2)
        
        # Apply hard limits for extreme temperatures
        thermal_suit[sst < (self.temp_min - self.thermal_tolerance)] = 0
        thermal_suit[sst > (self.temp_max + self.thermal_tolerance)] = 0
        
        return thermal_suit
    
    def detect_thermal_fronts(self, sst: np.ndarray, min_gradient: float = 0.5) -> np.ndarray:
        """
        Detect thermal fronts that may concentrate prey.
        
        Args:
            sst: Sea surface temperature field
            min_gradient: Minimum temperature gradient (°C/pixel) for front detection
            
        Returns:
            Thermal front intensity
        """
        gy, gx = np.gradient(sst)
        thermal_gradient = np.sqrt(gx**2 + gy**2)
        
        # Normalize and threshold
        thermal_fronts = np.maximum(thermal_gradient - min_gradient, 0)
        thermal_fronts = thermal_fronts / (np.nanmax(thermal_fronts) + 1e-6)
        
        return thermal_fronts


class PreyAggregationModel:
    """
    Mathematical model for prey aggregation and availability.
    """
    
    def __init__(self):
        pass
    
    def compute_prey_density_index(self, 
                                 chlorophyll: np.ndarray,
                                 zooplankton_proxy: Optional[np.ndarray] = None,
                                 small_fish_proxy: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute integrated prey density index.
        
        Args:
            chlorophyll: Chlorophyll-a concentration
            zooplankton_proxy: Proxy for zooplankton biomass (e.g., backscatter)
            small_fish_proxy: Proxy for small fish biomass
            
        Returns:
            Integrated prey density index
        """
        # Ensure proper data type
        chlorophyll = np.asarray(chlorophyll, dtype=np.float64)
        
        # Handle invalid values
        chlorophyll = np.where(np.isfinite(chlorophyll) & (chlorophyll >= 0), chlorophyll, 0)
        
        # Base prey index from chlorophyll (primary productivity)
        prey_index = np.log1p(chlorophyll)  # log(1+x) to handle zeros
        
        # Add zooplankton component if available
        if zooplankton_proxy is not None:
            zooplankton_proxy = np.asarray(zooplankton_proxy, dtype=np.float64)
            zooplankton_proxy = np.where(np.isfinite(zooplankton_proxy) & (zooplankton_proxy >= 0), zooplankton_proxy, 0)
            zoo_component = np.log1p(zooplankton_proxy) * 0.7  # Weight factor
            prey_index += zoo_component
        
        # Add small fish component if available
        if small_fish_proxy is not None:
            small_fish_proxy = np.asarray(small_fish_proxy, dtype=np.float64)
            small_fish_proxy = np.where(np.isfinite(small_fish_proxy) & (small_fish_proxy >= 0), small_fish_proxy, 0)
            fish_component = np.log1p(small_fish_proxy) * 0.5
            prey_index += fish_component
        
        # Normalize to [0, 1] with safe operations
        try:
            prey_min = np.nanmin(prey_index)
            prey_max = np.nanmax(prey_index)
            if np.isfinite(prey_min) and np.isfinite(prey_max) and prey_max > prey_min:
                prey_index = (prey_index - prey_min) / (prey_max - prey_min)
            else:
                # If normalization fails, return binary mask of positive values
                prey_index = (prey_index > 0).astype(np.float64)
        except Exception:
            # Ultimate fallback: binary mask
            prey_index = (chlorophyll > 0).astype(np.float64)
        
        return prey_index
    
    def detect_prey_aggregation_zones(self, 
                                    prey_density: np.ndarray,
                                    aggregation_scale: int = 5) -> np.ndarray:
        """
        Detect zones of prey aggregation using local maxima and clustering.
        
        Args:
            prey_density: Prey density field
            aggregation_scale: Spatial scale for aggregation detection (pixels)
            
        Returns:
            Prey aggregation zones
        """
        # Apply Gaussian smoothing to identify aggregation zones
        smoothed = ndimage.gaussian_filter(prey_density, sigma=aggregation_scale)
        
        # Find local maxima
        local_maxima = ndimage.maximum_filter(smoothed, size=aggregation_scale*2) == smoothed
        
        # Threshold to keep only significant aggregations
        threshold = np.nanpercentile(smoothed, 75)
        aggregation_zones = local_maxima & (smoothed > threshold)
        
        return aggregation_zones.astype(float)


class AdvancedHSI:
    """
    Advanced Habitat Suitability Index incorporating multiple mathematical models.
    """
    
    def __init__(self):
        self.eddy_detector = EddyDetector()
        self.thermal_model = ThermalHabitatModel()
        self.prey_model = PreyAggregationModel()
    
    def compute_advanced_hsi(self, 
                           fields: Dict[str, np.ndarray],
                           trophic_response: Optional[np.ndarray] = None,
                           weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Compute advanced HSI incorporating multiple habitat factors.
        
        Args:
            fields: Dictionary of input fields (chlor, sst, ssh, etc.)
            trophic_response: Output from trophic lag model
            weights: Custom weights for different components
            
        Returns:
            Dictionary with all computed habitat components and final HSI
        """
        results = {}
        
        # Extract fields with safe access
        chlor = fields.get('chlor')
        sst = fields.get('sst')
        ssh = fields.get('ssh')
        lat = fields.get('lat')
        lon = fields.get('lon')
        
        # Validate chlorophyll data (main requirement)
        if chlor is None:
            raise ValueError("Chlorophyll data is required for advanced HSI computation")
        
        # Ensure chlorophyll is 2D spatial data
        if chlor.ndim != 2:
            raise ValueError(f"Chlorophyll data must be 2D spatial array, got shape {chlor.shape}")
        
        # Ensure all data is proper float type for mathematical operations
        chlor = np.asarray(chlor, dtype=np.float64)
        if sst is not None:
            sst = np.asarray(sst, dtype=np.float64)
        if ssh is not None:
            ssh = np.asarray(ssh, dtype=np.float64)
        if lat is not None:
            lat = np.asarray(lat, dtype=np.float64)
        if lon is not None:
            lon = np.asarray(lon, dtype=np.float64)
        if trophic_response is not None:
            trophic_response = np.asarray(trophic_response, dtype=np.float64)
        
        # Default weights (will be adjusted based on available data)
        default_weights = {
            'prey_density': 0.4,      # Increased weight since this is always available
            'frontal_zones': 0.3,     # Chlorophyll fronts always available
            'trophic_response': 0.3   # If available
        }
        if weights:
            default_weights.update(weights)
        
        # Compute prey density index (always available with chlorophyll)
        try:
            print("Debug: Computing prey density...")
            results['prey_density'] = self.prey_model.compute_prey_density_index(chlor)
            print("Debug: Computing prey aggregation...")
            results['prey_aggregation'] = self.prey_model.detect_prey_aggregation_zones(results['prey_density'])
            print("Debug: Prey computations successful")
        except Exception as e:
            print(f"Warning: Could not compute prey density: {e}")
            import traceback
            traceback.print_exc()
        
        # Compute thermal suitability (only if SST available)
        if sst is not None and sst.ndim == 2 and sst.shape == chlor.shape:
            try:
                results['thermal_suitability'] = self.thermal_model.compute_thermal_suitability(sst)
                results['thermal_fronts'] = self.thermal_model.detect_thermal_fronts(sst)
                default_weights['thermal_suitability'] = 0.2
            except Exception as e:
                print(f"Warning: Could not compute thermal features: {e}")
        
        # Detect frontal zones from chlorophyll (always available)
        try:
            print("Debug: Computing chlorophyll fronts...")
            results['chlor_fronts'] = self.eddy_detector.detect_fronts_from_gradient(chlor)
            print("Debug: Chlorophyll fronts successful")
        except Exception as e:
            print(f"Warning: Could not compute chlorophyll fronts: {e}")
            import traceback
            traceback.print_exc()
        
        # Detect eddies and retention zones (only if SSH and coordinates available)
        if (ssh is not None and lat is not None and lon is not None and 
            ssh.ndim == 2 and ssh.shape == chlor.shape):
            try:
                eddy_results = self.eddy_detector.detect_eddies_from_ssh(ssh, lat, lon)
                results.update(eddy_results)
                default_weights['eddy_retention'] = 0.15
            except Exception as e:
                print(f"Warning: Could not compute eddy features: {e}")
        
        # Combine frontal features
        frontal_components = []
        if 'chlor_fronts' in results:
            frontal_components.append(results['chlor_fronts'])
        if 'thermal_fronts' in results:
            frontal_components.append(results['thermal_fronts'])
        
        if frontal_components:
            results['frontal_zones'] = np.mean(frontal_components, axis=0)
        
        # Include trophic response if provided and compatible
        if (trophic_response is not None and 
            trophic_response.ndim == 2 and 
            trophic_response.shape == chlor.shape):
            results['trophic_response'] = trophic_response
        
        # Compute final Advanced HSI with available components
        print("Debug: Starting HSI combination...")
        hsi_components = []
        total_weight = 0
        
        for component, weight in default_weights.items():
            if component in results:
                print(f"Debug: Processing component {component}...")
                try:
                    comp_data = results[component]
                    print(f"Debug: Component {component} shape: {comp_data.shape}, dtype: {comp_data.dtype}")
                    
                    if comp_data.shape == chlor.shape:  # Ensure compatible shape
                        # Ensure component is float64 for safe operations
                        comp_data = np.asarray(comp_data, dtype=np.float64)
                        print(f"Debug: Converted {component} to float64")
                        
                        # Normalize component to [0, 1] with safe operations
                        try:
                            print(f"Debug: Computing min/max for {component}...")
                            comp_min = np.nanmin(comp_data)
                            comp_max = np.nanmax(comp_data)
                            print(f"Debug: {component} range: [{comp_min}, {comp_max}]")
                            
                            if np.isfinite(comp_min) and np.isfinite(comp_max) and comp_max > comp_min:
                                comp_norm = (comp_data - comp_min) / (comp_max - comp_min)
                                hsi_components.append(comp_norm * weight)
                                total_weight += weight
                                print(f"Debug: Successfully normalized {component}")
                            else:
                                print(f"Warning: Component {component} has invalid range [{comp_min}, {comp_max}]")
                        except Exception as norm_error:
                            print(f"Warning: Could not normalize component {component}: {norm_error}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"Debug: Component {component} shape mismatch: {comp_data.shape} vs {chlor.shape}")
                except Exception as e:
                    print(f"Warning: Could not process component {component}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Debug: HSI components count: {len(hsi_components)}, total_weight: {total_weight}")
        
        if hsi_components and total_weight > 0:
            try:
                print("Debug: Attempting to sum HSI components...")
                for i, comp in enumerate(hsi_components):
                    print(f"Debug: Component {i}: shape={comp.shape}, dtype={comp.dtype}, has_nan={np.any(np.isnan(comp))}")
                
                print("Debug: Computing sum...")
                hsi_sum = np.sum(hsi_components, axis=0)
                print(f"Debug: Sum successful, shape: {hsi_sum.shape}, dtype: {hsi_sum.dtype}")
                
                print("Debug: Computing final HSI...")
                advanced_hsi = hsi_sum / total_weight
                print(f"Debug: Final HSI successful, shape: {advanced_hsi.shape}, dtype: {advanced_hsi.dtype}")
                
                results['Advanced_HSI'] = advanced_hsi
                print("Debug: Advanced HSI computation completed successfully!")
                
            except Exception as final_error:
                print(f"Error in final HSI computation: {final_error}")
                import traceback
                traceback.print_exc()
                
                # Fallback to simple prey density
                if 'prey_density' in results:
                    print("Debug: Using prey_density as fallback HSI")
                    results['Advanced_HSI'] = results['prey_density']
                else:
                    raise ValueError("No valid components available for HSI computation")
        else:
            # Fallback to simple prey density if no other components work
            if 'prey_density' in results:
                print("Debug: No valid components, using prey_density as HSI")
                results['Advanced_HSI'] = results['prey_density']
            else:
                raise ValueError("No valid components available for HSI computation")
        
        return results
    
    def compute_uncertainty(self, hsi: np.ndarray, input_fields: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute uncertainty in HSI predictions based on data quality and coverage.
        
        Args:
            hsi: Computed HSI field
            input_fields: Dictionary of input data fields
            
        Returns:
            Uncertainty field (0-1, where 1 is highest uncertainty)
        """
        uncertainty = np.zeros_like(hsi)
        
        # Data coverage uncertainty
        for field_name, field_data in input_fields.items():
            if field_data is not None:
                # Add uncertainty where data is missing or poor quality
                missing_mask = np.isnan(field_data)
                uncertainty += missing_mask.astype(float) * 0.2
        
        # Edge effects uncertainty
        edge_uncertainty = np.ones_like(hsi) * 0.1
        edge_uncertainty[5:-5, 5:-5] = 0  # Reduce uncertainty in interior
        uncertainty += edge_uncertainty
        
        # Normalize to [0, 1]
        uncertainty = np.clip(uncertainty, 0, 1)
        
        return uncertainty


if __name__ == "__main__":
    # Example usage
    print("Advanced mathematical framework for shark habitat prediction initialized.")
    print("Key components:")
    print("- Eddy detection using Okubo-Weiss parameter")
    print("- Thermal habitat modeling with Gaussian preferences")  
    print("- Prey aggregation detection using local maxima")
    print("- Integrated Advanced HSI with uncertainty quantification")
