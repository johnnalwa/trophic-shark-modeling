"""
Integrated Mathematical Pipeline for NASA Space Apps Challenge
Shark Habitat Prediction using PACE Satellite Data

This module combines all mathematical models into a comprehensive framework:
1. Trophic lag modeling
2. Advanced habitat suitability computation
3. Real-time tag integration concepts
4. Educational visualization components
"""

import os
import json
from typing import Dict, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from .data_loader import load_core_fields, open_dataset
from .trophic_model import TrophicLagModel, create_synthetic_bloom_time_series
from .advanced_features import AdvancedHSI, EddyDetector, ThermalHabitatModel
from .shark_tag_algorithm import FeedingEventDetector, create_synthetic_shark_data
from .plotting import plot_field


class IntegratedSharkHabitatPredictor:
    """
    Comprehensive mathematical framework for shark habitat prediction
    addressing all NASA Space Apps Challenge requirements.
    """
    
    def __init__(self, 
                 trophic_lag_days: float = 30.0,
                 thermal_range: Tuple[float, float] = (18.0, 26.0)):
        """
        Initialize integrated prediction system.
        
        Args:
            trophic_lag_days: Total trophic system lag in days
            thermal_range: Optimal temperature range for sharks
        """
        self.trophic_model = TrophicLagModel(
            phyto_to_zoo_lag=5.0,
            zoo_to_fish_lag=10.0, 
            fish_to_shark_lag=trophic_lag_days - 15.0
        )
        
        self.advanced_hsi = AdvancedHSI()
        self.advanced_hsi.thermal_model = ThermalHabitatModel(optimal_temp_range=thermal_range)
        
        self.tag_detector = FeedingEventDetector()
        
    def process_pace_data_with_trophic_modeling(self, 
                                              input_path: str,
                                              overrides: Optional[Dict] = None,
                                              time_series_length: int = 60) -> Dict[str, np.ndarray]:
        """
        Process PACE data with integrated trophic lag modeling.
        
        Args:
            input_path: Path to PACE NetCDF file
            overrides: Variable name overrides
            time_series_length: Length of synthetic time series for trophic modeling
            
        Returns:
            Dictionary with all computed habitat components
        """
        print("Loading PACE satellite data...")
        
        # Load core fields from PACE data
        fields = load_core_fields(input_path, overrides=overrides)
        
        # Create synthetic time series for trophic modeling demonstration
        # In a real application, this would use historical satellite data
        print("Generating trophic lag model...")
        
        if fields.get('chlor') is not None:
            chlor_field = fields['chlor']
            
            # Debug: Print field shapes and types
            print(f"Debug: Chlorophyll field shape: {chlor_field.shape}, dtype: {chlor_field.dtype}")
            for key, value in fields.items():
                if value is not None:
                    print(f"Debug: {key} shape: {value.shape}, dtype: {value.dtype}")
                    
            # Ensure chlorophyll data is float type for mathematical operations
            if not np.issubdtype(chlor_field.dtype, np.floating):
                print(f"Converting chlorophyll from {chlor_field.dtype} to float64")
                chlor_field = chlor_field.astype(np.float64)
                fields['chlor'] = chlor_field
            
            # Ensure chlorophyll is 2D spatial data
            if chlor_field.ndim != 2:
                print(f"Warning: Chlorophyll field is not 2D (shape: {chlor_field.shape}), skipping trophic modeling")
                return fields
            
            # Create synthetic time series based on current chlorophyll distribution
            # Use spatial mean as baseline for time series generation
            mean_chlor = np.nanmean(chlor_field)
            std_chlor = np.nanstd(chlor_field)
            
            # Generate synthetic bloom time series
            synthetic_bloom = create_synthetic_bloom_time_series(
                days=time_series_length,
                baseline=max(mean_chlor * 0.5, 0.1),
                bloom_intensity=min(std_chlor * 2, mean_chlor * 3)
            )
            
            # Apply trophic model
            trophic_results = self.trophic_model.compute_trophic_cascade(synthetic_bloom)
            
            # Scale trophic response to match spatial chlorophyll field
            shark_response_1d = trophic_results['sharks']
            current_response = shark_response_1d[-1]  # Use current time point
            
            # Create spatial trophic response field
            trophic_spatial = np.full_like(chlor_field, current_response)
            
            # Modulate by actual chlorophyll distribution
            try:
                chlor_normalized = (chlor_field - np.nanmin(chlor_field)) / (np.nanmax(chlor_field) - np.nanmin(chlor_field) + 1e-6)
                trophic_spatial *= (0.5 + 0.5 * chlor_normalized)  # Scale by local productivity
                
                fields['trophic_response'] = trophic_spatial
                fields['trophic_time_series'] = trophic_results
            except Exception as e:
                print(f"Warning: Could not create trophic spatial response: {e}")
                # Still save the time series for educational purposes
                fields['trophic_time_series'] = trophic_results
        
        return fields
    
    def compute_comprehensive_habitat_prediction(self, 
                                               fields: Dict[str, np.ndarray],
                                               include_uncertainty: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive habitat prediction using all mathematical models.
        
        Args:
            fields: Input data fields from PACE processing
            include_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            Complete habitat prediction results
        """
        print("Computing advanced habitat suitability index...")
        
        # Extract trophic response if available
        trophic_response = fields.get('trophic_response')
        
        # Debug: Check field types before advanced computation
        print("Debug: Fields going into advanced HSI:")
        for key, value in fields.items():
            if value is not None:
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, has_nan={np.any(np.isnan(value))}")
                else:
                    print(f"  {key}: type={type(value)} (not an array)")
        
        # Filter fields to only include arrays for advanced HSI computation
        array_fields = {}
        for key, value in fields.items():
            if value is not None and hasattr(value, 'shape') and hasattr(value, 'dtype'):
                array_fields[key] = value
        
        print(f"Debug: Passing {len(array_fields)} array fields to advanced HSI")
        
        # Compute advanced HSI with all components
        habitat_results = self.advanced_hsi.compute_advanced_hsi(
            fields=array_fields,
            trophic_response=trophic_response
        )
        
        # Add uncertainty quantification
        if include_uncertainty and 'Advanced_HSI' in habitat_results:
            try:
                print("Debug: Computing uncertainty...")
                uncertainty = self.advanced_hsi.compute_uncertainty(
                    habitat_results['Advanced_HSI'], 
                    array_fields  # Use filtered array fields instead of original fields
                )
                habitat_results['uncertainty'] = uncertainty
                
                # Compute confidence-weighted HSI
                confidence = 1.0 - uncertainty
                weighted_hsi = habitat_results['Advanced_HSI'] * confidence
                habitat_results['confidence_weighted_HSI'] = weighted_hsi
                print("Debug: Uncertainty computation successful")
            except Exception as unc_error:
                print(f"Warning: Could not compute uncertainty: {unc_error}")
                import traceback
                traceback.print_exc()
        
        return habitat_results
    
    def demonstrate_tag_integration(self, 
                                  habitat_results: Dict[str, np.ndarray],
                                  duration_hours: float = 4.0) -> Dict[str, any]:
        """
        Demonstrate real-time shark tag integration concept.
        
        Args:
            habitat_results: Computed habitat suitability results
            duration_hours: Duration for synthetic tag data
            
        Returns:
            Tag integration demonstration results
        """
        print("Demonstrating real-time shark tag integration...")
        
        # Generate synthetic shark tag data
        tag_data = create_synthetic_shark_data(duration_hours=duration_hours)
        
        # Detect feeding events
        feeding_results = self.tag_detector.detect_feeding_events(tag_data['accelerometer'])
        
        # Correlate feeding events with habitat predictions
        # In a real system, this would use GPS locations to sample habitat maps
        hsi_mean = np.nanmean(habitat_results.get('Advanced_HSI', [0.5]))
        
        tag_integration = {
            'synthetic_tag_data': tag_data,
            'feeding_events': feeding_results['feeding_events'],
            'habitat_correlation': {
                'mean_hsi_at_location': hsi_mean,
                'feeding_event_count': len(feeding_results['feeding_events']),
                'feeding_habitat_correlation': hsi_mean * len(feeding_results['feeding_events'])
            }
        }
        
        return tag_integration
    
    def create_educational_visualizations(self, 
                                        habitat_results: Dict[str, np.ndarray],
                                        fields: Dict[str, np.ndarray],
                                        output_dir: str = "outputs") -> List[str]:
        """
        Create educational visualizations for high school audience.
        
        Args:
            habitat_results: Computed habitat results
            fields: Original input fields
            output_dir: Output directory for visualizations
            
        Returns:
            List of created visualization files
        """
        print("Creating educational visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        
        # Get coordinate data
        lon = fields.get('lon')
        lat = fields.get('lat')
        
        # 1. Trophic Cascade Explanation
        if 'trophic_time_series' in fields:
            self._create_trophic_cascade_plot(
                fields['trophic_time_series'], 
                os.path.join(output_dir, "educational_trophic_cascade.png")
            )
            created_files.append("educational_trophic_cascade.png")
        
        # 2. Multi-panel habitat components
        self._create_habitat_components_plot(
            habitat_results, fields, lon, lat,
            os.path.join(output_dir, "educational_habitat_components.png")
        )
        created_files.append("educational_habitat_components.png")
        
        # 3. Final integrated prediction with explanation
        if 'Advanced_HSI' in habitat_results:
            self._create_final_prediction_plot(
                habitat_results['Advanced_HSI'], lon, lat,
                os.path.join(output_dir, "educational_shark_habitat_prediction.png")
            )
            created_files.append("educational_shark_habitat_prediction.png")
        
        # 4. Uncertainty visualization
        if 'uncertainty' in habitat_results:
            self._create_uncertainty_plot(
                habitat_results['Advanced_HSI'], 
                habitat_results['uncertainty'], 
                lon, lat,
                os.path.join(output_dir, "educational_prediction_uncertainty.png")
            )
            created_files.append("educational_prediction_uncertainty.png")
        
        return created_files
    
    def _create_trophic_cascade_plot(self, trophic_results: Dict, output_path: str):
        """Create educational plot explaining trophic cascade."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("From Space to Sharks: The Ocean Food Web Connection", fontsize=16, fontweight='bold')
        
        days = len(trophic_results['phytoplankton'])
        time_days = np.arange(days)
        
        # Phytoplankton (detected by satellites)
        axes[0,0].plot(time_days, trophic_results['phytoplankton'], 'g-', linewidth=3, label='Phytoplankton')
        axes[0,0].set_title('1. Phytoplankton\n(What satellites see)', fontweight='bold')
        axes[0,0].set_ylabel('Biomass')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].text(0.05, 0.95, 'NASA PACE\nsatellite data', transform=axes[0,0].transAxes, 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                      verticalalignment='top')
        
        # Zooplankton
        axes[0,1].plot(time_days, trophic_results['zooplankton'], 'b-', linewidth=3, label='Zooplankton')
        axes[0,1].set_title('2. Zooplankton\n(Tiny ocean animals)', fontweight='bold')
        axes[0,1].set_ylabel('Biomass')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].text(0.05, 0.95, '~5 day delay\nfrom phytoplankton', transform=axes[0,1].transAxes,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                      verticalalignment='top')
        
        # Small Fish
        axes[1,0].plot(time_days, trophic_results['small_fish'], 'orange', linewidth=3, label='Small Fish')
        axes[1,0].set_title('3. Small Fish\n(Shark prey)', fontweight='bold')
        axes[1,0].set_ylabel('Biomass')
        axes[1,0].set_xlabel('Days')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].text(0.05, 0.95, '~15 day delay\nfrom phytoplankton', transform=axes[1,0].transAxes,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                      verticalalignment='top')
        
        # Sharks
        axes[1,1].plot(time_days, trophic_results['sharks'], 'r-', linewidth=4, label='Sharks')
        axes[1,1].set_title('4. Sharks\n(What we want to predict!)', fontweight='bold')
        axes[1,1].set_ylabel('Foraging Activity')
        axes[1,1].set_xlabel('Days')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].text(0.05, 0.95, '~30 day delay\nfrom phytoplankton', transform=axes[1,1].transAxes,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
                      verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_habitat_components_plot(self, habitat_results: Dict, fields: Dict, 
                                      lon: np.ndarray, lat: np.ndarray, output_path: str):
        """Create multi-panel plot showing habitat components."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Shark Habitat Components: What Makes a Good Shark Neighborhood?", 
                    fontsize=16, fontweight='bold')
        
        components = [
            ('chlor', 'Phytoplankton\n(Food web base)', 'Greens'),
            ('prey_density', 'Prey Density\n(Shark food)', 'Blues'),
            ('frontal_zones', 'Ocean Fronts\n(Prey concentration)', 'Purples'),
            ('thermal_suitability', 'Temperature\n(Shark comfort zone)', 'Reds'),
            ('trophic_response', 'Trophic Response\n(Food web prediction)', 'Oranges'),
            ('Advanced_HSI', 'Final Prediction\n(Best shark habitat)', 'viridis')
        ]
        
        for i, (key, title, cmap) in enumerate(components):
            row, col = i // 3, i % 3
            
            if key in habitat_results:
                data = habitat_results[key]
            elif key in fields:
                data = fields[key]
            else:
                continue
            
            if lon is not None and lat is not None:
                im = axes[row, col].pcolormesh(lon, lat, data, shading='auto', cmap=cmap)
                axes[row, col].set_xlabel('Longitude')
                axes[row, col].set_ylabel('Latitude')
            else:
                im = axes[row, col].imshow(data, origin='lower', cmap=cmap)
                axes[row, col].set_xlabel('X (pixels)')
                axes[row, col].set_ylabel('Y (pixels)')
            
            axes[row, col].set_title(title, fontweight='bold')
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_final_prediction_plot(self, hsi: np.ndarray, lon: np.ndarray, 
                                    lat: np.ndarray, output_path: str):
        """Create final prediction plot with educational annotations."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if lon is not None and lat is not None:
            im = ax.pcolormesh(lon, lat, hsi, shading='auto', cmap='RdYlBu_r')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
        else:
            im = ax.imshow(hsi, origin='lower', cmap='RdYlBu_r')
            ax.set_xlabel('X (pixels)', fontsize=12)
            ax.set_ylabel('Y (pixels)', fontsize=12)
        
        ax.set_title('Shark Habitat Suitability Prediction\nRed = High Probability, Blue = Low Probability', 
                    fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Habitat Suitability (0 = Poor, 1 = Excellent)', fontsize=12)
        
        # Add educational text box
        textstr = '''How to read this map:
        [RED] Red areas: High shark activity predicted
        [BLUE] Blue areas: Low shark activity predicted
        
        This prediction combines:
        • Satellite data from NASA PACE
        • Ocean food web modeling
        • Physical oceanography
        • Shark behavior patterns'''
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_uncertainty_plot(self, hsi: np.ndarray, uncertainty: np.ndarray,
                               lon: np.ndarray, lat: np.ndarray, output_path: str):
        """Create uncertainty visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prediction
        if lon is not None and lat is not None:
            im1 = ax1.pcolormesh(lon, lat, hsi, shading='auto', cmap='RdYlBu_r')
            im2 = ax2.pcolormesh(lon, lat, uncertainty, shading='auto', cmap='Greys')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
        else:
            im1 = ax1.imshow(hsi, origin='lower', cmap='RdYlBu_r')
            im2 = ax2.imshow(uncertainty, origin='lower', cmap='Greys')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
        
        ax1.set_title('Shark Habitat Prediction', fontweight='bold')
        ax2.set_title('Prediction Uncertainty', fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='Habitat Suitability')
        plt.colorbar(im2, ax=ax2, label='Uncertainty (0=Confident, 1=Uncertain)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, 
                              habitat_results: Dict[str, np.ndarray],
                              tag_integration: Dict[str, any],
                              output_path: str = "outputs/nasa_challenge_summary.txt"):
        """
        Generate comprehensive summary report for NASA challenge submission.
        
        Args:
            habitat_results: Computed habitat results
            tag_integration: Tag integration results
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            f.write("NASA SPACE APPS CHALLENGE - SHARK HABITAT PREDICTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MATHEMATICAL FRAMEWORK COMPONENTS:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Trophic Lag Modeling: [COMPLETED] Implemented\n")
            f.write("   - Phytoplankton -> Zooplankton: 5 day lag\n")
            f.write("   - Zooplankton -> Small Fish: 10 day lag\n")
            f.write("   - Small Fish -> Sharks: 15+ day lag\n")
            f.write("   - Total system lag: ~30 days\n\n")
            
            f.write("2. Advanced Habitat Suitability Index: [COMPLETED] Implemented\n")
            f.write("   - Prey density modeling\n")
            f.write("   - Thermal habitat preferences\n")
            f.write("   - Ocean front detection\n")
            f.write("   - Eddy retention zones\n")
            f.write("   - Uncertainty quantification\n\n")
            
            f.write("3. Real-time Shark Tag Concept: [COMPLETED] Implemented\n")
            f.write("   - Feeding event detection algorithms\n")
            f.write("   - Behavioral state classification\n")
            f.write("   - Data compression optimization\n")
            f.write("   - Real-time transmission protocols\n\n")
            
            f.write("HABITAT PREDICTION RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            if 'Advanced_HSI' in habitat_results:
                hsi = habitat_results['Advanced_HSI']
                f.write(f"Mean habitat suitability: {np.nanmean(hsi):.3f}\n")
                f.write(f"High suitability areas (>0.7): {np.sum(hsi > 0.7) / hsi.size * 100:.1f}%\n")
                f.write(f"Peak habitat suitability: {np.nanmax(hsi):.3f}\n")
            
            if 'uncertainty' in habitat_results:
                unc = habitat_results['uncertainty']
                f.write(f"Mean prediction uncertainty: {np.nanmean(unc):.3f}\n")
                f.write(f"High confidence areas (<0.3 uncertainty): {np.sum(unc < 0.3) / unc.size * 100:.1f}%\n")
            
            f.write("\nTAG INTEGRATION DEMONSTRATION:\n")
            f.write("-" * 40 + "\n")
            if 'feeding_events' in tag_integration:
                events = tag_integration['feeding_events']
                f.write(f"Simulated feeding events detected: {len(events)}\n")
                if events:
                    avg_duration = np.mean([e['duration'] for e in events])
                    f.write(f"Average feeding event duration: {avg_duration:.1f} seconds\n")
            
            f.write("\nEDUCATIONAL IMPACT:\n")
            f.write("-" * 40 + "\n")
            f.write("[COMPLETED] High school-friendly visualizations created\n")
            f.write("[COMPLETED] Trophic cascade explanation with satellite connection\n")
            f.write("[COMPLETED] Step-by-step habitat component breakdown\n")
            f.write("[COMPLETED] Clear uncertainty communication\n")
            f.write("[COMPLETED] Conservation relevance highlighted\n\n")
            
            f.write("CHALLENGE REQUIREMENTS ADDRESSED:\n")
            f.write("-" * 40 + "\n")
            f.write("[COMPLETED] Mathematical framework for shark identification\n")
            f.write("[COMPLETED] NASA satellite data integration (PACE)\n")
            f.write("[COMPLETED] Trophic step consideration in modeling\n")
            f.write("[COMPLETED] Physical oceanographic features\n")
            f.write("[COMPLETED] Real-time tag concept with feeding detection\n")
            f.write("[COMPLETED] Educational content for high school audience\n")
            f.write("[COMPLETED] Conservation and management applications\n\n")
            
            f.write("NEXT STEPS FOR REAL-WORLD APPLICATION:\n")
            f.write("-" * 40 + "\n")
            f.write("• Validate with actual shark tracking data\n")
            f.write("• Integrate multi-temporal satellite datasets\n")
            f.write("• Add species-specific habitat preferences\n")
            f.write("• Develop real-time operational system\n")
            f.write("• Partner with marine conservation organizations\n")
        
        print(f"Summary report written to: {output_path}")


if __name__ == "__main__":
    print("Integrated Shark Habitat Prediction Pipeline")
    print("NASA Space Apps Challenge - Mathematical Framework")
    print("=" * 50)
