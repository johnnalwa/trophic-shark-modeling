import os
import argparse
import json

from src.data_loader import load_core_fields, open_dataset, detect_core_variables, summarize_dataset, dump_group_tree
from src.features import compute_hsi
from src.plotting import plot_field
from src.integrated_pipeline import IntegratedSharkHabitatPredictor


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_overrides(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # only keep known keys
        allowed = {k: v for k, v in data.items() if k in {"lat", "lon", "chlor", "kd490", "bbp"}}
        return allowed
    except Exception:
        return {}


def run(input_path: str, out_dir: str = "outputs", overrides_path: str = "overrides.json", 
        use_advanced_model: bool = True) -> None:
    ensure_dir(out_dir)
    overrides = load_overrides(overrides_path)

    # Inspect dataset and detection
    ds = open_dataset(input_path)
    detected = detect_core_variables(ds)
    summary_txt = summarize_dataset(ds, max_vars=200)
    with open(os.path.join(out_dir, "dataset_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_txt)
        f.write("\n\nDetected variables (before overrides):\n")
        for k in ("lat", "lon", "chlor", "kd490", "bbp"):
            f.write(f" - {k}: {detected.get(k)}\n")
        if overrides:
            f.write("\nOverrides provided:\n")
            for k, v in overrides.items():
                f.write(f" - {k}: {v}\n")

    print("Wrote dataset summary to outputs/dataset_summary.txt")
    dump_group_tree(input_path, os.path.join(out_dir, "group_tree.txt"))

    if use_advanced_model:
        print("\n🦈 NASA SPACE APPS CHALLENGE - ADVANCED SHARK HABITAT PREDICTION 🦈")
        print("=" * 70)
        
        # Initialize integrated prediction system
        predictor = IntegratedSharkHabitatPredictor()
        
        try:
            # Process PACE data with trophic modeling
            fields = predictor.process_pace_data_with_trophic_modeling(
                input_path, overrides=overrides
            )
            
            # Compute comprehensive habitat prediction
            habitat_results = predictor.compute_comprehensive_habitat_prediction(fields)
            
            # Demonstrate tag integration
            tag_integration = predictor.demonstrate_tag_integration(habitat_results)
            
            # Create educational visualizations
            educational_files = predictor.create_educational_visualizations(
                habitat_results, fields, out_dir
            )
            
            # Generate summary report
            predictor.generate_summary_report(
                habitat_results, tag_integration, 
                os.path.join(out_dir, "nasa_challenge_summary.txt")
            )
            
            print(f"\n✅ ADVANCED ANALYSIS COMPLETE!")
            print(f"📊 Educational visualizations: {len(educational_files)} files created")
            print(f"📋 Summary report: nasa_challenge_summary.txt")
            
            # Save advanced results
            lon = fields.get("lon")
            lat = fields.get("lat")
            
            advanced_plots = [
                (habitat_results.get("Advanced_HSI"), "Advanced Habitat Suitability Index", "Advanced_HSI.png"),
                (habitat_results.get("prey_density"), "Prey Density Index", "prey_density.png"),
                (habitat_results.get("thermal_suitability"), "Thermal Suitability", "thermal_suitability.png"),
                (habitat_results.get("frontal_zones"), "Frontal Zones", "frontal_zones.png"),
                (habitat_results.get("trophic_response"), "Trophic Response", "trophic_response.png"),
                (habitat_results.get("uncertainty"), "Prediction Uncertainty", "uncertainty.png"),
            ]
            
            for arr, title, name in advanced_plots:
                if arr is not None:
                    out_path = os.path.join(out_dir, name)
                    plot_field(arr, lon, lat, title, out_path)
                    print(f"Saved {out_path}")
            
        except Exception as e:
            print(f"⚠️  Advanced model failed, falling back to basic model: {e}")
            use_advanced_model = False
    
    if not use_advanced_model:
        # Original basic model as fallback
        fields = load_core_fields(input_path, overrides=overrides if overrides else None)
        try:
            feats = compute_hsi(fields)
        except Exception as e:
            error_path = os.path.join(out_dir, "ERROR.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(str(e))
                f.write("\n\nIf variables were not detected, check outputs/dataset_summary.txt and outputs/group_tree.txt and create overrides.json like:\n")
                f.write('{"lat": "latitude", "lon": "longitude", "chlor": "chlor_a", "kd490": "kd_490", "bbp": "bbp_443"}\n')
            print(f"Could not compute HSI. See {error_path} and outputs/dataset_summary.txt")
            return

        # Save basic plots
        lon = fields.get("lon")
        lat = fields.get("lat")

        to_plot = [
            (feats.get("chl_scaled"), "Chlorophyll (scaled)", "chl_scaled.png"),
            (feats.get("kd_scaled"), "-Kd490 (scaled)", "kd_scaled.png"),
            (feats.get("bbp_scaled"), "bbp (scaled)", "bbp_scaled.png"),
            (feats.get("fronts"), "Fronts (grad mag of chl)", "fronts.png"),
            (feats.get("bloom"), "Bloom mask (>=85th %)", "bloom.png"),
            (feats.get("HSI"), "Habitat Suitability Index (0-1)", "HSI.png"),
        ]

        for arr, title, name in to_plot:
            if arr is None:
                continue
            out_path = os.path.join(out_dir, name)
            plot_field(arr, lon, lat, title, out_path)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA Space Apps Challenge - Shark Habitat Prediction from PACE L2 OC/BGC data")
    parser.add_argument("input", nargs="?", default="PACE_OCI.20251004T015456.L2.OC_BGC.V3_1.NRT.nc", help="Path to PACE NetCDF file")
    parser.add_argument("--out", default="outputs", help="Output directory for figures")
    parser.add_argument("--overrides", default="overrides.json", help="Optional JSON mapping for variable overrides")
    parser.add_argument("--basic", action="store_true", help="Use basic model only (skip advanced features)")
    args = parser.parse_args()

    run(args.input, args.out, args.overrides, use_advanced_model=not args.basic)
