# 🦈 Trophic Shark Modeling - NASA Space Apps Challenge

A comprehensive mathematical framework for predicting shark habitat suitability using NASA PACE satellite data, incorporating trophic cascade modeling and advanced oceanographic features.

## 🌊 Project Overview

This project addresses the NASA Space Apps Challenge by developing an integrated system that:
- Predicts shark habitat suitability using satellite oceanographic data
- Models trophic cascades with realistic time lags
- Incorporates advanced oceanographic features (fronts, eddies, thermal zones)
- Demonstrates real-time shark tag integration concepts
- Provides educational visualizations for conservation awareness

## 🚀 Features

### Core Capabilities
- **Trophic Lag Modeling**: Implements realistic time delays in the food web (phytoplankton → zooplankton → fish → sharks)
- **Advanced Habitat Suitability Index (HSI)**: Multi-factor habitat prediction with uncertainty quantification
- **Oceanographic Feature Detection**: Identifies ocean fronts, eddies, and thermal zones
- **Educational Visualizations**: High school-friendly graphics explaining marine ecosystems
- **Real-time Tag Simulation**: Demonstrates feeding event detection and behavioral classification

### Mathematical Framework
- **Trophic System Lag**: ~30 days total (5 + 10 + 15+ day cascading delays)
- **Prey Density Modeling**: Spatially explicit food availability estimation
- **Thermal Habitat Preferences**: Species-specific temperature suitability
- **Uncertainty Quantification**: Confidence intervals for all predictions
- **Multi-scale Analysis**: From satellite pixels to ecosystem-wide patterns

## 📊 Input Data

The system processes NASA PACE L2 Ocean Color/Biogeochemical data:
- **Chlorophyll-a concentration** (`chlor_a`)
- **Diffuse attenuation coefficient** (`kd_490`)
- **Backscattering coefficient** (`bbp_443`)
- **Geographic coordinates** (`latitude`, `longitude`)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd trophic-shark-modeling

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python main.py [input_file.nc]
```

## 📈 Usage

### Basic Usage
```bash
# Process PACE data with default settings
python main.py PACE_OCI.20251004T015456.L2.OC_BGC.V3_1.NRT.nc

# Specify custom output directory
python main.py input.nc --out custom_outputs

# Use basic model only (faster processing)
python main.py input.nc --basic
```

### Variable Override Configuration
If automatic variable detection fails, create `overrides.json`:
```json
{
    "lat": "latitude",
    "lon": "longitude", 
    "chlor": "chlor_a",
    "kd490": "kd_490",
    "bbp": "bbp_443"
}
```

### Advanced Configuration
```bash
# Use custom variable mappings
python main.py input.nc --overrides custom_overrides.json
```

## 📁 Project Structure

```
trophic-shark-modeling/
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── src/                        # Source code modules
│   ├── data_loader.py         # NetCDF data processing
│   ├── features.py            # Basic habitat features
│   ├── trophic_model.py       # Trophic cascade modeling
│   ├── advanced_features.py   # Advanced oceanographic features
│   ├── shark_tag_algorithm.py # Tag simulation and analysis
│   ├── integrated_pipeline.py # Main prediction framework
│   └── plotting.py            # Visualization utilities
├── outputs/                   # Generated results
│   ├── *.png                 # Habitat prediction maps
│   ├── dataset_summary.txt   # Input data analysis
│   └── nasa_challenge_summary.txt # Results summary
└── README.md                 # This file
```

## 🎯 Output Products

### Habitat Prediction Maps
- **Advanced_HSI.png**: Comprehensive habitat suitability index
- **prey_density.png**: Estimated food availability
- **thermal_suitability.png**: Temperature-based habitat quality
- **frontal_zones.png**: Ocean front detection
- **uncertainty.png**: Prediction confidence levels

### Educational Visualizations
- **educational_shark_habitat_prediction.png**: Overview for students
- **educational_trophic_cascade.png**: Food web explanation
- **educational_habitat_components.png**: Feature breakdown
- **educational_prediction_uncertainty.png**: Uncertainty communication

### Analysis Reports
- **dataset_summary.txt**: Input data characteristics
- **nasa_challenge_summary.txt**: Complete results summary
- **group_tree.txt**: NetCDF file structure analysis

## 🔬 Scientific Approach

### Trophic Cascade Modeling
The system implements realistic time lags in marine food webs:
1. **Phytoplankton bloom** (satellite-detected chlorophyll)
2. **Zooplankton response** (5-day lag)
3. **Small fish aggregation** (10-day additional lag)
4. **Shark habitat suitability** (15+ day additional lag)

### Advanced Features
- **Ocean Front Detection**: Gradient analysis of chlorophyll fields
- **Eddy Identification**: Circulation pattern recognition
- **Thermal Habitat Modeling**: Species-specific temperature preferences
- **Uncertainty Quantification**: Monte Carlo-based confidence estimation

### Real-time Tag Integration
Demonstrates concepts for:
- Feeding event detection from accelerometer data
- Behavioral state classification
- Data compression for satellite transmission
- Real-time habitat validation

## 🎓 Educational Impact

The project creates accessible visualizations explaining:
- How satellites monitor ocean ecosystems
- Trophic cascades in marine food webs
- Shark habitat requirements and conservation
- Uncertainty in scientific predictions
- Real-world applications of space technology

## 🌍 Conservation Applications

### Management Support
- **Habitat Hotspot Identification**: Priority areas for protection
- **Seasonal Prediction**: Timing of shark aggregations
- **Fishing Impact Assessment**: Overlap analysis with commercial activities
- **Climate Change Monitoring**: Long-term habitat trend analysis

### Research Integration
- **Tagging Study Design**: Optimal deployment locations
- **Validation Framework**: Ground-truth comparison protocols
- **Multi-species Modeling**: Extensible to other marine predators
- **Ecosystem Monitoring**: Integrated ocean health assessment

## 🔧 Technical Requirements

### System Requirements
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ free space for outputs
- **Processing**: Multi-core CPU beneficial for large datasets

### Data Requirements
- NASA PACE L2 Ocean Color/Biogeochemical NetCDF files
- Geographic coverage: Global ocean regions
- Temporal resolution: Daily to weekly composites

## 🤝 Contributing

We welcome contributions to improve the modeling framework:
1. Fork the repository
2. Create a feature branch
3. Implement improvements with tests
4. Submit a pull request with detailed description

### Development Areas
- Species-specific habitat models
- Additional satellite data integration
- Real-time processing optimization
- Validation with field data
- User interface development

## 📚 References

### Scientific Background
- NASA PACE Mission: Ocean color and biogeochemistry
- Marine trophic cascade dynamics
- Shark habitat ecology and behavior
- Satellite oceanography applications

### Technical Documentation
- NetCDF data format specifications
- Python scientific computing ecosystem
- Geospatial analysis best practices
- Marine data visualization standards

## 🏆 NASA Space Apps Challenge

This project addresses multiple challenge requirements:
- ✅ Mathematical framework for shark identification
- ✅ NASA satellite data integration
- ✅ Trophic step consideration
- ✅ Physical oceanographic features
- ✅ Real-time tag concept demonstration
- ✅ Educational content creation
- ✅ Conservation applications

## 📞 Contact

For questions about the modeling framework or collaboration opportunities, please open an issue in the repository.

---

*Developed for the NASA Space Apps Challenge - Connecting space technology with ocean conservation* 🌊🛰️🦈
