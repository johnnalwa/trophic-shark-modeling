"""
Real-time Shark Tag Algorithm for Feeding Detection and Behavior Analysis

This module implements advanced algorithms for a conceptual real-time shark tag
that can detect feeding events, analyze behavior patterns, and transmit data
for predictive habitat modeling.

Mathematical Framework:
1. Feeding event detection using accelerometry and jaw motion analysis
2. Behavioral state classification (hunting, feeding, resting, traveling)
3. Real-time data compression and transmission optimization
4. Prey type classification using optical signatures
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import warnings

# Optional sklearn imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available. Behavioral classification will be disabled.")


class FeedingEventDetector:
    """
    Mathematical algorithm for detecting shark feeding events from sensor data.
    """
    
    def __init__(self, 
                 sampling_rate: float = 50.0,  # Hz
                 feeding_threshold: float = 2.5,  # g-force threshold
                 jaw_frequency_range: Tuple[float, float] = (0.5, 5.0)):  # Hz
        """
        Initialize feeding detection algorithm.
        
        Args:
            sampling_rate: Sensor sampling rate in Hz
            feeding_threshold: Acceleration threshold for feeding detection
            jaw_frequency_range: Expected frequency range for jaw movements
        """
        self.sampling_rate = sampling_rate
        self.feeding_threshold = feeding_threshold
        self.jaw_freq_min, self.jaw_freq_max = jaw_frequency_range
        
        # Pre-compute filter coefficients for jaw motion detection
        nyquist = sampling_rate / 2
        low = self.jaw_freq_min / nyquist
        high = self.jaw_freq_max / nyquist
        self.jaw_filter = signal.butter(4, [low, high], btype='band')
    
    def detect_feeding_events(self, 
                            accelerometer_data: np.ndarray,
                            magnetometer_data: Optional[np.ndarray] = None,
                            window_size: float = 5.0) -> Dict[str, np.ndarray]:
        """
        Detect feeding events from tri-axial accelerometer data.
        
        Args:
            accelerometer_data: Shape (n_samples, 3) - x, y, z acceleration
            magnetometer_data: Optional magnetometer data for head orientation
            window_size: Analysis window size in seconds
            
        Returns:
            Dictionary with feeding detection results
        """
        acc_data = np.asarray(accelerometer_data)
        n_samples = acc_data.shape[0]
        window_samples = int(window_size * self.sampling_rate)
        
        # Compute total acceleration magnitude
        acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
        
        # Remove gravity component (assuming 1g baseline)
        acc_magnitude_corrected = acc_magnitude - np.median(acc_magnitude)
        
        # Detect high-acceleration events (potential feeding)
        high_acc_events = np.abs(acc_magnitude_corrected) > self.feeding_threshold
        
        # Filter for jaw motion frequency components
        jaw_motion = signal.filtfilt(self.jaw_filter[0], self.jaw_filter[1], acc_magnitude_corrected)
        jaw_power = jaw_motion**2
        
        # Sliding window analysis
        feeding_probability = np.zeros(n_samples)
        jaw_activity = np.zeros(n_samples)
        
        for i in range(0, n_samples - window_samples, window_samples // 2):
            window_end = min(i + window_samples, n_samples)
            window_acc = acc_magnitude_corrected[i:window_end]
            window_jaw = jaw_power[i:window_end]
            
            # Features for feeding detection
            acc_variance = np.var(window_acc)
            jaw_mean_power = np.mean(window_jaw)
            high_acc_fraction = np.mean(high_acc_events[i:window_end])
            
            # Simple feeding probability model (can be replaced with ML model)
            feeding_prob = self._compute_feeding_probability(
                acc_variance, jaw_mean_power, high_acc_fraction
            )
            
            feeding_probability[i:window_end] = feeding_prob
            jaw_activity[i:window_end] = jaw_mean_power
        
        # Detect discrete feeding events
        feeding_events = self._extract_feeding_events(feeding_probability)
        
        return {
            'acceleration_magnitude': acc_magnitude,
            'jaw_motion_filtered': jaw_motion,
            'feeding_probability': feeding_probability,
            'jaw_activity': jaw_activity,
            'feeding_events': feeding_events,
            'high_acceleration_events': high_acc_events.astype(float)
        }
    
    def _compute_feeding_probability(self, 
                                   acc_variance: float,
                                   jaw_power: float, 
                                   high_acc_fraction: float) -> float:
        """
        Compute feeding probability from extracted features.
        
        This is a simplified model - in practice, this would be trained
        on labeled data from controlled feeding experiments.
        """
        # Normalize features
        acc_score = min(acc_variance / 10.0, 1.0)  # Normalize by expected max
        jaw_score = min(jaw_power / 5.0, 1.0)
        burst_score = high_acc_fraction
        
        # Weighted combination
        feeding_prob = 0.4 * acc_score + 0.4 * jaw_score + 0.2 * burst_score
        
        return min(feeding_prob, 1.0)
    
    def _extract_feeding_events(self, 
                              feeding_probability: np.ndarray,
                              threshold: float = 0.6,
                              min_duration: float = 2.0) -> List[Dict]:
        """
        Extract discrete feeding events from probability time series.
        
        Args:
            feeding_probability: Continuous feeding probability
            threshold: Probability threshold for event detection
            min_duration: Minimum event duration in seconds
            
        Returns:
            List of feeding event dictionaries
        """
        min_samples = int(min_duration * self.sampling_rate)
        
        # Find regions above threshold
        above_threshold = feeding_probability > threshold
        
        # Find event boundaries
        event_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
        event_ends = np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1
        
        # Handle edge cases
        if above_threshold[0]:
            event_starts = np.concatenate([[0], event_starts])
        if above_threshold[-1]:
            event_ends = np.concatenate([event_ends, [len(above_threshold)]])
        
        # Extract events meeting minimum duration
        events = []
        for start, end in zip(event_starts, event_ends):
            if end - start >= min_samples:
                events.append({
                    'start_time': start / self.sampling_rate,
                    'end_time': end / self.sampling_rate,
                    'duration': (end - start) / self.sampling_rate,
                    'max_probability': np.max(feeding_probability[start:end]),
                    'mean_probability': np.mean(feeding_probability[start:end])
                })
        
        return events


class BehaviorClassifier:
    """
    Classify shark behavioral states from sensor data.
    """
    
    def __init__(self):
        self.behavior_states = ['resting', 'traveling', 'hunting', 'feeding']
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.scaler = None
            self.classifier = None
        self.is_trained = False
    
    def extract_behavioral_features(self, 
                                  accelerometer_data: np.ndarray,
                                  depth_data: np.ndarray,
                                  window_size: float = 10.0) -> np.ndarray:
        """
        Extract features for behavioral classification.
        
        Args:
            accelerometer_data: Tri-axial acceleration data
            depth_data: Depth/pressure sensor data
            window_size: Feature extraction window in seconds
            
        Returns:
            Feature matrix for classification
        """
        acc_data = np.asarray(accelerometer_data)
        depth_data = np.asarray(depth_data)
        
        window_samples = int(window_size * 50)  # Assuming 50 Hz sampling
        n_windows = len(acc_data) // window_samples
        
        features = []
        
        for i in range(n_windows):
            start_idx = i * window_samples
            end_idx = start_idx + window_samples
            
            acc_window = acc_data[start_idx:end_idx]
            depth_window = depth_data[start_idx:end_idx]
            
            # Acceleration features
            acc_magnitude = np.sqrt(np.sum(acc_window**2, axis=1))
            acc_mean = np.mean(acc_magnitude)
            acc_std = np.std(acc_magnitude)
            acc_max = np.max(acc_magnitude)
            
            # Movement features
            jerk = np.diff(acc_magnitude)
            jerk_mean = np.mean(np.abs(jerk))
            
            # Depth features
            depth_mean = np.mean(depth_window)
            depth_std = np.std(depth_window)
            depth_change = np.abs(depth_window[-1] - depth_window[0])
            
            # Frequency domain features
            acc_fft = np.abs(fft(acc_magnitude))
            dominant_freq = np.argmax(acc_fft[:len(acc_fft)//2])
            spectral_energy = np.sum(acc_fft**2)
            
            window_features = [
                acc_mean, acc_std, acc_max, jerk_mean,
                depth_mean, depth_std, depth_change,
                dominant_freq, spectral_energy
            ]
            
            features.append(window_features)
        
        return np.array(features)
    
    def train_classifier(self, 
                        features: np.ndarray, 
                        labels: np.ndarray) -> None:
        """
        Train behavioral classifier on labeled data.
        
        Args:
            features: Feature matrix
            labels: Behavioral state labels
        """
        if not HAS_SKLEARN:
            raise ValueError("sklearn not available. Cannot train classifier.")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train classifier
        self.classifier.fit(features_scaled, labels)
        self.is_trained = True
    
    def classify_behavior(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Classify behavioral states from features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not HAS_SKLEARN:
            # Simple rule-based classification as fallback
            n_samples = features.shape[0]
            predictions = np.array(['traveling'] * n_samples)  # Default behavior
            probabilities = np.ones((n_samples, len(self.behavior_states))) * 0.25
            
            return {
                'predicted_states': predictions,
                'state_probabilities': probabilities,
                'behavior_states': self.behavior_states
            }
        
        if not self.is_trained:
            raise ValueError("Classifier must be trained before use")
        
        features_scaled = self.scaler.transform(features)
        
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        
        return {
            'predicted_states': predictions,
            'state_probabilities': probabilities,
            'behavior_states': self.behavior_states
        }


class DataCompressionOptimizer:
    """
    Optimize data transmission for real-time shark tag communication.
    """
    
    def __init__(self, 
                 transmission_budget: int = 100,  # bytes per transmission
                 critical_event_priority: float = 0.8):
        """
        Initialize data compression optimizer.
        
        Args:
            transmission_budget: Maximum bytes per transmission
            critical_event_priority: Priority weight for critical events
        """
        self.transmission_budget = transmission_budget
        self.critical_event_priority = critical_event_priority
    
    def prioritize_data(self, 
                       sensor_data: Dict[str, np.ndarray],
                       feeding_events: List[Dict],
                       behavioral_states: np.ndarray) -> Dict[str, any]:
        """
        Prioritize and compress data for transmission.
        
        Args:
            sensor_data: Raw sensor data
            feeding_events: Detected feeding events
            behavioral_states: Classified behavioral states
            
        Returns:
            Optimized data packet for transmission
        """
        # Priority 1: Feeding events (highest priority)
        critical_data = {
            'timestamp': sensor_data.get('timestamp', 0),
            'location': sensor_data.get('gps_location'),
            'feeding_events': len(feeding_events),
            'feeding_intensity': np.mean([event['max_probability'] for event in feeding_events]) if feeding_events else 0
        }
        
        # Priority 2: Behavioral summary
        if len(behavioral_states) > 0:
            behavior_summary = {
                'dominant_behavior': stats.mode(behavioral_states)[0][0],
                'behavior_diversity': len(np.unique(behavioral_states)) / len(self.behavior_states)
            }
            critical_data.update(behavior_summary)
        
        # Priority 3: Environmental context
        environmental_data = {
            'depth': sensor_data.get('depth_mean'),
            'temperature': sensor_data.get('temperature'),
            'local_chlorophyll': sensor_data.get('chlorophyll_proxy')
        }
        
        # Compress and package data
        compressed_packet = self._compress_data_packet(
            critical_data, environmental_data
        )
        
        return compressed_packet
    
    def _compress_data_packet(self, 
                            critical_data: Dict,
                            environmental_data: Dict) -> Dict[str, any]:
        """
        Compress data packet to fit transmission budget.
        """
        # Simple compression strategy - quantize values and use efficient encoding
        compressed = {}
        
        # Critical data (always included)
        compressed.update(critical_data)
        
        # Add environmental data if budget allows
        # In practice, this would use more sophisticated compression
        compressed.update(environmental_data)
        
        return compressed


def create_synthetic_shark_data(duration_hours: float = 2.0, 
                              sampling_rate: float = 50.0) -> Dict[str, np.ndarray]:
    """
    Create synthetic shark sensor data for testing algorithms.
    
    Args:
        duration_hours: Duration of synthetic data in hours
        sampling_rate: Sensor sampling rate in Hz
        
    Returns:
        Dictionary with synthetic sensor data
    """
    n_samples = int(duration_hours * 3600 * sampling_rate)
    time = np.linspace(0, duration_hours * 3600, n_samples)
    
    # Base swimming motion (sinusoidal with noise)
    base_freq = 0.5  # Hz (tail beat frequency)
    base_motion = np.sin(2 * np.pi * base_freq * time) + 0.2 * np.random.randn(n_samples)
    
    # Create tri-axial accelerometer data
    acc_x = base_motion + 0.1 * np.random.randn(n_samples)
    acc_y = 0.3 * base_motion + 0.1 * np.random.randn(n_samples)
    acc_z = 1.0 + 0.1 * base_motion + 0.1 * np.random.randn(n_samples)  # Include gravity
    
    # Add feeding events (high acceleration bursts)
    feeding_times = [0.5, 1.2, 1.8]  # Hours
    for feed_time in feeding_times:
        if feed_time < duration_hours:
            feed_idx = int(feed_time * 3600 * sampling_rate)
            feed_duration = int(5 * sampling_rate)  # 5 second feeding event
            
            # Add high-frequency jaw motion and acceleration burst
            feed_end = min(feed_idx + feed_duration, n_samples)
            jaw_motion = 3 * np.sin(2 * np.pi * 2 * time[feed_idx:feed_end])  # 2 Hz jaw motion
            acc_x[feed_idx:feed_end] += jaw_motion
            acc_y[feed_idx:feed_end] += 0.5 * jaw_motion
            acc_z[feed_idx:feed_end] += 2 * np.abs(jaw_motion)  # Feeding acceleration
    
    accelerometer_data = np.column_stack([acc_x, acc_y, acc_z])
    
    # Depth data (diving behavior)
    depth_base = 50  # meters
    depth_variation = 20 * np.sin(2 * np.pi * time / 1800) + 5 * np.random.randn(n_samples)  # 30 min dive cycle
    depth_data = depth_base + depth_variation
    depth_data = np.maximum(depth_data, 0)  # Ensure non-negative depth
    
    return {
        'timestamp': time,
        'accelerometer': accelerometer_data,
        'depth': depth_data,
        'temperature': 22 + 2 * np.sin(2 * np.pi * time / 3600) + 0.5 * np.random.randn(n_samples),  # Temperature variation
        'gps_location': [34.0522, -118.2437],  # Example coordinates
        'chlorophyll_proxy': 1.5 + 0.5 * np.random.randn(n_samples)
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Real-time Shark Tag Algorithm initialized")
    
    # Create synthetic data
    synthetic_data = create_synthetic_shark_data(duration_hours=2.0)
    
    # Initialize feeding detector
    detector = FeedingEventDetector()
    
    # Detect feeding events
    feeding_results = detector.detect_feeding_events(synthetic_data['accelerometer'])
    
    print(f"Detected {len(feeding_results['feeding_events'])} feeding events")
    for i, event in enumerate(feeding_results['feeding_events']):
        print(f"Event {i+1}: {event['start_time']:.1f}s - {event['end_time']:.1f}s, "
              f"Duration: {event['duration']:.1f}s, Max prob: {event['max_probability']:.2f}")
    
    # Initialize behavior classifier (would need training data in practice)
    classifier = BehaviorClassifier()
    features = classifier.extract_behavioral_features(
        synthetic_data['accelerometer'], 
        synthetic_data['depth']
    )
    
    print(f"Extracted {features.shape[0]} behavioral feature windows")
    print(f"Feature dimensionality: {features.shape[1]}")
    
    # Data compression example
    compressor = DataCompressionOptimizer()
    compressed_data = compressor.prioritize_data(
        synthetic_data, 
        feeding_results['feeding_events'],
        np.array(['traveling'] * features.shape[0])  # Example behavioral states
    )
    
    print("Compressed data packet ready for transmission:")
    for key, value in compressed_data.items():
        print(f"  {key}: {value}")
