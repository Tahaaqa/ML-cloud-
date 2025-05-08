import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import RobustScaler
from collections import deque
from dotenv import load_dotenv

class ModelLoader:
    def __init__(self, model_dir="None"):
        self.model_dir =  model_dir or os.getenv("MODEL_DIR", "inference_model")
        self.model_path = os.path.join(self.model_dir, "posture_model.keras")
        self.scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        self.config_path = os.path.join(self.model_dir, "config.json")
        
        # Model parameters (will be loaded from config)
        self.seq_length = None
        self.n_features = None
        self.feature_names = None
        self.anomaly_threshold = None
        self.input_columns = None
        self.output_columns = None
        
        # Load the model components
        self.model = None
        self.scaler = None
        self.config = None
        
        # Buffer for sequence construction
        self.angle_buffer = None
        
        self._load_components()
    
    def _load_components(self):
        """Load model, scaler and configuration"""
        # Check if model directory exists
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Load configuration
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                self.seq_length = self.config.get('seq_length', 30)
                self.n_features = self.config.get('n_features', 24)  # 8 original + 8 diff + 8 stats
                self.feature_names = self.config.get('feature_names', [])
                self.anomaly_threshold = self.config.get('anomaly_threshold', 0.5)
                self.input_columns = self.config.get('input_columns', [])
                self.output_columns = self.config.get('output_columns', [])
        except Exception as e:
            raise ValueError(f"Error loading model configuration: {e}")
        
        # Load scaler
        try:
            import joblib
            self.scaler = joblib.load(self.scaler_path)

        except Exception as e:
            raise ValueError(f"Error loading scaler: {e}")
        
        # Load TensorFlow model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Model expects input shape: {self.model.input_shape}")
            print(f"Model outputs: {len(self.model.outputs)} tensors")
        except Exception as e:
            raise ValueError(f"Error loading TensorFlow model: {e}")
        
        # Initialize buffer for sequence construction
        self.angle_buffer = deque(maxlen=self.seq_length)
        
        print("All model components loaded successfully")
    
    def preprocess_angles(self, angles_dict):
    # CamelCase (model) -> snake_case (PoseAnalyzer) mapping
        key_mapping = {
        "Right_Elbow_Angle": "right_elbow",
        "Left_Elbow_Angle": "left_elbow",
        "Right_Shoulder_Angle": "right_shoulder",
        "Left_Shoulder_Angle": "left_shoulder",
        "Right_Wrist_Rotation": "right_wrist_rotation",
        "Left_Wrist_Rotation": "left_wrist_rotation",
        "Torso_Angle": "torso_angle",
        "Neck_Angle": "neck_angle"
             }

    # Use CamelCase columns from config (old format)
        angle_columns = self.input_columns or list(key_mapping.keys())

    # Extract values using reverse mapping
        try:
            angles_array = np.array([
            angles_dict[key_mapping[col]]  # Map to snake_case
            for col in angle_columns
        ]).reshape(1, -1)
        except KeyError as e:
            missing_key = str(e).strip("'")
            camel_key = [k for k, v in key_mapping.items() if v == missing_key][0]
            raise ValueError(f"PoseAnalyzer missing key: {missing_key} (needed for model column: {camel_key})")

    # Scale and buffer
        scaled_angles = self.scaler.transform(angles_array).flatten()
        self.angle_buffer.append(scaled_angles)
    
        return scaled_angles
    
    def create_sequence(self):
        """Create a sequence from the buffer for model inference"""
        # If buffer is not full yet, return None
        if len(self.angle_buffer) < self.seq_length:
            return None
        
        # Convert buffer to numpy array
        angles_seq = np.array(list(self.angle_buffer))
        
        # Calculate temporal differences (velocity)
        diffs = np.zeros_like(angles_seq)
        diffs[1:] = angles_seq[1:] - angles_seq[:-1]
        
        # Calculate rolling means
        window_size = 5  # Small window for stats
        stats = np.zeros_like(angles_seq)
        
        for i in range(len(angles_seq)):
            start_idx = max(0, i - window_size + 1)
            stats[i] = np.mean(angles_seq[start_idx:i+1], axis=0)
        
        # Combine original, diffs, and stats
        combined = np.concatenate([
            angles_seq,
            diffs,
            stats
        ], axis=1)
        
        # Reshape for model input [batch_size=1, seq_length, n_features]
        model_input = combined.reshape(1, self.seq_length, -1)
        
        return model_input
    
    def get_model(self):
        """Return the loaded model"""
        return self.model
    
    def get_anomaly_threshold(self):
        """Return the configured anomaly threshold"""
        return self.anomaly_threshold