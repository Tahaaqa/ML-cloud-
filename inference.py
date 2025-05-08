import os
import numpy as np
import pandas as pd
from model_loader import ModelLoader

class InferenceEngine:
    def __init__(self, model_dir="inference_model"):
        """Initialize the inference engine with model loader"""
        self.model_loader = ModelLoader(os.path.join(os.path.dirname(__file__), "inference_model"))
        self.model = self.model_loader.get_model()
        self.anomaly_threshold = self.model_loader.get_anomaly_threshold()
        self.sequence_ready = False
        self.last_prediction = None
        self.prediction_count = 0
        
        # Output RULA column names for result mapping
        self.rula_columns = [
            "right_shoulder_rula", "left_shoulder_rula",
            "right_elbow_rula", "left_elbow_rula",
            "right_wrist_rula", "left_wrist_rula",
            "torso_rula", "neck_rula", "total_rula"
        ]
    
    def process_frame(self, angles_dict):
        """
        Process a frame of joint angles and make predictions if enough frames are collected
        
        Args:
            angles_dict: Dictionary with joint angle measurements
            
        Returns:
            Dictionary containing predictions and anomaly status
        """
        # Preprocess the angles
        processed_angles = self.model_loader.preprocess_angles(angles_dict)
        
        # Try to create a sequence
        sequence = self.model_loader.create_sequence()
        
        if sequence is not None:
            # We have enough frames to make a prediction
            self.sequence_ready = True
            predictions = self.make_prediction(sequence)
            self.last_prediction = predictions
            self.prediction_count += 1
            return predictions
        else:
            # Not enough frames yet, sequence is building
            self.sequence_ready = False
            return {
                "status": "collecting_sequence",
                "frames_collected": len(self.model_loader.angle_buffer),
                "frames_needed": self.model_loader.seq_length
            }
    
    def make_prediction(self, sequence):
        """
        Make predictions using the model
        
        Args:
            sequence: Preprocessed sequence of joint angles
            
        Returns:
            Dictionary with RULA predictions and anomaly detection
        """
        # Run inference
        model_outputs = self.model.predict(sequence, verbose=0)
        
        # Extract RULA scores and anomaly prediction
        if isinstance(model_outputs, list):
            # Multi-output model (RULA scores + anomaly)
            rula_scores = model_outputs[0][0]  # First output tensor, first batch item
            anomaly_prob = model_outputs[1][0][0]  # Second output tensor, first batch item, first (only) value
        else:
            # Single output model (just RULA scores)
            rula_scores = model_outputs[0]
            anomaly_prob = 0.0  # Default if model doesn't predict anomaly
        
        # Map scores to output columns
        predictions = {}
        for i, col in enumerate(self.rula_columns):
            if i < len(rula_scores):
                # Round RULA scores appropriately (they should be integers)
                predictions[col] = round(float(rula_scores[i]))
        
        # Determine if this is an anomalous posture
        is_anomaly = anomaly_prob > self.anomaly_threshold
        
        # Add anomaly information
        predictions["anomaly_probability"] = float(anomaly_prob)
        predictions["anomaly_detected"] = bool(is_anomaly)
        predictions["anomaly_threshold"] = self.anomaly_threshold
        
        # Generate risk assessments
        for col in self.rula_columns:
            if col in predictions:
                risk_col = col + "_Risk"
                predictions[risk_col] = self.evaluate_rula_risk(predictions[col])
        
        return predictions
    
    def evaluate_rula_risk(self, score):
        """Evaluate RULA risk level"""
        if score <= 2:
            return "Risque Faible"
        elif score <= 4:
            return "Risque Modéré"
        elif score <= 6:
            return "Risque Élevé"
        else:
            return "Risque Très Élevé"
    
    def get_status(self):
        """Get the current status of the inference engine"""
        return {
            "sequence_ready": self.sequence_ready,
            "frames_collected": len(self.model_loader.angle_buffer) if hasattr(self.model_loader, "angle_buffer") else 0,
            "frames_needed": self.model_loader.seq_length,
            "predictions_made": self.prediction_count,
            "last_prediction_time": pd.Timestamp.now().isoformat() if self.last_prediction else None
        }