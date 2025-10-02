"""
Machine Learning models for security smell detection.
Implements various ML algorithms and ensemble methods.
"""
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
 


@dataclass
class PredictionResult:
    """Result from ML prediction."""
    smell_type: str
    confidence: float
    features_importance: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None


class SecuritySmellClassifier:
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
        # Security smell classes
        self.smell_classes = [
            'HTTP_WITHOUT_TLS',
            'OPEN_BINDING', 
            'HARDCODED_SECRET',
            'WEAK_PASSWORD',
            'WEAK_CRYPTOGRAPHY',
            'NO_SMELL'  # Clean code
        ]
        
    def _create_model(self):
        """Create the ML model based on type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=150,
                max_depth=10,
                learning_rate=0.1,
                objective='multi:softprob',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'neural_net':
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        
        elif self.model_type == 'ensemble':
            # Create ensemble of models
            models = [
                ('rf', RandomForestClassifier(
                    n_estimators=150, max_depth=15, 
                    class_weight='balanced', random_state=42
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=100, max_depth=8,
                    learning_rate=0.1, random_state=42
                )),
                ('lr', LogisticRegression(
                    max_iter=1000, class_weight='balanced',
                    random_state=42
                ))
            ]
            
            models.append(('xgb', xgb.XGBClassifier(
                    n_estimators=100, max_depth=8,
                    use_label_encoder=False, random_state=42
                )))
            
            return VotingClassifier(
                estimators=models,
                voting='soft',
                n_jobs=-1
            )
        
        else:
            # Default to Random Forest
            return RandomForestClassifier(
                n_estimators=150,
                class_weight='balanced',
                random_state=42
            )
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Train the classifier
        """
        print(f"Training {self.model_type} classifier...")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and train model
        self.model = self._create_model()
        
        # Use cross validation for better generalization
        cv_scores = cross_val_score(
            self.model, X_scaled, y_encoded, 
            cv=5, scoring='f1_weighted'
        )
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train on full dataset
        self.model.fit(X_scaled, y_encoded)
        
        # Calibrate probabilities for better confidence scores
        self.model = CalibratedClassifierCV(self.model, cv=3)
        self.model.fit(X_scaled, y_encoded)
        
        self.is_trained = True
        print("Training completed!")
    
    def predict(self, X: np.ndarray) -> List[PredictionResult]:
        """
        Predict security smells
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            # Get predicted class and confidence
            smell_type = self.label_encoder.inverse_transform([pred])[0]
            confidence = probs[pred]
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                if self.feature_names:
                    feature_importance = dict(zip(self.feature_names, importance))
            
            results.append(PredictionResult(
                smell_type=smell_type,
                confidence=float(confidence),
                features_importance=feature_importance
            ))
        
        return results
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for all classes."""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Get predictions
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        # Calculate metrics
        report = classification_report(
            y_test_encoded, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(y_test_encoded, y_pred)
        
        # Calculate AUC for multiclass
        try:
            auc_score = roc_auc_score(
                y_test_encoded, y_proba,
                multi_class='ovr', average='weighted'
            )
        except:
            auc_score = None
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'auc_score': auc_score,
            'f1_weighted': f1_score(y_test_encoded, y_pred, average='weighted')
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if not self.is_trained:
            return None
        
        # Extract feature importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importance = np.abs(self.model.coef_).mean(axis=0)
        else:
            return None
        
        if self.feature_names:
            importance_dict = dict(zip(self.feature_names, importance))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return None
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {path}")


class HybridDetector:
    """
    Hybrid detector combining ML predictions with rule based validation
    """
    
    def __init__(self, ml_classifier: SecuritySmellClassifier, 
                 confidence_threshold: float = 0.7):
        self.ml_classifier = ml_classifier
        self.confidence_threshold = confidence_threshold
        
    def detect(self, features: np.ndarray, 
               rule_based_result: Optional[str] = None) -> Tuple[str, float, str]:
        """
        Perform hybrid detection
        """
        # Get ML prediction
        ml_results = self.ml_classifier.predict(features.reshape(1, -1))
        ml_result = ml_results[0]
        
        # Decision logic
        if ml_result.confidence >= self.confidence_threshold:
            # High confidence ML prediction
            if ml_result.smell_type == 'NO_SMELL':
                if rule_based_result:
                    # ML says clean but rules found something
                    # Trust rules for critical issues
                    return rule_based_result, 0.5, 'rule_override'
                else:
                    return 'NO_SMELL', ml_result.confidence, 'ml_high_conf'
            else:
                # ML found a smell with high confidence
                return ml_result.smell_type, ml_result.confidence, 'ml_high_conf'
        
        elif rule_based_result:
            # Low ML confidence but rules found something
            return rule_based_result, 0.6, 'rule_based'
        
        elif ml_result.confidence >= 0.5:
            # Medium confidence ML prediction
            return ml_result.smell_type, ml_result.confidence, 'ml_medium_conf'
        
        else:
            # Low confidence, no smell detected
            return 'NO_SMELL', ml_result.confidence, 'low_conf'
    
    def batch_detect(self, features: np.ndarray,
                    rule_based_results: List[Optional[str]] = None) -> List[Tuple[str, float, str]]:
        """
        Perform batch hybrid detection
        """
        if rule_based_results is None:
            rule_based_results = [None] * len(features)
        
        results = []
        for i, feature_vector in enumerate(features):
            result = self.detect(feature_vector, rule_based_results[i])
            results.append(result)
        
        return results