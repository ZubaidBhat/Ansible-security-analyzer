"""
Explainability module for ML-based security smell detection.
"""
import numpy as np
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non interactive backend
import shap
import lime
from lime.tabular import LimeTabularExplainer

from ml_models import SecuritySmellClassifier


class ExplainabilityEngine:
    """Provide explanations for ML model predictions."""
    
    def __init__(self, classifier: SecuritySmellClassifier):
        self.classifier = classifier
        self.feature_names = classifier.feature_names or []
        self.explainer_shap = None
        self.explainer_lime = None
        
        if not classifier.is_trained:
            raise ValueError("Classifier must be trained before explanations")
        
        self._setup_explainers()
    
    def _setup_explainers(self):
        """Setup SHAP and LIME explainers if available."""
        # For tree-based models, use TreeExplainer
        if hasattr(self.classifier.model, 'estimators_'):
            self.explainer_shap = shap.TreeExplainer(self.classifier.model)
        else:
            # For other models, use KernelExplainer (slower but works for any model)
            background = np.zeros((1, len(self.feature_names)))  # Use zero baseline
            self.explainer_shap = shap.KernelExplainer(
                self.classifier.predict_proba, background
            )
        
        # Create dummy training data for LIME
        dummy_data = np.random.random((100, len(self.feature_names)))
        self.explainer_lime = LimeTabularExplainer(
            dummy_data,
            feature_names=self.feature_names,
            class_names=self.classifier.label_encoder.classes_,
            mode='classification'
        )
    
    def explain_prediction(self, features: np.ndarray, 
                          method: str = 'feature_importance') -> Dict[str, Any]:
        """
        Explain a single prediction.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get prediction
        prediction = self.classifier.predict(features)[0]
        
        explanation = {
            'prediction': prediction.smell_type,
            'confidence': prediction.confidence,
            'method': method
        }
        
        if method == 'feature_importance':
            explanation.update(self._explain_with_feature_importance(features))
        elif method == 'shap':
            explanation.update(self._explain_with_shap(features))
        elif method == 'lime':
            explanation.update(self._explain_with_lime(features))
        else:
            # Fallback to feature importance
            explanation.update(self._explain_with_feature_importance(features))
            explanation['method'] = 'feature_importance (fallback)'
        
        return explanation
    
    def _explain_with_feature_importance(self, features: np.ndarray) -> Dict[str, Any]:
        """Explain using global feature importance."""
        importance = self.classifier.get_feature_importance()
        
        if not importance:
            return {'explanation': 'Feature importance not available'}
        
        # Get non zero features
        non_zero_features = {}
        for i, (name, value) in enumerate(zip(self.feature_names, features[0])):
            if value != 0 and name in importance:
                non_zero_features[name] = {
                    'value': float(value),
                    'importance': importance[name],
                    'contribution': float(value * importance[name])
                }
        
        # Sort by contribution
        sorted_features = dict(sorted(
            non_zero_features.items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        ))
        
        return {
            'explanation': 'Global feature importance weighted by feature values',
            'top_features': dict(list(sorted_features.items())[:10]),
            'feature_importance': importance
        }
    
    def _explain_with_shap(self, features: np.ndarray) -> Dict[str, Any]:
        """Explain using SHAP values."""
        if not self.explainer_shap:
            return {'explanation': 'SHAP not available'}
        
        try:
            # Get SHAP values
            if hasattr(self.explainer_shap, 'shap_values'):
                shap_values = self.explainer_shap.shap_values(features)
            else:
                shap_values = self.explainer_shap(features)
            
            # Handle multi class case
            if isinstance(shap_values, list):
                # Use values for predicted class
                predicted_class = self.classifier.model.predict(
                    self.classifier.scaler.transform(features)
                )[0]
                shap_vals = shap_values[predicted_class][0]
            else:
                shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
            
            # Create feature importance from SHAP values
            shap_importance = {}
            for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_vals)):
                if abs(shap_val) > 0.001:  # Only include significant features
                    shap_importance[name] = {
                        'shap_value': float(shap_val),
                        'feature_value': float(features[0, i]),
                        'abs_importance': abs(float(shap_val))
                    }
            
            # Sort by absolute SHAP value
            sorted_shap = dict(sorted(
                shap_importance.items(),
                key=lambda x: x[1]['abs_importance'],
                reverse=True
            ))
            
            return {
                'explanation': 'SHAP (SHapley Additive exPlanations) values',
                'top_features': dict(list(sorted_shap.items())[:10]),
                'shap_values': shap_vals.tolist()
            }
        
        except Exception as e:
            return {
                'explanation': f'SHAP explanation failed: {str(e)}',
                'fallback': self._explain_with_feature_importance(features)
            }
    
    def _explain_with_lime(self, features: np.ndarray) -> Dict[str, Any]:
        """Explain using LIME."""
        if not self.explainer_lime:
            return {'explanation': 'LIME not available'}
        
        try:
            # Get LIME explanation
            exp = self.explainer_lime.explain_instance(
                features[0],
                self.classifier.predict_proba,
                num_features=10
            )
            
            # Extract feature importances
            lime_importance = {}
            for feature_idx, importance in exp.as_list():
                if feature_idx < len(self.feature_names):
                    name = self.feature_names[feature_idx]
                    lime_importance[name] = {
                        'lime_importance': float(importance),
                        'feature_value': float(features[0, feature_idx]),
                        'abs_importance': abs(float(importance))
                    }
            
            # Sort by absolute importance
            sorted_lime = dict(sorted(
                lime_importance.items(),
                key=lambda x: x[1]['abs_importance'],
                reverse=True
            ))
            
            return {
                'explanation': 'LIME (Local Interpretable Model-agnostic Explanations)',
                'top_features': sorted_lime,
                'lime_score': exp.score
            }
        
        except Exception as e:
            return {
                'explanation': f'LIME explanation failed: {str(e)}',
                'fallback': self._explain_with_feature_importance(features)
            }
    
    def generate_explanation_report(self, features: np.ndarray, 
                                  output_file: str = 'explanation_report.txt') -> str:
        """
        Generate a human-readable explanation report.
        """
        explanation = self.explain_prediction(features, method='shap')
        
        report = []
        report.append("=" * 80)
        report.append("SECURITY SMELL PREDICTION EXPLANATION")
        report.append("=" * 80)
        
        # Basic prediction info
        report.append(f"\nPrediction: {explanation['prediction']}")
        report.append(f"Confidence: {explanation['confidence']:.3f}")
        report.append(f"Explanation Method: {explanation['method']}")
        
        # Top contributing features
        report.append(f"\nTop Contributing Features:")
        report.append("-" * 40)
        
        top_features = explanation.get('top_features', {})
        for i, (feature_name, feature_info) in enumerate(list(top_features.items())[:10], 1):
            if isinstance(feature_info, dict):
                value = feature_info.get('feature_value', feature_info.get('value', 'N/A'))
                if 'shap_value' in feature_info:
                    importance = feature_info['shap_value']
                    impact = "positive" if importance > 0 else "negative"
                elif 'contribution' in feature_info:
                    importance = feature_info['contribution']
                    impact = "positive" if importance > 0 else "negative"
                else:
                    importance = feature_info.get('importance', 0)
                    impact = "positive"
                
                report.append(f"{i:2d}. {feature_name}")
                report.append(f"    Value: {value:.3f}, Impact: {importance:.3f} ({impact})")
            else:
                report.append(f"{i:2d}. {feature_name}: {feature_info}")
        
        # Additional context
        if explanation['prediction'] != 'NO_SMELL':
            report.append(f"\nSecurity Concern: {explanation['prediction']}")
            report.append("This prediction indicates a potential security vulnerability.")
            report.append("Review the highlighted features and consider the suggested fixes.")
        else:
            report.append("\nNo Security Smells Detected")
            report.append("The task appears to follow security best practices.")
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def visualize_feature_importance(self, output_file: str = 'feature_importance.png'):
        """Create visualization of global feature importance."""
        importance = self.classifier.get_feature_importance()
        
        if not importance:
            print("Feature importance not available")
            return
        
        # Get top 15 features
        top_features = dict(list(importance.items())[:15])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        names = list(top_features.keys())
        values = list(top_features.values())
        
        # Horizontal bar chart
        bars = ax.barh(names, values, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Customize
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Top 15 Most Important Features for Security Smell Detection', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance visualization saved to: {output_file}")


def main():
    """Demonstration of explainability features."""
    print("Explainability Module Demonstration")
    print("=" * 50)
    print("Note: This is a demonstration. Load your trained model for real explanations.")
    
    print(" SHAP available for detailed explanations")
    print(" LIME available for local explanations")


if __name__ == '__main__':
    main()