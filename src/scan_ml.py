"""
Enhanced scanner with ML based detection capabilities
Combines rule based and machine learning approaches for robust security smell detection
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsers import AnsibleParser
from detectors import SecurityDetector, SecuritySmell
from report import Reporter
from feature_extractor import FeatureExtractor
from ml_models import SecuritySmellClassifier, HybridDetector


class MLEnhancedScanner:
    """Scanner with ML based detection capabilities"""
    
    def __init__(self, use_ml: bool = True, ml_model_path: str = None):
        self.parser = AnsibleParser()
        self.rule_detector = SecurityDetector()
        self.feature_extractor = FeatureExtractor()
        self.use_ml = use_ml
        self.ml_classifier = None
        self.hybrid_detector = None
        
        if use_ml:
            self._load_ml_model(ml_model_path)
    
    def _load_ml_model(self, model_path: str = None):
        """Load the ML model"""
        if model_path is None:
            # Default model path
            model_path = Path('models') / 'security_smell_classifier.pkl'
        
        if not Path(model_path).exists():
            print(f"  ML model not found at {model_path}")
            print("   Run 'python src/train_model.py' to train a model")
            print("   Falling back to rule based detection only")
            self.use_ml = False
            return
        
        try:
            self.ml_classifier = SecuritySmellClassifier()
            self.ml_classifier.load_model(str(model_path))
            self.hybrid_detector = HybridDetector(
                self.ml_classifier,
                confidence_threshold=0.7
            )
            print(f" ML model loaded from {model_path}")
        except Exception as e:
            print(f"  Error loading ML model: {e}")
            print("   Falling back to rule based detection only")
            self.use_ml = False
    
    def scan_directory(self, directory: str) -> List[SecuritySmell]:
        """Scan all YAML files in a directory"""
        print(f"\n Scanning directory: {directory}")
        if self.use_ml:
            print("   Mode: Hybrid (ML + Rules)")
        else:
            print("   Mode: Rule based only")
        
        yaml_files = self.parser.find_yaml_files(directory)
        
        if not yaml_files:
            print(f"  No YAML files found in {directory}")
            return []
        
        print(f" Found {len(yaml_files)} YAML file(s)")
        
        all_smells = []
        files_with_issues = 0
        ml_detections = 0
        rule_detections = 0
        
        for file_path in yaml_files:
            file_smells, ml_count, rule_count = self.scan_file(file_path)
            if file_smells:
                files_with_issues += 1
                all_smells.extend(file_smells)
                ml_detections += ml_count
                rule_detections += rule_count
        
        print(f"\n Scan complete: {len(yaml_files)} files scanned, {files_with_issues} with issues")
        if self.use_ml:
            print(f"   ML detections: {ml_detections}, Rule detections: {rule_detections}")
        
        return all_smells
    
    def scan_file(self, file_path: str) -> Tuple[List[SecuritySmell], int, int]:
        """
        Scan a single file.
        """
        print(f"   Scanning: {file_path}")
        
        playbook = self.parser.load_yaml(file_path)
        if not playbook:
            return [], 0, 0
        
        file_smells = []
        ml_count = 0
        rule_count = 0
        
        for task, path, task_path in self.parser.traverse_tasks(playbook, file_path):
            if not isinstance(task, dict):
                continue
            
            module, params = self.parser.extract_module_params(task)
            if not module:
                continue
            
            # Rule based detection
            rule_smells = self.rule_detector.detect_all(task, path, task_path, module, params)
            
            if self.use_ml:
                # Extract features for ML
                features = self.feature_extractor.extract_features(task, module, params)
                
                # Get rule based result for hybrid detection
                rule_result = rule_smells[0].smell_type if rule_smells else None
                
                # Hybrid detection
                smell_type, confidence, method = self.hybrid_detector.detect(
                    features, rule_result
                )
                
                if smell_type != 'NO_SMELL':
                    # Create SecuritySmell object
                    smell = SecuritySmell(
                        smell_type=smell_type,
                        file_path=path,
                        task_path=".".join(task_path),
                        task_name=task.get('name', 'unnamed'),
                        module=module,
                        description=f"{smell_type} detected with {confidence:.2f} confidence",
                        fix_suggestion=self._get_fix_suggestion(smell_type),
                        severity=self._get_severity(smell_type),
                        line_content=f"Detection method: {method}"
                    )
                    
                    # Check if already detected by rules
                    if not any(s.smell_type == smell_type for s in rule_smells):
                        file_smells.append(smell)
                        if 'ml' in method:
                            ml_count += 1
                        else:
                            rule_count += 1
                
                # Add rule detected smells
                for smell in rule_smells:
                    if smell.smell_type not in [s.smell_type for s in file_smells]:
                        file_smells.append(smell)
                        rule_count += 1
            else:
                # Rule based only
                file_smells.extend(rule_smells)
                rule_count += len(rule_smells)
        
        if file_smells:
            print(f"       Found {len(file_smells)} issue(s)")
            if self.use_ml:
                print(f"        ML: {ml_count}, Rules: {rule_count}")
        
        return file_smells, ml_count, rule_count
    
    def _get_fix_suggestion(self, smell_type: str) -> str:
        """Get fix suggestion for a smell type"""
        suggestions = {
            'HTTP_WITHOUT_TLS': 'Use HTTPS instead of HTTP for secure communication',
            'OPEN_BINDING': 'Bind to specific interface (e.g., 127.0.0.1 for localhost only)',
            'HARDCODED_SECRET': 'Use Ansible Vault or environment variables to store secrets',
            'WEAK_PASSWORD': 'Use a strong password with mix of characters, stored in Ansible Vault',
            'WEAK_CRYPTOGRAPHY': 'Use strong algorithms like SHA256, SHA3, or AES',
        }
        return suggestions.get(smell_type, 'Review and fix the security issue')
    
    def _get_severity(self, smell_type: str) -> str:
        """Get severity for a smell type"""
        severities = {
            'HTTP_WITHOUT_TLS': 'HIGH',
            'OPEN_BINDING': 'MEDIUM',
            'HARDCODED_SECRET': 'HIGH',
            'WEAK_PASSWORD': 'HIGH',
            'WEAK_CRYPTOGRAPHY': 'HIGH',
        }
        return severities.get(smell_type, 'MEDIUM')


def main():
    parser = argparse.ArgumentParser(
        description='ML enhanced Ansible Security Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python src/scan_ml.py samples/                    # Hybrid ML + rules scan
  python src/scan_ml.py samples/ --no-ml           # Rule based only
  python src/scan_ml.py samples/ --train           # Train new ML model
  python src/scan_ml.py samples/ --json --chart    # Export results
        '''
    )
    
    parser.add_argument(
        'path',
        help='Path to Ansible playbook file or directory'
    )
    
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Disable ML detection, use rules only'
    )
    
    parser.add_argument(
        '--model',
        help='Path to ML model file (default: models/security_smell_classifier.pkl)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new ML model before scanning'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.7,
        help='Confidence threshold for ML detection (0.0-1.0)'
    )
    
    parser.add_argument(
        '--json',
        nargs='?',
        const='security_smells.json',
        help='Export results to JSON file'
    )
    
    parser.add_argument(
        '--chart',
        action='store_true',
        help='Generate a bar chart of security smells'
    )
    
    parser.add_argument(
        '--chart-out',
        default='security_smells_chart.png',
        help='Output filename for chart'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress CLI output'
    )
    
    args = parser.parse_args()
    
    # Train model if requested
    if args.train:
        print(" Training new ML model...")
        from train_model import train_models
        train_models()
        print()
    
    # Validate input path
    if not os.path.exists(args.path):
        print(f" Error: Path '{args.path}' does not exist")
        sys.exit(1)
    
    # Initialize scanner
    use_ml = not args.no_ml
    scanner = MLEnhancedScanner(use_ml=use_ml, ml_model_path=args.model)
    
    # Update confidence threshold if ML is used
    if scanner.hybrid_detector and args.confidence:
        scanner.hybrid_detector.confidence_threshold = args.confidence
    
    # Start timing
    start_time = time.time()
    
    # Perform scan
    if os.path.isfile(args.path):
        smells, _, _ = scanner.scan_file(args.path)
    else:
        smells = scanner.scan_directory(args.path)
    
    # Calculate scan time
    scan_time = time.time() - start_time
    
    # Create reporter
    reporter = Reporter(smells)
    
    # Output results
    if not args.quiet:
        reporter.print_cli_report()
        print(f"\n⏱  Scan completed in {scan_time:.2f} seconds")
        if use_ml and scanner.use_ml:
            print("    ML enhanced detection was used")
    
    # Export JSON if requested
    if args.json:
        reporter.export_json(args.json)
    
    # Generate chart if requested
    if args.chart:
        reporter.generate_chart(args.chart_out)
    
    # Exit with appropriate code
    if smells:
        sys.exit(1)
    else:
        if not args.quiet:
            print("\n All clear! No security smells detected")
        sys.exit(0)


if __name__ == '__main__':
    main()