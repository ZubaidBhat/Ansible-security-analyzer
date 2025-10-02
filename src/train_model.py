"""
Training pipeline for ML-based security smell detection.
Generates synthetic training data and trains models.
"""
import numpy as np
import json
from pathlib import Path
import random
from typing import List, Tuple, Dict
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import FeatureExtractor
from ml_models import SecuritySmellClassifier
from parsers import AnsibleParser
from detectors import SecurityDetector
import xgboost

class TrainingDataGenerator:
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.parser = AnsibleParser()
        self.detector = SecurityDetector()
        
        # Templates for generating synthetic vulnerable tasks
        self.vulnerable_templates = {
            'HTTP_WITHOUT_TLS': [
                {'get_url': {'url': 'http://example.com/file.tar.gz', 'dest': '/tmp/file.tar.gz'}},
                {'uri': {'url': 'http://api.example.com/data', 'method': 'GET'}},
                {'yum': {'name': 'package', 'baseurl': 'http://repo.example.com'}},
                {'apt_repository': {'repo': 'http://ppa.example.com/ubuntu'}},
            ],
            'OPEN_BINDING': [
                {'lineinfile': {'path': '/etc/config', 'line': 'bind 0.0.0.0'}},
                {'template': {'src': 'config.j2', 'dest': '/etc/app.conf'}, '_content': 'listen: 0.0.0.0:8080'},
                {'docker_container': {'name': 'app', 'ports': ['0.0.0.0:80:80']}},
                {'firewalld': {'service': 'http', 'source': '0.0.0.0/0'}},
            ],
            'HARDCODED_SECRET': [
                {'user': {'name': 'admin', 'password': 'SecretPassword123!'}},
                {'postgresql_user': {'name': 'dbuser', 'password': 'MyDatabaseP@ssw0rd'}},
                {'copy': {'content': 'api_key=sk_live_4eC39HqLyjWDarjtT1zdp7dc', 'dest': '/etc/config'}},
                {'environment': {'API_TOKEN': 'ghp_1234567890abcdef1234567890abcdef'}},
            ],
            'WEAK_PASSWORD': [
                {'user': {'name': 'admin', 'password': 'admin'}},
                {'mysql_user': {'name': 'root', 'password': '12345'}},
                {'postgresql_user': {'name': 'postgres', 'password': 'password'}},
                {'authorized_key': {'user': 'root', 'key': 'ssh-rsa AAAAB3NzaC1yc2E... password'}},
            ],
            'WEAK_CRYPTOGRAPHY': [
                {'copy': {'content': 'hash_algorithm=md5', 'dest': '/etc/security.conf'}},
                {'openssl_certificate': {'path': '/etc/ssl/cert.pem', 'signature_algorithms': ['sha1']}},
                {'crypto': {'cipher': 'des', 'action': 'encrypt'}},
                {'hash': {'data': 'sensitive', 'algorithm': 'md5'}},
            ]
        }
        
        # Templates for clean tasks
        self.clean_templates = [
            {'package': {'name': 'nginx', 'state': 'present'}},
            {'service': {'name': 'nginx', 'state': 'started'}},
            {'file': {'path': '/var/log/app', 'state': 'directory', 'mode': '0755'}},
            {'copy': {'src': 'files/config.txt', 'dest': '/etc/app/config.txt'}},
            {'template': {'src': 'templates/index.j2', 'dest': '/var/www/index.html'}},
            {'user': {'name': 'appuser', 'state': 'present', 'shell': '/bin/bash'}},
            {'group': {'name': 'appgroup', 'state': 'present'}},
            {'cron': {'name': 'backup', 'hour': '2', 'job': '/usr/bin/backup.sh'}},
            {'systemd': {'name': 'app', 'enabled': 'yes', 'daemon_reload': 'yes'}},
            {'apt': {'update_cache': 'yes'}},
            {'pip': {'name': 'requests', 'version': '2.28.0'}},
            {'git': {'repo': 'https://github.com/ansible/ansible.git', 'dest': '/opt/ansible'}},
        ]
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate synthetic training data
        """
        X = []
        y = []
        task_samples = []
        
        samples_per_class = n_samples // 6  # 6 classes including NO_SMELL
        
        # Generate vulnerable samples
        for smell_type, templates in self.vulnerable_templates.items():
            for _ in range(samples_per_class):
                # Randomly select a template
                template = random.choice(templates)
                task = self._mutate_task(template.copy())
                
                # Extract module and params
                module = list(task.keys())[0] if task else 'unknown'
                params = task.get(module, {})
                
                # Add random task metadata
                task_dict = {
                    'name': f"Task for {module}",
                    module: params
                }
                
                # Extract features
                features = self.feature_extractor.extract_features(task_dict, module, params)
                X.append(features)
                y.append(smell_type)
                task_samples.append(task_dict)
        
        # Generate clean samples
        for _ in range(samples_per_class):
            template = random.choice(self.clean_templates)
            task = self._mutate_task(template.copy(), clean=True)
            
            module = list(task.keys())[0] if task else 'unknown'
            params = task.get(module, {})
            
            task_dict = {
                'name': f"Clean task for {module}",
                module: params
            }
            
            features = self.feature_extractor.extract_features(task_dict, module, params)
            X.append(features)
            y.append('NO_SMELL')
            task_samples.append(task_dict)
        
        return np.array(X), np.array(y), task_samples
    
    def _mutate_task(self, task: Dict, clean: bool = False) -> Dict:
        """Add variations to task to increase diversity."""
        # Add random parameters
        if random.random() > 0.5:
            module = list(task.keys())[0]
            params = task[module]
            
            if isinstance(params, dict):
                # Add common Ansible parameters
                if random.random() > 0.5:
                    params['become'] = random.choice([True, False])
                if random.random() > 0.5:
                    params['when'] = f"ansible_os_family == '{random.choice(['Debian', 'RedHat'])}'"
                if random.random() > 0.5:
                    params['tags'] = [random.choice(['security', 'config', 'install'])]
                
                # Add noise to values
                if not clean and random.random() > 0.7:
                    # Add some noise that doesn't change the vulnerability
                    for key in list(params.keys()):
                        if isinstance(params[key], str) and random.random() > 0.8:
                            params[key] = params[key] + f"_{random.randint(1, 100)}"
        
        return task
    
    def load_real_data(self, directory: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and label real Ansible playbooks
        """
        X = []
        y = []
        task_samples = []
        
        yaml_files = self.parser.find_yaml_files(directory)
        
        for file_path in yaml_files:
            playbook = self.parser.load_yaml(file_path)
            
            for task, _, task_path in self.parser.traverse_tasks(playbook, file_path):
                if not isinstance(task, dict):
                    continue
                
                module, params = self.parser.extract_module_params(task)
                
                if module:
                    # Extract features
                    features = self.feature_extractor.extract_features(task, module, params)
                    
                    # Get label from rule based detector
                    smells = self.detector.detect_all(task, file_path, task_path, module, params)
                    
                    if smells:
                        # Use the first detected smell as label
                        label = smells[0].smell_type
                    else:
                        label = 'NO_SMELL'
                    
                    X.append(features)
                    y.append(label)
                    task_samples.append(f"{file_path}: {task.get('name', 'unnamed')}")
        
        return np.array(X), np.array(y), task_samples


def train_models():
    """Main training pipeline."""
    print("=" * 80)
    print("ML-Based Security Smell Detection Training Pipeline")
    print("=" * 80)
    
    # Initialize components
    generator = TrainingDataGenerator()
    feature_extractor = FeatureExtractor()
    
    # Generate synthetic training data
    print("\n1. Generating synthetic training data...")
    X_synthetic, y_synthetic, _ = generator.generate_synthetic_data(n_samples=1200)
    print(f"   Generated {len(X_synthetic)} synthetic samples")
    
    # Load real data if available
    X_real, y_real, _ = [], [], []
    if Path('samples').exists():
        print("\n2. Loading real playbook data...")
        X_real, y_real, _ = generator.load_real_data('samples')
        print(f"   Loaded {len(X_real)} real samples")
    
    # Combine datasets
    if len(X_real) > 0:
        X = np.vstack([X_synthetic, X_real])
        y = np.hstack([y_synthetic, y_real])
    else:
        X = X_synthetic
        y = y_synthetic
    
    print(f"\n3. Total training samples: {len(X)}")
    print(f"   Feature dimensions: {X.shape[1]}")
    
    # Get feature names
    feature_names = feature_extractor.get_feature_names()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n4. Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train different model types
    model_types = ['random_forest', 'ensemble', 'xgboost']
    
    best_model = None
    best_score = 0
    
    for model_type in model_types:
        print(f"\n5. Training {model_type} model...")
        
        classifier = SecuritySmellClassifier(model_type=model_type)
        classifier.train(X_train, y_train, feature_names)
        
        # Evaluate
        print(f"\n   Evaluating {model_type}...")
        metrics = classifier.evaluate(X_test, y_test)
        f1_score = metrics['f1_weighted']
        
        print(f"   F1 Score (weighted): {f1_score:.3f}")
        
        if f1_score > best_score:
            best_score = f1_score
            best_model = classifier
            best_model_type = model_type
    
    print(f"\n6. Best model: {best_model_type} with F1 score: {best_score:.3f}")
    
    # Save the best model
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    
    model_file = model_path / 'security_smell_classifier.pkl'
    best_model.save_model(str(model_file))
    
    # Get and display feature importance
    print("\n7. Top 10 Most Important Features:")
    importance = best_model.get_feature_importance()
    if importance:
        for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
            print(f"   {i}. {feature}: {score:.4f}")
    
    # Save training metadata
    metadata = {
        'model_type': best_model_type,
        'f1_score': float(best_score),
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features': int(X.shape[1]),
        'feature_names': feature_names
    }
    
    with open(model_path / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n8. Training complete!")
    print(f"   Model saved to: {model_file}")
    print(f"   Metadata saved to: {model_path / 'training_metadata.json'}")
    
    return best_model


if __name__ == '__main__':
    
    # Run training
    train_models()