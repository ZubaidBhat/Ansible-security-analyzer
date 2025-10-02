"""
Feature extraction module for machine learning-based security smell detection.
Extracts structural, textual, and security-specific features from Ansible playbooks.
"""
import re
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import hashlib
import math
import string
from pathlib import Path


class FeatureExtractor:
    """Extract features from Ansible tasks for ML models."""
    
    def __init__(self):
        # Security related keywords
        self.security_keywords = {
            'credential': ['password', 'passwd', 'pwd', 'secret', 'token', 'key', 'api_key', 
                          'access_key', 'private_key', 'auth', 'credential'],
            'network': ['bind', 'listen', 'host', 'address', 'port', 'interface', 'ip'],
            'crypto': ['encrypt', 'decrypt', 'hash', 'algorithm', 'cipher', 'ssl', 'tls', 
                       'certificate', 'md5', 'sha1', 'sha256', 'aes', 'des', 'rsa'],
            'permission': ['chmod', 'chown', 'mode', 'owner', 'group', 'become', 'sudo'],
            'sensitive_ops': ['copy', 'template', 'lineinfile', 'blockinfile', 'shell', 'command']
        }
        
        # Common weak patterns
        self.weak_patterns = {
            'weak_passwords': ['password', 'admin', '12345', 'default', 'test', 'demo'],
            'weak_crypto': ['md5', 'sha1', 'des', '3des', 'rc4'],
            'insecure_protocols': ['http://', 'ftp://', 'telnet://'],
            'open_bindings': ['0.0.0.0', '0:0:0:0:0:0:0:0', '*']
        }
        
        # Module risk scores based on security implications
        self.module_risk_scores = {
            'shell': 0.9, 'command': 0.9, 'raw': 0.9,
            'user': 0.7, 'authorized_key': 0.7,
            'copy': 0.5, 'template': 0.5, 'lineinfile': 0.5,
            'service': 0.4, 'systemd': 0.4,
            'package': 0.3, 'yum': 0.3, 'apt': 0.3,
            'file': 0.2, 'stat': 0.1
        }
    
    def extract_features(self, task: Dict, module: str, params: Dict) -> np.ndarray:
        """
        Extract feature vector from a task.
        """
        features = []
        
        # 1. Structural features
        features.extend(self._extract_structural_features(task, module, params))
        
        # 2. Module-specific features
        features.extend(self._extract_module_features(module, params))
        
        # 3. Security keyword features
        features.extend(self._extract_keyword_features(task, params))
        
        # 4. Pattern-based features
        features.extend(self._extract_pattern_features(params))
        
        # 5. Statistical features
        features.extend(self._extract_statistical_features(params))
        
        # 6. Entropy features
        features.extend(self._extract_entropy_features(params))
        
        # 7. Network-related features
        features.extend(self._extract_network_features(params))
        
        # 8. Cryptographic features
        features.extend(self._extract_crypto_features(params))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_structural_features(self, task: Dict, module: str, params: Dict) -> List[float]:
        """Extract structural features from task."""
        features = []
        
        # Task complexity by number of parameters
        features.append(len(params) if isinstance(params, dict) else 0)
        
        # Has conditional execution
        features.append(1.0 if 'when' in task else 0.0)
        
        # Has become
        features.append(1.0 if task.get('become', False) else 0.0)
        
        # Has loop
        features.append(1.0 if any(k in task for k in ['loop', 'with_items', 'with_dict']) else 0.0)
        
        # Task name length
        task_name = task.get('name', '')
        features.append(len(task_name))
        
        # Module risk score
        features.append(self.module_risk_scores.get(module, 0.1))
        
        # Number of nested levels
        features.append(self._calculate_nesting_depth(params))
        
        # Has error handling
        features.append(1.0 if any(k in task for k in ['ignore_errors', 'failed_when']) else 0.0)
        
        return features
    
    def _extract_module_features(self, module: str, params: Dict) -> List[float]:
        """Extract module-specific features."""
        features = []
        
        # Encoding for common security relevant modules (top 10)
        security_modules = ['user', 'copy', 'template', 'lineinfile', 'shell', 
                          'command', 'file', 'service', 'authorized_key', 'git']
        for mod in security_modules:
            features.append(1.0 if module == mod else 0.0)
        
        # Is it a network related module
        network_modules = ['uri', 'get_url', 'npm', 'pip', 'docker_container']
        features.append(1.0 if module in network_modules else 0.0)
        
        # Is it a database module
        db_modules = ['postgresql_user', 'mysql_user', 'mongodb_user', 'redis']
        features.append(1.0 if module in db_modules else 0.0)
        
        return features
    
    def _extract_keyword_features(self, task: Dict, params: Dict) -> List[float]:
        """Extract security keyword frequency features."""
        features = []
        
        # Convert params to string for analysis
        param_str = str(params).lower() if params else ""
        task_str = str(task).lower()
        
        # Count security keywords by category
        for category, keywords in self.security_keywords.items():
            count = sum(1 for keyword in keywords if keyword in param_str)
            features.append(count)
        
        # Presence of specific sensitive keywords
        features.append(1.0 if 'password' in param_str else 0.0)
        features.append(1.0 if 'token' in param_str else 0.0)
        features.append(1.0 if 'secret' in param_str else 0.0)
        features.append(1.0 if 'key' in param_str else 0.0)
        
        return features
    
    def _extract_pattern_features(self, params: Dict) -> List[float]:
        """Extract pattern-based security features."""
        features = []
        param_str = str(params) if params else ""
        
        # Check for weak password patterns
        weak_pwd_count = sum(1 for pwd in self.weak_patterns['weak_passwords'] 
                           if pwd in param_str.lower())
        features.append(weak_pwd_count)
        
        # Check for weak crypto
        weak_crypto_count = sum(1 for crypto in self.weak_patterns['weak_crypto'] 
                              if crypto in param_str.lower())
        features.append(weak_crypto_count)
        
        # Check for insecure protocols
        insecure_proto_count = sum(1 for proto in self.weak_patterns['insecure_protocols'] 
                                  if proto in param_str.lower())
        features.append(insecure_proto_count)
        
        # Check for open bindings
        open_binding_count = sum(1 for binding in self.weak_patterns['open_bindings'] 
                               if binding in param_str)
        features.append(open_binding_count)
        
        # URL count and types
        http_urls = len(re.findall(r'http://', param_str, re.IGNORECASE))
        https_urls = len(re.findall(r'https://', param_str, re.IGNORECASE))
        features.append(http_urls)
        features.append(https_urls)
        features.append(http_urls / (http_urls + https_urls + 0.001))  # HTTP ratio
        
        # IP address patterns
        ipv4_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ip_addresses = re.findall(ipv4_pattern, param_str)
        features.append(len(ip_addresses))
        features.append(1.0 if '0.0.0.0' in ip_addresses else 0.0)
        features.append(1.0 if '127.0.0.1' in ip_addresses else 0.0)
        
        # Base64 encoded content for potential secrets
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        base64_matches = len(re.findall(base64_pattern, param_str))
        features.append(base64_matches)
        
        # Hex strings for potential keys/tokens
        hex_pattern = r'[0-9a-fA-F]{32,}'
        hex_matches = len(re.findall(hex_pattern, param_str))
        features.append(hex_matches)
        
        return features
    
    def _extract_statistical_features(self, params: Dict) -> List[float]:
        """Extract statistical features from parameter values."""
        features = []
        
        if not isinstance(params, dict):
            return [0] * 10  # Return zeros if not a dict
        
        # String value statistics
        string_values = [str(v) for v in params.values() if isinstance(v, (str, int, float))]
        
        if string_values:
            # Average length
            avg_len = np.mean([len(s) for s in string_values])
            features.append(avg_len)
            
            # Max length
            max_len = max([len(s) for s in string_values])
            features.append(max_len)
            
            # Min length
            min_len = min([len(s) for s in string_values])
            features.append(min_len)
            
            # Length variance
            len_var = np.var([len(s) for s in string_values])
            features.append(len_var)
        else:
            features.extend([0, 0, 0, 0])
        
        # Character diversity in values
        all_chars = ''.join(string_values)
        if all_chars:
            # Ratio of alphabetic characters
            alpha_ratio = sum(c.isalpha() for c in all_chars) / len(all_chars)
            features.append(alpha_ratio)
            
            # Ratio of numeric characters
            numeric_ratio = sum(c.isdigit() for c in all_chars) / len(all_chars)
            features.append(numeric_ratio)
            
            # Ratio of special characters
            special_ratio = sum(not c.isalnum() and not c.isspace() for c in all_chars) / len(all_chars)
            features.append(special_ratio)
            
            # Number of unique characters
            unique_chars = len(set(all_chars))
            features.append(unique_chars)
            
            # Character frequency entropy
            char_freq = Counter(all_chars)
            total_chars = len(all_chars)
            entropy = -sum((count/total_chars) * math.log2(count/total_chars) 
                          for count in char_freq.values())
            features.append(entropy)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Boolean flags
        features.append(1.0 if any('=' in str(v) for v in params.values()) else 0.0)
        
        return features
    
    def _extract_entropy_features(self, params: Dict) -> List[float]:
        """Calculate entropy-based features for randomness detection."""
        features = []
        
        def calculate_shannon_entropy(s: str) -> float:
            """Calculate Shannon entropy of a string."""
            if not s:
                return 0
            prob = [float(s.count(c)) / len(s) for c in set(s)]
            entropy = -sum(p * math.log2(p) for p in prob if p > 0)
            return entropy
        
        if isinstance(params, dict):
            # Entropy of each parameter value
            entropies = []
            for key, value in params.items():
                if isinstance(value, str) and len(value) > 5:
                    entropies.append(calculate_shannon_entropy(value))
            
            if entropies:
                features.append(np.mean(entropies))  # Average entropy
                features.append(np.max(entropies))   # Max entropy
                features.append(np.min(entropies))   # Min entropy
                # High entropy might indicate random tokens/passwords
                features.append(sum(1 for e in entropies if e > 4.0))
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def _extract_network_features(self, params: Dict) -> List[float]:
        """Extract network related features."""
        features = []
        param_str = str(params).lower() if params else ""
        
        # Port numbers
        port_pattern = r':(\d{1,5})\b'
        ports = re.findall(port_pattern, param_str)
        
        if ports:
            port_numbers = [int(p) for p in ports]
            # Common insecure ports
            insecure_ports = [21, 23, 80, 139, 445, 3389]  # FTP, Telnet, HTTP, SMB, RDP
            features.append(sum(1 for p in port_numbers if p in insecure_ports))
            # Well known ports
            features.append(sum(1 for p in port_numbers if p < 1024))
            # High ports
            features.append(sum(1 for p in port_numbers if p > 1024))
        else:
            features.extend([0, 0, 0])
        
        # Protocol indicators
        features.append(1.0 if 'ssl' in param_str else 0.0)
        features.append(1.0 if 'tls' in param_str else 0.0)
        features.append(1.0 if 'https' in param_str else 0.0)
        features.append(1.0 if 'http' in param_str and 'https' not in param_str else 0.0)
        
        # Network binding patterns
        features.append(1.0 if 'localhost' in param_str else 0.0)
        features.append(1.0 if '127.0.0.1' in param_str else 0.0)
        features.append(1.0 if '0.0.0.0' in param_str else 0.0)
        
        return features
    
    def _extract_crypto_features(self, params: Dict) -> List[float]:
        """Extract cryptography-related features."""
        features = []
        param_str = str(params).lower() if params else ""
        
        # Weak algorithms
        weak_algos = ['md5', 'sha1', 'des', '3des', 'rc4', 'rc2']
        for algo in weak_algos:
            features.append(1.0 if algo in param_str else 0.0)
        
        # Strong algorithms
        strong_algos = ['sha256', 'sha512', 'sha3', 'aes', 'rsa2048', 'ecdsa']
        for algo in strong_algos:
            features.append(1.0 if algo in param_str else 0.0)
        
        # Key length indicators
        key_lengths = re.findall(r'\b(128|192|256|512|1024|2048|4096)\b', param_str)
        if key_lengths:
            avg_key_length = np.mean([int(k) for k in key_lengths])
            features.append(avg_key_length)
            features.append(1.0 if any(int(k) < 256 for k in key_lengths) else 0.0)  # Weak key
        else:
            features.extend([0, 0])
        
        # Certificate/key indicators
        features.append(1.0 if 'certificate' in param_str else 0.0)
        features.append(1.0 if 'private' in param_str and 'key' in param_str else 0.0)
        features.append(1.0 if 'public' in param_str and 'key' in param_str else 0.0)
        
        return features
    
    def _calculate_nesting_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate maximum nesting depth of data structure."""
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(self._calculate_nesting_depth(v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return depth
            return max(self._calculate_nesting_depth(item, depth + 1) for item in obj)
        else:
            return depth
    
    def get_feature_names(self) -> List[str]:
        """Return names of all features for interpretability."""
        names = []
        
        # Structural features
        names.extend(['param_count', 'has_when', 'has_become', 'has_loop', 
                     'task_name_length', 'module_risk_score', 'nesting_depth', 'has_error_handling'])
        
        # Module features
        names.extend(['is_user', 'is_copy', 'is_template', 'is_lineinfile', 'is_shell',
                     'is_command', 'is_file', 'is_service', 'is_authorized_key', 'is_git',
                     'is_network_module', 'is_database_module'])
        
        # Keyword features
        names.extend(['credential_keywords', 'network_keywords', 'crypto_keywords',
                     'permission_keywords', 'sensitive_ops_keywords',
                     'has_password', 'has_token', 'has_secret', 'has_key'])
        
        # Pattern features
        names.extend(['weak_pwd_count', 'weak_crypto_count', 'insecure_proto_count',
                     'open_binding_count', 'http_urls', 'https_urls', 'http_ratio',
                     'ip_count', 'has_0000', 'has_localhost', 'base64_count', 'hex_count'])
        
        # Statistical features
        names.extend(['avg_value_length', 'max_value_length', 'min_value_length', 'length_variance',
                     'alpha_ratio', 'numeric_ratio', 'special_ratio', 'unique_chars',
                     'char_entropy', 'has_assignment'])
        
        # Entropy features
        names.extend(['avg_entropy', 'max_entropy', 'min_entropy', 'high_entropy_count'])
        
        # Network features
        names.extend(['insecure_ports', 'wellknown_ports', 'high_ports',
                     'has_ssl', 'has_tls', 'has_https', 'has_http_only',
                     'has_localhost_bind', 'has_127_bind', 'has_0000_bind'])
        
        # Crypto features
        names.extend(['has_md5', 'has_sha1', 'has_des', 'has_3des', 'has_rc4', 'has_rc2',
                     'has_sha256', 'has_sha512', 'has_sha3', 'has_aes', 'has_rsa2048', 'has_ecdsa',
                     'avg_key_length', 'has_weak_key', 'has_certificate', 'has_private_key', 'has_public_key'])
        
        return names
    
    def extract_batch_features(self, tasks: List[Tuple[Dict, str, Dict]]) -> np.ndarray:
        """
        Extract features from multiple tasks.
        """
        feature_matrix = []
        for task, module, params in tasks:
            features = self.extract_features(task, module, params)
            feature_matrix.append(features)
        
        return np.array(feature_matrix)