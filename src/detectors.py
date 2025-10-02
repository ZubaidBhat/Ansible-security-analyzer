"""
Security smell detectors for Ansible playbooks.
"""
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class SecuritySmell:
    """Represents a detected security smell."""
    smell_type: str
    file_path: str
    task_path: str
    task_name: str
    module: str
    description: str
    fix_suggestion: str
    severity: str  # LOW, MEDIUM, HIGH
    line_content: Optional[str] = None


class SecurityDetector:
    """Base class for security smell detectors."""
    
    def __init__(self):
        self.smells = []
    
    def detect_http_without_tls(self, task: Dict, file_path: str, task_path: List, module: str, params: Dict) -> Optional[SecuritySmell]:
        """Detect HTTP URLs without TLS."""
        # Check for http:// URLs in various fields
        http_pattern = re.compile(r'http://[^\s]+', re.IGNORECASE)
        
        def search_dict(d, path=""):
            results = []
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, str) and http_pattern.search(value):
                        # Exclude localhost and 127.0.0.1
                        if not re.search(r'http://(localhost|127\.0\.0\.1)', value, re.IGNORECASE):
                            results.append((f"{path}.{key}" if path else key, value))
                    elif isinstance(value, (dict, list)):
                        results.extend(search_dict(value, f"{path}.{key}" if path else key))
            elif isinstance(d, list):
                for idx, item in enumerate(d):
                    if isinstance(item, str) and http_pattern.search(item):
                        if not re.search(r'http://(localhost|127\.0\.0\.1)', item, re.IGNORECASE):
                            results.append((f"{path}[{idx}]", item))
                    elif isinstance(item, (dict, list)):
                        results.extend(search_dict(item, f"{path}[{idx}]"))
            return results
        
        findings = search_dict(params)
        if findings:
            path_str, content = findings[0]
            return SecuritySmell(
                smell_type="HTTP_WITHOUT_TLS",
                file_path=file_path,
                task_path=".".join(task_path),
                task_name=task.get('name', 'unnamed'),
                module=module,
                description=f"Insecure HTTP URL found in {path_str}",
                fix_suggestion="Use HTTPS instead of HTTP for secure communication",
                severity="HIGH",
                line_content=content[:100]
            )
        return None
    
    def detect_open_binding(self, task: Dict, file_path: str, task_path: List, module: str, params: Dict) -> Optional[SecuritySmell]:
        """Detect binding to 0.0.0.0 (all interfaces)."""
        # Common parameters where binding addresses appear
        binding_params = ['host', 'bind', 'listen', 'address', 'bind_address', 'listen_address']
        
        for param in binding_params:
            if param in params:
                value = str(params[param])
                if '0.0.0.0' in value or value == '0':
                    return SecuritySmell(
                        smell_type="OPEN_BINDING",
                        file_path=file_path,
                        task_path=".".join(task_path),
                        task_name=task.get('name', 'unnamed'),
                        module=module,
                        description=f"Service bound to all interfaces (0.0.0.0) in parameter '{param}'",
                        fix_suggestion="Bind to specific interface (e.g., 127.0.0.1 for localhost only)",
                        severity="MEDIUM",
                        line_content=value
                    )
        
        # Check in string content
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, str) and '0.0.0.0' in value:
                    return SecuritySmell(
                        smell_type="OPEN_BINDING",
                        file_path=file_path,
                        task_path=".".join(task_path),
                        task_name=task.get('name', 'unnamed'),
                        module=module,
                        description=f"Service bound to all interfaces (0.0.0.0) in parameter '{key}'",
                        fix_suggestion="Bind to specific interface (e.g., 127.0.0.1 for localhost only)",
                        severity="MEDIUM",
                        line_content=value[:100]
                    )
        return None
    
    def detect_hardcoded_secrets(self, task: Dict, file_path: str, task_path: List, module: str, params: Dict) -> Optional[SecuritySmell]:
        """Detect hardcoded secrets, tokens, and API keys."""
        # Patterns for detecting secrets
        secret_patterns = [
            (re.compile(r'(api[_-]?key|apikey)\s*[:=]\s*["\']?[\w\-]{20,}', re.IGNORECASE), "API key"),
            (re.compile(r'(token|access[_-]?token)\s*[:=]\s*["\']?[\w\-]{20,}', re.IGNORECASE), "Access token"),
            (re.compile(r'(secret|private[_-]?key)\s*[:=]\s*["\']?[\w\-]{20,}', re.IGNORECASE), "Secret/Private key"),
            (re.compile(r'BEGIN\s+(RSA|DSA|EC|OPENSSH)\s+PRIVATE\s+KEY', re.IGNORECASE), "Private key"),
            (re.compile(r'aws[_-]?access[_-]?key[_-]?id\s*[:=]\s*["\']?[A-Z0-9]{20}', re.IGNORECASE), "AWS Access Key"),
            (re.compile(r'aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*["\']?[\w\+/]{40}', re.IGNORECASE), "AWS Secret Key"),
        ]
        
        def search_value(value):
            if isinstance(value, str):
                for pattern, secret_type in secret_patterns:
                    if pattern.search(value):
                        # Check if it's a variable reference (Ansible variable)
                        if not re.match(r'{{\s*\w+\s*}}', value.strip()):
                            return secret_type, value
            return None, None
        
        # Check all parameters
        for key, value in params.items() if isinstance(params, dict) else []:
            secret_type, content = search_value(str(value))
            if secret_type:
                return SecuritySmell(
                    smell_type="HARDCODED_SECRET",
                    file_path=file_path,
                    task_path=".".join(task_path),
                    task_name=task.get('name', 'unnamed'),
                    module=module,
                    description=f"Hardcoded {secret_type} found in parameter '{key}'",
                    fix_suggestion="Use Ansible Vault or environment variables to store secrets",
                    severity="HIGH",
                    line_content=content[:50] + "..." if len(content) > 50 else content
                )
        return None
    
    def detect_weak_password(self, task: Dict, file_path: str, task_path: List, module: str, params: Dict) -> Optional[SecuritySmell]:
        """Detect weak or empty passwords."""
        weak_passwords = [
            'password', 'admin', '12345', '123456', 'password123', 'admin123',
            'root', 'toor', 'pass', 'test', 'guest', 'default', 'changeme',
            '1234567890', 'qwerty', 'abc123', 'letmein', 'welcome', 'monkey',
            '123456789', 'password1', '12345678', '1234', '111111', '123123'
        ]
        
        password_params = ['password', 'passwd', 'pass', 'pwd', 'secret', 'passphrase']
        
        for param in password_params:
            if param in params:
                value = str(params[param]).strip().lower()
                # Check for empty password
                if not value or value in ['""', "''", 'none', 'null']:
                    return SecuritySmell(
                        smell_type="EMPTY_PASSWORD",
                        file_path=file_path,
                        task_path=".".join(task_path),
                        task_name=task.get('name', 'unnamed'),
                        module=module,
                        description=f"Empty password in parameter '{param}'",
                        fix_suggestion="Set a strong password using Ansible Vault",
                        severity="HIGH",
                        line_content=str(params[param])
                    )
                # Check for weak password
                elif value in weak_passwords:
                    return SecuritySmell(
                        smell_type="WEAK_PASSWORD",
                        file_path=file_path,
                        task_path=".".join(task_path),
                        task_name=task.get('name', 'unnamed'),
                        module=module,
                        description=f"Weak password '{value}' in parameter '{param}'",
                        fix_suggestion="Use a strong password with mix of characters, stored in Ansible Vault",
                        severity="HIGH",
                        line_content=str(params[param])
                    )
        return None
    
    def detect_weak_crypto(self, task: Dict, file_path: str, task_path: List, module: str, params: Dict) -> Optional[SecuritySmell]:
        """Detect weak cryptographic algorithms."""
        weak_algorithms = {
            'md5': 'MD5 is cryptographically broken',
            'sha1': 'SHA1 is deprecated for security use',
            'des': 'DES is obsolete and insecure',
            '3des': '3DES is deprecated',
            'rc4': 'RC4 is broken and should not be used',
            'rc2': 'RC2 is weak and obsolete'
        }
        
        def check_crypto(value):
            if isinstance(value, str):
                value_lower = value.lower()
                for algo, reason in weak_algorithms.items():
                    if algo in value_lower:
                        # Check if it's actually being used for crypto
                        crypto_indicators = ['hash', 'crypt', 'cipher', 'algorithm', 'digest', 'encrypt']
                        if any(indicator in value_lower for indicator in crypto_indicators):
                            return algo.upper(), reason
            return None, None
        
        # Check module specific crypto parameters
        crypto_params = ['algorithm', 'hash', 'cipher', 'crypto', 'digest', 'encryption', 'hash_type']
        
        for param in crypto_params:
            if param in params:
                algo, reason = check_crypto(str(params[param]))
                if algo:
                    return SecuritySmell(
                        smell_type="WEAK_CRYPTOGRAPHY",
                        file_path=file_path,
                        task_path=".".join(task_path),
                        task_name=task.get('name', 'unnamed'),
                        module=module,
                        description=f"Weak cryptographic algorithm {algo} used: {reason}",
                        fix_suggestion="Use strong algorithms like SHA256, SHA3, or AES",
                        severity="HIGH",
                        line_content=str(params[param])
                    )
        
        # Check in general content
        for key, value in params.items() if isinstance(params, dict) else []:
            algo, reason = check_crypto(str(value))
            if algo:
                return SecuritySmell(
                    smell_type="WEAK_CRYPTOGRAPHY",
                    file_path=file_path,
                    task_path=".".join(task_path),
                    task_name=task.get('name', 'unnamed'),
                    module=module,
                    description=f"Weak cryptographic algorithm {algo} found: {reason}",
                    fix_suggestion="Use strong algorithms like SHA256, SHA3, or AES",
                    severity="HIGH",
                    line_content=str(value)[:100]
                )
        return None
    
    def detect_all(self, task: Dict, file_path: str, task_path: List, module: str, params: Dict) -> List[SecuritySmell]:
        """Run all detectors on a task."""
        detectors = [
            self.detect_http_without_tls,
            self.detect_open_binding,
            self.detect_hardcoded_secrets,
            self.detect_weak_password,
            self.detect_weak_crypto
        ]
        
        smells = []
        for detector in detectors:
            smell = detector(task, file_path, task_path, module, params)
            if smell:
                smells.append(smell)
        
        return smells