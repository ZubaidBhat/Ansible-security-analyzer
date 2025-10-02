"""
Main scanner CLI for detecting security smells in Ansible playbooks
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List
import time

from parsers import AnsibleParser
from detectors import SecurityDetector, SecuritySmell
from report import Reporter


class AnsibleSecurityScanner:
    
    def __init__(self):
        self.parser = AnsibleParser()
        self.detector = SecurityDetector()
        self.smells = []
    
    def scan_directory(self, directory: str) -> List[SecuritySmell]:
        """Scan all YAML files in a directory for security smells"""
        print(f"\n Scanning directory: {directory}")
        
        # Find all YAML files
        yaml_files = self.parser.find_yaml_files(directory)
        
        if not yaml_files:
            print(f"  No YAML files found in {directory}")
            return []
        
        print(f" Found {len(yaml_files)} YAML file(s)")
        
        all_smells = []
        files_with_issues = 0
        
        # Process each file
        for file_path in yaml_files:
            file_smells = self.scan_file(file_path)
            if file_smells:
                files_with_issues += 1
                all_smells.extend(file_smells)
        
        print(f"\n Scan complete: {len(yaml_files)} files scanned, {files_with_issues} with issues")
        
        return all_smells
    
    def scan_file(self, file_path: str) -> List[SecuritySmell]:
        """Scan a single YAML file for security smells"""
        print(f"   Scanning: {file_path}")
        
        # Load YAML content
        playbook = self.parser.load_yaml(file_path)
        
        if not playbook:
            return []
        
        file_smells = []
        
        # Traverse all tasks in the playbook
        for task, path, task_path in self.parser.traverse_tasks(playbook, file_path):
            if not isinstance(task, dict):
                continue
            
            # Extract module and parameters
            module, params = self.parser.extract_module_params(task)
            
            if module:
                # Run all detectors
                task_smells = self.detector.detect_all(task, path, task_path, module, params)
                file_smells.extend(task_smells)
        
        if file_smells:
            print(f"       Found {len(file_smells)} issue(s)")
        
        return file_smells


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Ansible Security Scanner - Detect security smells in Ansible playbooks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python src/scan.py samples/                  # Scan directory
  python src/scan.py samples/ --json          # Export to JSON
  python src/scan.py samples/ --chart         # Generate chart
  python src/scan.py samples/ --json results.json --chart --chart-out analysis.png
        '''
    )
    
    parser.add_argument(
        'path',
        help='Path to Ansible playbook file or directory'
    )
    
    parser.add_argument(
        '--json',
        nargs='?',
        const='security_smells.json',
        help='Export results to JSON file (default: security_smells.json)'
    )
    
    parser.add_argument(
        '--chart',
        action='store_true',
        help='Generate a bar chart of security smells'
    )
    
    parser.add_argument(
        '--chart-out',
        default='security_smells_chart.png',
        help='Output filename for chart (default: security_smells_chart.png)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress CLI output (useful with --json)'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.path):
        print(f" Error: Path '{args.path}' does not exist")
        sys.exit(1)
    
    # Initialize scanner
    scanner = AnsibleSecurityScanner()
    
    # Start timing
    start_time = time.time()
    
    # Perform scan
    if os.path.isfile(args.path):
        # Single file scan
        smells = scanner.scan_file(args.path)
    else:
        # Directory scan
        smells = scanner.scan_directory(args.path)
    
    # Calculate scan time
    scan_time = time.time() - start_time
    
    # Create reporter
    reporter = Reporter(smells)
    
    # Output results
    if not args.quiet:
        reporter.print_cli_report()
        print(f"\n  Scan completed in {scan_time:.2f} seconds")
    
    # Export JSON if requested
    if args.json:
        reporter.export_json(args.json)
    
    # Generate chart if requested
    if args.chart:
        reporter.generate_chart(args.chart_out)
    
    # Exit with appropriate code
    if smells:
        sys.exit(1)  # Issues found
    else:
        if not args.quiet:
            print("\n All clear! No security smells detected.")
        sys.exit(0)  # No issues


if __name__ == '__main__':
    main()