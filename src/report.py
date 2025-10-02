"""
Reporting module for outputting scan results in various formats.
"""
import json
import sys
from typing import List, Dict, Any
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non interactive backend

from detectors import SecuritySmell


class Reporter:
    """Handle various output formats for security scan results."""
    
    def __init__(self, smells: List[SecuritySmell]):
        self.smells = smells
    
    def print_cli_report(self, verbose: bool = True):
        """Print formatted CLI report."""
        if not self.smells:
            print("\n No security smells detected!")
            return
        
        print(f"\n  Found {len(self.smells)} security smell(s)\n")
        print("=" * 80)
        
        # Group smells by file
        smells_by_file = {}
        for smell in self.smells:
            if smell.file_path not in smells_by_file:
                smells_by_file[smell.file_path] = []
            smells_by_file[smell.file_path].append(smell)
        
        for file_path, file_smells in smells_by_file.items():
            print(f"\n File: {file_path}")
            print("-" * 80)
            
            for idx, smell in enumerate(file_smells, 1):
                self._print_smell(smell, idx)
        
        # Print summary
        self._print_summary()
    
    def _print_smell(self, smell: SecuritySmell, index: int):
        """Print a single smell in formatted output."""
        severity_icons = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        
        print(f"\n  [{index}] {severity_icons.get(smell.severity, '⚪')} {smell.smell_type}")
        print(f"       Location: {smell.task_path}")
        print(f"       Task: {smell.task_name}")
        print(f"       Module: {smell.module}")
        print(f"       Issue: {smell.description}")
        print(f"       Fix: {smell.fix_suggestion}")
        if smell.line_content:
            print(f"       Content: {smell.line_content}")
    
    def _print_summary(self):
        """Print summary statistics."""
        smell_counts = Counter(smell.smell_type for smell in self.smells)
        severity_counts = Counter(smell.severity for smell in self.smells)
        
        print("\n" + "=" * 80)
        print(" SUMMARY")
        print("=" * 80)
        
        print("\nBy Type:")
        for smell_type, count in smell_counts.most_common():
            print(f"  • {smell_type}: {count}")
        
        print("\nBy Severity:")
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            if severity in severity_counts:
                print(f"  • {severity}: {severity_counts[severity]}")
        
        print("\n" + "=" * 80)
    
    def export_json(self, output_file: str = None) -> str:
        """Export results as JSON."""
        results = {
            'total_smells': len(self.smells),
            'smells': []
        }
        
        for smell in self.smells:
            results['smells'].append({
                'type': smell.smell_type,
                'file': smell.file_path,
                'location': smell.task_path,
                'task_name': smell.task_name,
                'module': smell.module,
                'description': smell.description,
                'fix': smell.fix_suggestion,
                'severity': smell.severity,
                'content': smell.line_content
            })
        
        # Add summary
        smell_counts = Counter(smell.smell_type for smell in self.smells)
        results['summary'] = {
            'by_type': dict(smell_counts),
            'by_severity': dict(Counter(smell.severity for smell in self.smells)),
            'files_affected': len(set(smell.file_path for smell in self.smells))
        }
        
        json_str = json.dumps(results, indent=2)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_str)
            print(f"\n JSON report saved to: {output_file}")
        
        return json_str
    
    def generate_chart(self, output_file: str = "security_smells_chart.png"):
        """Generate a bar chart of security smells."""
        if not self.smells:
            print("No smells to chart")
            return
        
        smell_counts = Counter(smell.smell_type for smell in self.smells)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        types = list(smell_counts.keys())
        counts = list(smell_counts.values())
        
        # Create bar chart
        bars = ax.bar(types, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        
        # Customize the chart
        ax.set_xlabel('Security Smell Type', fontsize=12)
        ax.set_ylabel('Number of Occurrences', fontsize=12)
        ax.set_title('Security Smells Detection Results', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}',
                   ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"\n Chart saved to: {output_file}")
        
        # Close the plot to free memory
        plt.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected smells."""
        if not self.smells:
            return {
                'total': 0,
                'by_type': {},
                'by_severity': {},
                'by_file': {},
                'files_affected': 0
            }
        
        return {
            'total': len(self.smells),
            'by_type': dict(Counter(smell.smell_type for smell in self.smells)),
            'by_severity': dict(Counter(smell.severity for smell in self.smells)),
            'by_file': dict(Counter(smell.file_path for smell in self.smells)),
            'files_affected': len(set(smell.file_path for smell in self.smells))
        }