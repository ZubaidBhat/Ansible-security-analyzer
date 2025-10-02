"""
YAML parser module for loading and traversing Ansible playbooks.
"""
import yaml
import os
from typing import Dict, List, Any, Generator, Tuple
from pathlib import Path


class AnsibleParser:
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """Load a YAML file and return its content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except (yaml.YAMLError, FileNotFoundError) as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    @staticmethod
    def find_yaml_files(directory: str) -> List[str]:
        """Recursively find all YAML files in a directory."""
        yaml_files = []
        path = Path(directory)
        
        if not path.exists():
            print(f"Directory {directory} does not exist")
            return []
        
        for file_path in path.rglob('*.y*ml'):
            yaml_files.append(str(file_path))
        
        return yaml_files
    
    @staticmethod
    def traverse_tasks(playbook: Any, file_path: str) -> Generator[Tuple[Dict, str, List], None, None]:
        """
        Traverse all tasks in a playbook.
        """
        if isinstance(playbook, list):
            for play_idx, play in enumerate(playbook):
                if not isinstance(play, dict):
                    continue
                    
                # Check for tasks in the play
                if 'tasks' in play:
                    for task_idx, task in enumerate(play.get('tasks', [])):
                        task_path = [f"play[{play_idx}]", f"tasks[{task_idx}]"]
                        yield task, file_path, task_path
                
                # Check for pre_tasks
                if 'pre_tasks' in play:
                    for task_idx, task in enumerate(play.get('pre_tasks', [])):
                        task_path = [f"play[{play_idx}]", f"pre_tasks[{task_idx}]"]
                        yield task, file_path, task_path
                
                # Check for post_tasks
                if 'post_tasks' in play:
                    for task_idx, task in enumerate(play.get('post_tasks', [])):
                        task_path = [f"play[{play_idx}]", f"post_tasks[{task_idx}]"]
                        yield task, file_path, task_path
                
                # Check for handlers
                if 'handlers' in play:
                    for handler_idx, handler in enumerate(play.get('handlers', [])):
                        task_path = [f"play[{play_idx}]", f"handlers[{handler_idx}]"]
                        yield handler, file_path, task_path
                
                # Check for blocks within tasks
                for task_section in ['tasks', 'pre_tasks', 'post_tasks']:
                    for task_idx, task in enumerate(play.get(task_section, [])):
                        if 'block' in task:
                            for block_idx, block_task in enumerate(task['block']):
                                task_path = [f"play[{play_idx}]", f"{task_section}[{task_idx}]", f"block[{block_idx}]"]
                                yield block_task, file_path, task_path
    
    @staticmethod
    def extract_module_params(task: Dict) -> Tuple[str, Dict]:
        """
        Extract module name and parameters from a task.
        """
        # Skip meta tasks
        if 'name' in task and isinstance(task.get('name'), str):
            task_name = task['name']
        else:
            task_name = "unnamed task"
        
        # Common Ansible task keywords to exclude
        exclude_keys = {
            'name', 'when', 'with_items', 'loop', 'notify', 
            'tags', 'become', 'become_user', 'register', 'delegate_to',
            'ignore_errors', 'changed_when', 'failed_when', 'until',
            'retries', 'delay', 'vars', 'environment', 'run_once',
            'block', 'rescue', 'always'
        }
        
        # Find the module and its parameters
        for key, value in task.items():
            if key not in exclude_keys:
                # This is likely the module
                if isinstance(value, dict):
                    return key, value
                elif isinstance(value, str):
                    # Module with single string parameter
                    return key, {'_raw': value}
                else:
                    return key, {}
        
        return None, {}