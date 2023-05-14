import sys
import json
from pathlib import Path

def mkdir_for_tasks(task_desc_file: str, parent_dir: str):
    task2desc = json.loads(Path(task_desc_file).read_text())
    for task, _ in task2desc.items():
        task_dir = Path(parent_dir) / task
        task_dir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    task_desc_file = sys.argv[1] # "../../bbh/bbh_task_description.json"
    parent_dir = sys.argv[2] # "./playground"
    mkdir_for_tasks(task_desc_file, parent_dir)