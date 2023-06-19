import json
from typing import List, Dict
from pathlib import Path

def build_taskname2desc(old_taskname2desc: List[Dict[str, str]]) -> Dict[str, str]:
    taskname2desc = {}
    for task in old_taskname2desc:
        taskname2desc[task["task_name"]] = task["task_description"]
    return taskname2desc

if __name__ == "__main__":
    old_path = Path("./mmlu-task_description.json")
    new_path = Path("./mmlu-task_description_formatted.json")
    
    old_taskname2desc = json.loads(old_path.read_text())
    new_taskname2desc = build_taskname2desc(old_taskname2desc)
    
    new_path.write_text(json.dumps(new_taskname2desc, indent=4))