import os
import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

def find_lacked_cases(exp_dir: str, test_sample_size: int, child_dir: str = None):
    lacked_cases = [] # list of (task_name, sample_idx)
    for task_dir in os.listdir(exp_dir):
        task_name = task_dir
        task_dir = os.path.join(exp_dir, task_dir)
        if os.path.isdir(task_dir):
            if child_dir is not None:
                task_dir = os.path.join(task_dir, child_dir)
            for sample_idx in range(test_sample_size):
                sample_path = os.path.join(task_dir, f"{sample_idx}.txt")
                if not os.path.exists(sample_path):
                    lacked_cases.append((task_name, sample_idx))
    return lacked_cases

def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--test_sample_size", type=int, required=True)
    parser.add_argument("--child_dir", type=str, default=None)
    parser.add_argument("--save_file", type=str, default="lacked_cases.json")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    lacked_case = find_lacked_cases(args.exp_dir, args.test_sample_size, args.child_dir)
    (Path(args.exp_dir) / args.save_file).write_text(json.dumps(lacked_case, indent=4))
    print(f"Found {len(lacked_case)} lacked cases")
