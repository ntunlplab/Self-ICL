import json
from pathlib import Path
from typing import Dict, List
from statsmodels.stats.contingency_tables import mcnemar

def compare_models(per_instance_A: Dict[str, List[int]], per_instance_B: Dict[str, List[int]], exact: bool, correction: bool) -> Dict[str, float]:
    """
    Suppose there are two models A and B.
    Item B✘ B✔
    A✘ [[a, b],
    A✔  [c, d]]
    """
    # check equal tasks
    for task_name in per_instance_A:
        if task_name not in per_instance_B:
            raise ValueError("Task name {} not found in model B".format(task_name))
    
    res = dict()
    mcnemar_all = [[0, 0], [0, 0]]
    for task_name in per_instance_A:
        mcnemar_task = [[0, 0], [0, 0]]
        binary_res_A = per_instance_A[task_name]
        binary_res_B = per_instance_B[task_name]
        if len(binary_res_A) != len(binary_res_B):
            raise ValueError("Task {} has different number of instances in two models".format(task_name))
        for a, b in zip(binary_res_A, binary_res_B):
            if (a == 0) and (b == 0):
                mcnemar_task[0][0] += 1
            elif (a == 0) and (b == 1):
                mcnemar_task[0][1] += 1
            elif (a == 1) and (b == 0):
                mcnemar_task[1][0] += 1
            elif (a == 1) and (b == 1):
                mcnemar_task[1][1] += 1
            else:
                raise ValueError("Unexpected values in binary results")
        for i in range(4):
            mcnemar_all[i // 2][i % 2] += mcnemar_task[i // 2][i % 2]
        res[task_name] = mcnemar(mcnemar_task, exact=exact, correction=correction).pvalue
    res["all"] = mcnemar(mcnemar_all, exact=exact, correction=correction).pvalue
    return res

if __name__ == "__main__":
    prefix_dir = "../model_outputs/stream"
    suffix_file = "per_instance_100-testsize.json"

    comps = {
        "zero-shot": {
            "A": "standard-zero-shot-class",
            "B": "self-icl-no-cot-diverse-class"
        },
        "cot": {
            "A": "standard-zero-shot-cot-class",
            "B": "self-icl-cot-diverse-class"
        },
        "cot_vs_self_icl": {
            "A": "standard-zero-shot-cot-class",
            "B": "self-icl-no-cot-diverse-class"
        }
    }
        
    for comp_name, d in comps.items():
        print(f"Setting: {comp_name}")
        print(f"Comparing {d['A']} and {d['B']}")
        A_path = (Path(prefix_dir) / d['A'] / suffix_file)
        B_path = (Path(prefix_dir) / d['B'] / suffix_file)
        
        per_instance_A = json.loads(A_path.read_text())
        per_instance_B = json.loads(B_path.read_text())
        
        res = compare_models(per_instance_A, per_instance_B, exact=False, correction=True)
        print(f"p-value = {res['all']}")
        print()