from typing import Dict, List
from statsmodels.stats.contingency_tables import mcnemar

def compare_models(per_instance_A: Dict[str, List[int]], per_instance_B: Dict[str, List[int]], exact: bool, correction: bool) -> Dict[str, float]:
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
