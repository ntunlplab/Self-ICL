import json
from typing import List, Dict, Union
from pathlib import Path

class Task(object):
    
    def __init__(
        self,
        task_desc: str,
        samples: List[Dict[str, str]],
        label_type: str,
        batch_size: int
    ):
        self._task_desc = task_desc
        self._samples = samples
        self._label_type = label_type # has to do with label sampling
        self._batch_size = batch_size
        self._counter = 0
    
    @property
    def task_desc(self) -> str:
        return self._task_desc
    
    @property
    def sample_size(self) -> int:
        return len(self._samples)
    
    @property
    def label_type(self) -> str:
        return self._label_type
    
    @property
    def counter(self) -> int:
        return self._counter
    
    def get_new_inputs(self) -> Union[str, List[str]]:
        inputs = []
        for _ in range(self._batch_size):
            if self._counter < self.sample_size:
                inputs.append(self._samples[self._counter]["input"])
                self._counter += 1
        return inputs if self._batch_size > 1 else inputs[0]

class TaskGenerator(object):
    task2label_type = {
        "boolean_expressions": "class",
        "causal_judgement": "class",
        "date_understanding": "class",
        "disambiguation_qa": "class",
        "dyck_languages": "sequence",
        "formal_fallacies": "class",
        "geometric_shapes": "class",
        "hyperbaton": "class",
        "logical_deduction_five_objects": "class",
        "logical_deduction_seven_objects": "class",
        "logical_deduction_three_objects": "class",
        "movie_recommendation": "class",
        "multistep_arithmetic_two": "integer", # both positive and negative
        "navigate": "class",
        "object_counting": "integer", # positive only
        "penguins_in_a_table": "class",
        "reasoning_about_colored_objects": "class",
        "ruin_names": "class",
        "salient_translation_error_detection": "class",
        "snarks": "class",
        "sports_understanding": "class",
        "temporal_sequences": "class",
        "tracking_shuffled_objects_five_objects": "class",
        "tracking_shuffled_objects_seven_objects": "class",
        "tracking_shuffled_objects_three_objects": "class",
        "web_of_lies": "class",
        "word_sorting": "permutation"
    }
    
    def __init__(
        self,
        task_input_path: str,
        task_desc_path: str,
        batch_size: int,
        verbose: bool = True
    ):
        self._task_input_path = Path(task_input_path)
        self._batch_size = batch_size
        self._task2desc = json.loads(Path(task_desc_path).read_text())
        self._verbose = verbose

    def get_task(self, task_name: str) -> Task:
        if task_name not in self._task2desc:
            raise ValueError(f"Task {task_name} not found")
        if self._verbose:
            print(f"Generating task '{task_name}'...")
        return Task(
            task_desc=self._task2desc[task_name],
            samples=json.loads((self._task_input_path / f"{task_name}.json").read_text())["examples"],
            label_type=self.task2label_type[task_name],
            batch_size=self._batch_size
        )