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
        if label_type == "class":
            self._label_set = set([sample["target"] for sample in samples])
        else: # TODO
            raise NotImplementedError
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
    def label_size(self) -> int:
        return len(self._label_set)
    
    @property
    def label_set(self) -> set:
        return self._label_set
    
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
    
    def get_new_labels(self) -> Union[str, List[str]]:
        labels = []
        for _ in range(self._batch_size):
            if self._counter < self.sample_size:
                labels.append(self._samples[self._counter]["target"])
                self._counter += 1
        return labels if self._batch_size > 1 else labels[0]
    
    def get_inputs(self, sample_ids: List[int]) -> Union[str, List[str]]:
        inputs = []
        for sample_id in sample_ids:
            inputs.append(self._samples[sample_id]["input"])
        return inputs if len(sample_ids) > 1 else inputs[0]
    
    def set_counter(self, counter: int) -> None:
        self._counter = counter

class TaskGenerator(object):
    task2label_type = {
        # BBH
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
        "word_sorting": "permutation",
        # MMLU
        "abstract_algebra": "choice",
        "anatomy": "choice",
        "astronomy": "choice",
        "business_ethics": "choice",
        "clinical_knowledge": "choice",
        "college_biology": "choice",
        "college_chemistry": "choice",
        "college_computer_science": "choice",
        "college_mathematics": "choice",
        "college_medicine": "choice",
        "college_physics": "choice",
        "computer_security": "choice",
        "conceptual_physics": "choice",
        "econometrics": "choice",
        "electrical_engineering": "choice",
        "elementary_mathematics": "choice",
        "formal_logic": "choice",
        "global_facts": "choice",
        "high_school_biology": "choice",
        "high_school_chemistry": "choice",
        "high_school_computer_science": "choice",
        "high_school_european_history": "choice",
        "high_school_geography": "choice",
        "high_school_government_and_politics": "choice",
        "high_school_macroeconomics": "choice",
        "high_school_mathematics": "choice",
        "high_school_microeconomics": "choice",
        "high_school_physics": "choice",
        "high_school_psychology": "choice",
        "high_school_statistics": "choice",
        "high_school_us_history": "choice",
        "high_school_world_history": "choice",
        "human_aging": "choice",
        "human_sexuality": "choice",
        "international_law": "choice",
        "jurisprudence": "choice",
        "logical_fallacies": "choice",
        "machine_learning": "choice",
        "management": "choice",
        "marketing": "choice",
        "medical_genetics": "choice",
        "miscellaneous": "choice",
        "moral_disputes": "choice",
        "moral_scenarios": "choice",
        "nutrition": "choice",
        "philosophy": "choice",
        "prehistory": "choice",
        "professional_accounting": "choice",
        "professional_law": "choice",
        "professional_medicine": "choice",
        "professional_psychology": "choice",
        "public_relations": "choice",
        "security_studies": "choice",
        "sociology": "choice",
        "us_foreign_policy": "choice",
        "virology": "choice",
        "world_religions": "choice"
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
    
    @property
    def task2desc(self) -> Dict[str, str]:
        return self._task2desc

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

# unit tests
if __name__ == "__main__":
    task_input_path = "./bbh/BIG-Bench-Hard/bbh/"
    task_desc_path = "./bbh/bbh_task_description.json"
    batch_size = 1
    
    task_gen = TaskGenerator(task_input_path, task_desc_path, batch_size)
    for task_name, label_type in TaskGenerator.task2label_type.items():
        if label_type == "class":
            task = task_gen.get_task(task_name)
            print("Label: ", end="")
            for label in task.label_set:
                print(label, end=" ")
            print(end="\n\n")