import os
import math
import json
import yaml
import pandas as pd
import random
from typing import Union
from pathlib import Path
from argparse import ArgumentParser, Namespace
from colorama import Fore, Style
from dataclasses import dataclass
from task import TaskGenerator
from model import Model
from prompt import StreamPrompt, BatchPrompt, Shot
from promptparser import PromptParser

@dataclass
class Config(object):
    exp_name: str
    # paths
    task_input_path: str
    task_desc_path: str
    few_shots_path: str
    log_path: str
    # experiment settings
    inference_mode: str # "stream" or "batch"
    exemplars_mode: str # "self-icl" or "standard"
    num_demos: int # self-icl: number of pseudo-demos to generate; standard: number of real demos to use (0: zero-shot, >0: few-shot a.k.a. standrad ICL)
    use_cot: bool # whether to use chain-of-thought
    label_method: str # "self" (LLM-generated) or "random" (randomly sample from the label space) -> only used when exemplars_mode == "self-icl"
    diverse_exemplars: bool # whether to generate diverse exemplars -> only used when exemplars_mode == "self-icl"
    # sizes
    batch_size: int # only used when inference_mode == "batch"
    test_sample_size: Union[int, str] # "full" or int
    # model hparams
    model: str # e.g., text-davinci-003
    max_tokens: int
    temperature: float
    demo_temperature: float # temperature when generating pseudo-demos (only used when exemplars_mode == "self-icl")
    top_p: float

class Experiment(object):
    self_icl_subdirs = ["demo-inputs", "demo-labels", "full-outputs"]
    cot_check = Fore.GREEN + "✔" + Style.RESET_ALL
    cot_cross = Fore.RED + "✘" + Style.RESET_ALL
    
    def __init__(self, config: Config) -> None:
        print(f"Initializing experiment {config.exp_name}...")
        self._config = config
        self._model = Model(
            Namespace(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
            )
        )
        # ensure correct batch_size
        if self._config.inference_mode == "stream":
            self._config.batch_size = 1
        elif self._config.inference_mode == "batch":
            assert self._config.batch_size > 1
        else:
            raise ValueError(f"Invalid inference_mode: {self._config.inference_mode}")
        # ensure correct exemplars_mode
        if self._config.exemplars_mode not in ["self-icl", "standard"]:
            raise ValueError(f"Invalid exemplars_mode: {self._config.exemplars_mode}")
        # make logging path
        self._log_path = Path(self._config.log_path) / self._config.inference_mode / self._config.exp_name
        self._log_path.mkdir(parents=True, exist_ok=True)
        (self._log_path / "config.yml").write_text(yaml.dump(vars(self._config)))
        # make prompt parser
        self._prompt_parser = PromptParser(num_demos=self._config.num_demos)
        
    def print_configs(self) -> None:
        print("Configs:")
        for k, v in vars(self._config).items():
            print(f"\t{k}: {v}")

    def run(
        self,
        task_continue_from: str = None,
        sample_start_from: int = 0,
        label_type: str = None,
        lacked_cases_path: Path = None,
        existing_demos_path: Path = None,
        random_pseudo_label: bool = False,
        rerun_pseudo_label: bool = False,
        step2_stream: bool = False,
        step3_stream: bool = False
    ) -> None:
        # handle passed arguments
        if label_type and (label_type not in set(TaskGenerator.task2label_type.values())):
            raise ValueError(f"Invalid label_type: {label_type}")
        if random_pseudo_label:
            assert existing_demos_path is not None

        lacked_cases_dict = dict() # dict of {task_name: set(sample_idx)}
        if lacked_cases_path:
            lacked_cases = json.loads(lacked_cases_path.read_text())
            for task_name, sample_idx in lacked_cases:
                if task_name not in lacked_cases_dict:
                    lacked_cases_dict[task_name] = set()
                lacked_cases_dict[task_name].add(sample_idx)
        
        self.print_configs()

        task_generator = TaskGenerator(
            task_input_path=self._config.task_input_path,
            task_desc_path=self._config.task_desc_path,
            batch_size=self._config.batch_size,
            verbose=True
        )
        continue_flag = True if task_continue_from else False
        failed_cases = [] # list of (task_name, sample_idx)
        for task_name in task_generator.task2desc.keys():
            # skip tasks before continue_from
            if continue_flag and (task_name != task_continue_from):
                continue
            continue_flag = False
            
            # make task log dir
            task_log_path = self._log_path / task_name
            task_log_path.mkdir(parents=True, exist_ok=True)
            if self._config.exemplars_mode == "self-icl":
                # make pseudo-demos log dir
                for subdir in self.self_icl_subdirs:
                    (task_log_path / subdir).mkdir(parents=True, exist_ok=True)
            
            # skip tasks with not specified 
            task_label_type = TaskGenerator.task2label_type[task_name]
            if label_type and (task_label_type != label_type):
                print(f"Skipping task {task_name} with label_type {task_label_type}...")
                continue
            
            # if lacked_cases_path is specified, skip tasks with no lacked cases
            if lacked_cases_path and (task_name not in lacked_cases_dict):
                print(f"Skipping task {task_name} with no lacked cases...")
                continue
            
            task = task_generator.get_task(task_name)
            add_parenthesis = list(task.label_set)[0][0] == '('
            if (task.label_type in ["class", "choice"]) and (not self._config.use_cot) and ((self._config.inference_mode == "stream") or step3_stream):
                label_set = task.label_set
            else:
                label_set = None

            if type(self._config.test_sample_size) == int:
                num_runs = math.ceil(self._config.test_sample_size / self._config.batch_size)
            elif self._config.test_sample_size == "full":
                num_runs = math.ceil(task.sample_size / self._config.batch_size)
            for i in range(num_runs):
                # skip samples before start_from
                if (i < sample_start_from) or (i >= task.sample_size // self._config.batch_size):
                    print(f"Skipping {'sample' if self._config.inference_mode == 'stream' else 'batch'} #{i}...")
                    continue
                # if lacked_cases_path is specified, skip samples with no lacked cases
                if lacked_cases_path and (i not in lacked_cases_dict[task_name]):
                    print(f"Skipping sample #{i} with no lacked cases...")
                    continue
                
                print(f"Running sample #{i}:")
                task_inputs = task.get_inputs(sample_ids=list(range(i * self._config.batch_size, (i + 1) * self._config.batch_size)))
                shots = []
                # get bbh shots for the standard few-shot setting
                if (self._config.exemplars_mode == "standard") and (self._config.num_demos > 0):
                    parsed_shots = json.loads((Path(self._config.few_shots_path) / f"{task_name}.json").read_text())
                    if len(parsed_shots) != self._config.num_demos:
                        raise ValueError(f"Invalid num_shots: {len(parsed_shots)} (expected {self._config.num_demos})")
                    for d in parsed_shots:
                        shot = Shot(_input=d['Q'], _label=d['A'])
                        shots.append(shot)
                # prepare initial prompt
                if self._config.inference_mode == "stream":
                    prompt = StreamPrompt(
                        task_desc=task.task_desc,
                        inputs=task_inputs,
                        num_demos=self._config.num_demos,
                        shots=shots
                    )
                else: # batch
                    prompt = BatchPrompt(
                        task_desc=task.task_desc,
                        inputs=task_inputs,
                        num_demos=self._config.num_demos,
                        shots=shots
                    )
                # augment prompt with pseudo-demos if "self-icl"
                if self._config.exemplars_mode == "self-icl":
                    # 1. Pseudo-demo inputs
                    if existing_demos_path is None: # generate pseudo-demo inputs
                        demo_prompt = prompt.gen_demo_inputs(diversity=self._config.diverse_exemplars)
                        demo_prompt, demo_inputs = self._model.complete(demo_prompt, label_set=None, temperature=self._config.demo_temperature)
                        full_demo_inputs = demo_prompt + demo_inputs
                        (task_log_path / "demo-inputs" / f"{i}.txt").write_text(full_demo_inputs)
                    else: # read pseudo-demo inputs
                        print(f"Reading demo-inputs in sample #{i}...")
                        full_demo_inputs = (existing_demos_path / task_name / "demo-inputs" / f"{i}.txt").read_text()
                    # parse demo inputs to separate instances
                    try:
                        sep_demo_inputs = self._prompt_parser.split_demo_inputs(full_demo_inputs)
                    except ValueError:
                        print(Fore.RED + f"Task {task_name} sample #{i} failed: failed to parse demo inputs" + Style.RESET_ALL)
                        failed_cases.append([task_name, i])
                        continue
                    # 2. Pseudo-demo labels
                    if (existing_demos_path is None) or rerun_pseudo_label:
                        if (self._config.inference_mode == "stream") or step2_stream:
                            shots = []
                            for j, sep_demo_input in enumerate(sep_demo_inputs):
                                sep_demo_prompt = StreamPrompt(
                                    task_desc=task.task_desc,
                                    inputs=sep_demo_input,
                                    num_demos=0, # NOTE
                                    shots=[]
                                ).gen_prediction(cot=self._config.use_cot)
                                print(f"Predicting demo #{j} (cot: {self.cot_check if self._config.use_cot else self.cot_cross}) -> ", end='')
                                sep_demo_prompt, sep_demo_label = self._model.complete(sep_demo_prompt, label_set, temperature=self._config.temperature)
                                if sep_demo_prompt[-1] == '(':
                                    sep_demo_label = '(' + sep_demo_label
                                shot = Shot(_input=sep_demo_input, _label=sep_demo_label.strip())
                                shots.append(shot)
                                # logging
                                print(sep_demo_label)
                                (task_log_path / "demo-labels" / f"{i}-{j}.txt").write_text(str(shot))
                        else:
                            print(f"Predicting demo-labels in sample #{i}...")
                            sep_demo_prompt = BatchPrompt(
                                task_desc=task.task_desc,
                                inputs=sep_demo_inputs,
                                num_demos=0, # NOTE
                                shots=[]
                            ).gen_prediction(add_parenthesis=add_parenthesis)
                            sep_demo_prompt, sep_demo_label = self._model.complete(sep_demo_prompt, label_set, temperature=self._config.temperature)
                            # update prompt to augmented prompt (shots == full_demo)
                            shots = sep_demo_prompt + sep_demo_label
                            shots = shots[shots.index("Q1:"):] # remove task description
                            print(sep_demo_label)
                            (task_log_path / "demo-labels" / f"{i}.txt").write_text(str(shots))
                        if step3_stream:
                            for j, task_input in enumerate(task_inputs):
                                print(f"Predicting batch #{i}-{j} (cot: {self.cot_check if self._config.use_cot else self.cot_cross}) -> ", end='')
                                prompt = StreamPrompt(
                                    task_desc=task.task_desc,
                                    inputs=task_input,
                                    num_demos=self._config.num_demos,
                                    shots=shots
                                )
                                pred_prompt = prompt.gen_prediction(cot=self._config.use_cot, add_parenthesis=add_parenthesis)
                                pred_prompt, res_text = self._model.complete(pred_prompt, label_set, temperature=self._config.temperature)
                                print(res_text)
                                full_text = pred_prompt + res_text
                                (task_log_path / "full-outputs" / f"{i * self._config.batch_size + j}.txt").write_text(full_text)
                            # continue to next run (i.e., next batch)
                            continue    
                    else: # (existing_demos_path is not None) and (not rerun_pseudo_label)
                        if self._config.inference_mode == "stream":
                            num_existing_demos = len(os.listdir(existing_demos_path / task_name / "demo-labels")) // task.sample_size
                            if self._config.num_demos == num_existing_demos:
                                js = list(range(num_existing_demos))
                            else:
                                js = random.sample(range(num_existing_demos), self._config.num_demos)
                            for j in js:
                                demo_input_label = (existing_demos_path / task_name / "demo-labels" / f"{i}-{j}.txt").read_text()
                                q_start = demo_input_label.index("Q:")
                                a_start = demo_input_label.index("A:")
                                demo_input = demo_input_label[q_start+len("Q:"):a_start].strip()
                                if random_pseudo_label:
                                    demo_label = random.sample(task.label_set, 1)[0]
                                    print("(random) ", end='')
                                else:
                                    demo_label = demo_input_label[a_start+len("A:"):].strip()
                                    print("(cached) ", end='')
                                print(f"Adding existing demo #{j} -> {demo_label}")
                                shot = Shot(_input=demo_input, _label=demo_label)
                                shots.append(shot)
                        else: # batch
                            raw_shots = (existing_demos_path / task_name / "demo-labels" / f"{i}.txt").read_text()
                            if step3_stream:
                                # parse shots to separate shots
                                shots = []
                                for k in range(1, self._config.num_demos + 1):
                                    q_start = raw_shots.index(f"Q{k}:") + len(f"Q{k}:")
                                    a_start = raw_shots.index(f"A{k}:") + len(f"A{k}:")
                                    q_end = raw_shots.index(f"Q{k + 1}:") if k < self._config.num_demos else raw_shots.index("A1")
                                    a_end = raw_shots.index(f"A{k + 1}:") if k < self._config.num_demos else len(raw_shots)
                                    q = raw_shots[q_start:q_end].strip()
                                    a = raw_shots[a_start:a_end].strip()
                                    shot = Shot(_input=q, _label=a)
                                    shots.append(shot)
                                # inference each task input
                                for j, task_input in enumerate(task_inputs):
                                    print(f"Predicting batch #{i}-{j} (cot: {self.cot_check if self._config.use_cot else self.cot_cross}) -> ", end='')
                                    prompt = StreamPrompt(
                                        task_desc=task.task_desc,
                                        inputs=task_input,
                                        num_demos=self._config.num_demos,
                                        shots=shots
                                    )
                                    pred_prompt = prompt.gen_prediction(cot=self._config.use_cot, add_parenthesis=add_parenthesis)
                                    pred_prompt, res_text = self._model.complete(pred_prompt, label_set, temperature=self._config.temperature)
                                    print(res_text)
                                    full_text = pred_prompt + res_text
                                    (task_log_path / "full-outputs" / f"{i * self._config.batch_size + j}.txt").write_text(full_text)
                                # continue to next run (i.e., next batch)
                                continue
                            else:
                                shots = raw_shots
                                
                    # augment the original prompt with self-icl exemplars
                    if self._config.inference_mode == "stream":
                        # update prompt to augmented prompt
                        prompt = StreamPrompt(
                            task_desc=task.task_desc,
                            inputs=task_inputs,
                            num_demos=self._config.num_demos,
                            shots=shots
                        )
                    else: # batch
                        prompt = BatchPrompt(
                            task_desc=task.task_desc, # shots already contain task description in current implementation
                            inputs=task_inputs,
                            num_demos=self._config.num_demos,
                            shots=shots
                        )
                    
                # run inference
                print(f"Predicting sample #{i} (cot: {self.cot_check if self._config.use_cot else self.cot_cross}) ->", end='')
                pred_prompt = prompt.gen_prediction(cot=self._config.use_cot, add_parenthesis=add_parenthesis)
                pred_prompt, res_text = self._model.complete(pred_prompt, label_set, temperature=self._config.temperature)
                print(res_text)
                # save results
                full_text = pred_prompt + res_text
                if self._config.exemplars_mode == "standard":
                    (task_log_path / f"{i}.txt").write_text(full_text)
                else: # self-icl
                    (task_log_path / "full-outputs" / f"{i}.txt").write_text(full_text)
        
        # save failed cases
        if len(failed_cases) > 0:
            print(f"Saving {len(failed_cases)} failed cases...")
            (self._log_path / "failed_cases.json").write_text(json.dumps(failed_cases, indent=4))
                    
    def evaluate(
        self,
        label_type: str = None,
        weighted_acc: bool = False,
        step3_stream: bool = False
    ) -> None:
        # for generating task labels
        task_gen = TaskGenerator(
            task_input_path=self._config.task_input_path,
            task_desc_path=self._config.task_desc_path,
            batch_size=1, # evaluate one by one during evaluation
            verbose=True
        )
        # start evaluation
        total_correct = 0
        total_predict = 0
        eval_results = dict()
        per_instance = dict() # store per-instance results (0: incorrect, 1: correct) -> for calculating significance
        for task_name in task_gen.task2desc.keys():
            task_label_type = TaskGenerator.task2label_type[task_name]
            if label_type and (task_label_type != label_type):
                print(f"Skipping task {task_name} with label_type {task_label_type}...")
                continue
            task = task_gen.get_task(task_name)
            task_log_path = self._log_path / task_name
            
            ncorrect = 0
            npredict = 0
            per_instance[task_name] = list()
            if type(self._config.test_sample_size) == int:
                num_runs = self._config.test_sample_size
            elif self._config.test_sample_size == "full":
                num_runs = task.sample_size # // self._config.batch_size) * self._config.batch_size
            for i in range(num_runs):
                if i < task.sample_size: # ensure i is within the sample size
                    # read inference result
                    filename = f"{i // (1 if step3_stream else self._config.batch_size)}.txt"
                    if self._config.exemplars_mode == "standard":
                        full_res = (task_log_path / filename).read_text()
                    else: # self-icl
                        full_res = (task_log_path / "full-outputs" / filename).read_text()
                    # parse inference result
                    label = task.get_new_labels().strip("()").upper()
                    if (self._config.inference_mode == "stream") or step3_stream:
                        pred = self._prompt_parser.extract_pred(full_res, use_cot=self._config.use_cot).strip("()").upper()
                    else: # batch
                        pred = self._prompt_parser.extract_pred_batch(full_res, answer_index=(i % self._config.batch_size) + 1 + self._config.num_demos).strip("()").upper()
                    print(f"Sample #{i}: label = {label}, pred = {pred} -> ", end='')
                    if label == pred:
                        print(Fore.GREEN + "✔")
                        ncorrect += 1
                        per_instance[task_name].append(1)
                    else:
                        print(Fore.RED + "✘")
                        per_instance[task_name].append(0)
                    npredict += 1
                    print(Style.RESET_ALL, end='')
            
            eval_results[task_name] = {
                "ncorrect": ncorrect,
                "total": npredict,
                "accuracy": ncorrect / npredict
            }
            print(f"Correct count: {Fore.BLUE}{ncorrect}/{npredict}{Style.RESET_ALL}")
            total_correct += ncorrect
            total_predict += npredict
        
        if weighted_acc:
            acc = 0
            for res in eval_results.values():
                acc += res["accuracy"]
            acc = acc / len(eval_results)
            print(f"{self._config.exp_name} -> #Tasks: {len(eval_results)}; Weighted Accuracy: {Fore.BLUE}{acc * 100:.2f}%{Style.RESET_ALL}");
        else:
            acc = total_correct / total_predict
            print(f"{self._config.exp_name} -> Total correct count: {Fore.BLUE}{total_correct}/{total_predict}{Style.RESET_ALL}; Accuracy: {Fore.BLUE}{acc * 100:.2f}%{Style.RESET_ALL}")
        # save evaluation results
        (self._log_path / f"eval_results_{self._config.test_sample_size}-testsize.txt").write_text(f"Accuracy = {(acc * 100):.2f}%\n")
        (self._log_path / f"per_instance_{self._config.test_sample_size}-testsize.json").write_text(json.dumps(per_instance, indent=4))
        df = pd.DataFrame(eval_results)
        df.to_csv(self._log_path / f"eval_results_{self._config.test_sample_size}-testsize.csv", index_label="Item")

    def estimate_cost(
        self,
        label_type: str = None,
        step2_stream: bool = False,
    ) -> None:
        """
        Format of num_tokens:
        {
            [task_name]: {
                "step1": [num_tokens_1, num_tokens_2, ...], # num tokens for each sample
                "step2": [num_tokens_1, num_tokens_2, ...],
                "step3": [num_tokens_1, num_tokens_2, ...],
                "total": [num_tokens_1, num_tokens_2, ...],
                "cost": [cost_1, cost_2, ...] # cost for each samples
                "total_cost": sum(cost)
            },
            [task_name]: ...
        }
        """
        task_gen = TaskGenerator(
            task_input_path=self._config.task_input_path,
            task_desc_path=self._config.task_desc_path,
            batch_size=1,
            verbose=False
        )
        num_tokens = dict()
        all_task_total_cost = 0
        for task_name in task_gen.task2desc.keys():
            task_label_type = TaskGenerator.task2label_type[task_name]
            if label_type and (task_label_type != label_type):
                print(f"Skipping task {task_name} with label_type {task_label_type}...")
                continue
            num_tokens[task_name] = {
                "step1": list(),
                "step2": list(),
                "step3": list(),
                "total": list(),
                "cost": list(),
                "total_cost": 0
            }
            task = task_gen.get_task(task_name)
            task_log_path = self._log_path / task_name
            # variables used later
            task_instruction = f"Task description: {task.task_desc}\n\n"
            for i in range(task.sample_size):
                # step 1 & 2
                step1_tokens, step2_tokens = 0, 0
                if self._config.exemplars_mode == "self-icl":
                    # step 1
                    demo_inputs_path = task_log_path / "demo-inputs"
                    if os.listdir(demo_inputs_path):
                        if (self._config.inference_mode == "stream"):
                            text = (demo_inputs_path / f"{i}.txt").read_text()
                            step1_tokens = self._model.count_tokens(text)
                        else:
                            raise NotImplementedError
                    # step 2
                    demo_labels_path = task_log_path / "demo-labels"
                    if os.listdir(demo_labels_path):
                        if (self._config.inference_mode == "stream"):
                            step2_tokens += self._model.count_tokens(task_instruction) * self._config.num_demos
                            for j in range(self._config.num_demos):
                                text = (demo_labels_path / f"{i}-{j}.txt").read_text()
                                step2_tokens += self._model.count_tokens(text)
                        elif step2_stream:
                            raise NotImplementedError
                        else: # batch
                            raise NotImplementedError
                num_tokens[task_name]["step1"].append(step1_tokens)
                num_tokens[task_name]["step2"].append(step2_tokens)
                # step 3
                if self._config.exemplars_mode == "self-icl":
                    text = (task_log_path / "full-outputs" / f"{i}.txt").read_text()
                else:
                    text = (task_log_path / f"{i}.txt").read_text()
                step3_tokens = self._model.count_tokens(text)
                num_tokens[task_name]["step3"].append(step3_tokens)
                # summary
                total_tokens = step1_tokens + step2_tokens + step3_tokens
                num_tokens[task_name]["total"].append(total_tokens)
                cost = total_tokens / 1000 * self._model.cost_per_1000tokens
                num_tokens[task_name]["cost"].append(cost)
                num_tokens[task_name]["total_cost"] += cost
            task_total_cost = num_tokens[task_name]['total_cost']
            all_task_total_cost += task_total_cost
            print(f"Task {task_name} -> Total cost: {Fore.GREEN}{task_total_cost:.2f} USD {Fore.RED}(~{task_total_cost * 30:.2f} TWD){Style.RESET_ALL}")
        print(f"\n ===== Total cost: {Fore.GREEN}{all_task_total_cost:.2f} USD {Fore.RED}(~{all_task_total_cost * 30:.2f} TWD){Style.RESET_ALL} =====")
        # save results
        (self._log_path / "num_tokens.json").write_text(json.dumps(num_tokens, indent=4))
        (self._log_path / "total_cost.txt").write_text(f"{all_task_total_cost:.2f} USD (~{all_task_total_cost * 30:.2f} TWD)")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--lacked_cases_path", type=Path, default=None)
    parser.add_argument("--task_continue_from", type=str, default=None)
    parser.add_argument("--sample_start_from", type=int, default=0)
    parser.add_argument("--label_type", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--estimate_cost", action="store_true")
    parser.add_argument("--weighted_acc", action="store_true")
    parser.add_argument("--existing_demos_path", type=Path, default=None)
    parser.add_argument("--random_pseudo_label", action="store_true")
    parser.add_argument("--rerun_pseudo_label", action="store_true")
    parser.add_argument("--step2_stream", action="store_true")
    parser.add_argument("--step3_stream", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = Config(**yaml.safe_load(args.config_path.read_text()))
    experiment = Experiment(config)
    
    if args.eval:
        experiment.evaluate(
            label_type=args.label_type,
            weighted_acc=args.weighted_acc,
            step3_stream=args.step3_stream
        )
    elif args.estimate_cost:
        experiment.estimate_cost(
            label_type=args.label_type,
            step2_stream=args.step2_stream
        )
    else:
        experiment.run(
            task_continue_from=args.task_continue_from,
            sample_start_from=args.sample_start_from,
            label_type=args.label_type,
            lacked_cases_path=args.lacked_cases_path,
            existing_demos_path=args.existing_demos_path,
            random_pseudo_label=args.random_pseudo_label,
            rerun_pseudo_label=args.rerun_pseudo_label,
            step2_stream=args.step2_stream,
            step3_stream=args.step3_stream
        )
