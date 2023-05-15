import abc
import unittest
from typing import List, Union
from dataclasses import dataclass

@dataclass
class Shot(object):
    _input: str
    _label: str
    
    @property
    def input(self) -> str:
        return self._input
    
    @property
    def label(self) -> str:
        return self._label
    
    def __repr__(self) -> str:
        return f"Q: {self._input}\nA: {self._label}"

# Currently only implement "stream" prompts
class Prompt(metaclass=abc.ABCMeta):
    
    def __init__(
        self,
        task_desc: str,
        inputs: Union[str, List[str]],
        num_demos: int,
        shots: List[Shot] = []
    ) -> None:
        self._task_desc = task_desc
        self._inputs = inputs
        self._num_demos = num_demos
        self._shots = shots
    
    @abc.abstractmethod  
    def gen_prediction(self, cot: bool = False) -> str:
        return NotImplemented

    @abc.abstractmethod
    def gen_demo_inputs(self, diversity: bool = False) -> str:
        return NotImplemented
    
class StreamPrompt(Prompt):

    def __init__(
        self,
        task_desc: str,
        inputs: str,
        num_demos: int,
        shots: List[Shot] = []
    ) -> None:
        super().__init__(task_desc, inputs, num_demos, shots)
    
    # TODO: Chain-of-thought (cot)
    def gen_prediction(self, cot: bool = False) -> str:
        """
        ### Prompting format:
        Task description: [task description].

        Q: [pseudo-demo-input 1]
        A: [pseudo-demo-label 1]

        ...

        Q: [pseudo-demo-input n]
        A: [pseudo-demo-label n]

        Q:
        """
        # task description
        prompt = [f"Task description: {self._task_desc}\n\n"]
        # in-context examples
        for shot in self._shots:
            prompt.append(f"Q: {shot.input}\n")
            prompt.append(f"A: {shot.label}\n\n")
        # current input
        prompt.append(f"Q: {self._inputs}\n")
        prompt.append(f"A:")
        return "".join(prompt)
    
    def gen_demo_inputs(self, diversity: bool = False) -> str:
        """
        ### Prompting format:
        Following is an example instance for the task: [task description]. Please come up with [num_shot] new[diverse_prompt] instances for the task.
        Example instance:
        [test input]

        New instance 1:
        """
        diverse_prompt = " and diverse" if diversity else ""
        return f"Following is an example instance for the task: {self._task_desc} Please come up with {self._num_demos} new{diverse_prompt} instances for the task.\nExample instance:\n{self._inputs}\n\nNew instance 1:"

class BatchPrompt(Prompt):
    
    def __init__(
        self,
        task_desc: str,
        inputs: List[str],
        num_demos: int,
        shots: List[Shot] = []
    ) -> None:
        super().__init__(task_desc, inputs, num_demos, shots)

class TestStreamPrompt(unittest.TestCase):

    def setUp(self):
        self.task_desc = "Evaluate the result of a random Boolean expression."
        self.inputs = "not ( True ) and ( True ) is"
        self.num_demos = 3
        self.zero_shots = []
        self.few_shots = [
            Shot("True and not not ( not False ) is", "True"),
            Shot("not True or False or ( False ) is", "False"),
            Shot("False or not ( True ) and False is", "False")
        ]
        
        self.zs_prompt = StreamPrompt(self.task_desc, self.inputs, self.num_demos, self.zero_shots)
        self.fs_prompt = StreamPrompt(self.task_desc, self.inputs, self.num_demos, self.few_shots)
    
    def test_gen_prediction(self):
        self.assertEqual(
            self.zs_prompt.gen_prediction(),
            "Task description: Evaluate the result of a random Boolean expression.\n\nQ: not ( True ) and ( True ) is\nA:"
        )
        self.assertEqual(
            self.fs_prompt.gen_prediction(),
            "Task description: Evaluate the result of a random Boolean expression.\n\nQ: True and not not ( not False ) is\nA: True\n\nQ: not True or False or ( False ) is\nA: False\n\nQ: False or not ( True ) and False is\nA: False\n\nQ: not ( True ) and ( True ) is\nA:"
        )
    
    def test_gen_demo_inputs(self):
        self.assertEqual(
            self.zs_prompt.gen_demo_inputs(),
            "Following is an example instance for the task: Evaluate the result of a random Boolean expression. Please come up with 3 new instances for the task.\nExample instance:\nnot ( True ) and ( True ) is\n\nNew instance 1:"
        )
        self.assertEqual(
            self.fs_prompt.gen_demo_inputs(),
            "Following is an example instance for the task: Evaluate the result of a random Boolean expression. Please come up with 3 new instances for the task.\nExample instance:\nnot ( True ) and ( True ) is\n\nNew instance 1:"
        )


# for running unit tests
if __name__ == "__main__":
    unittest.main()