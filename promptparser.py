import re
from pathlib import Path

class PromptParser(object):
    
    def __init__(self, num_demos: int):
        self._num_demos = num_demos
        
    def split_demo_inputs(self, full_demo_inputs: str):
        sep_demo_inputs = re.findall(pattern=r"tance \d:\nQ:\s(.*?)(?:\n*New ins|\Z)", string=full_demo_inputs, flags=re.DOTALL)
        if len(sep_demo_inputs) != self._num_demos:
            raise ValueError(f"Number of demos in full_demo_inputs ({len(sep_demo_inputs)}) does not match num_demos ({self._num_demos})")
        return sep_demo_inputs
    
    def extract_pred(self, full_res: str):
        pred = re.findall(pattern=r"A:\W*(\w+?)\W*\Z", string=full_res)
        if len(pred) != 1:
            raise ValueError(f"Number of predictions in full_res ({len(pred)}) is not 1")
        return pred[0]
    
# unit tests
if __name__ == "__main__":
    num_demos = 3
    prompt_parser = PromptParser(num_demos)
    
    full_demo_inputs = Path("./log/model_outputs/stream/api/self-icl-3-demos/boolean_expressions/demo-inputs/0.txt").read_text()
    print(full_demo_inputs)
    demo_inputs = prompt_parser.split_demo_inputs(full_demo_inputs)
    print(demo_inputs)