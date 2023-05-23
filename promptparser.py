import re
from pathlib import Path
from argparse import ArgumentParser, Namespace

class PromptParser(object):
    
    def __init__(self, num_demos: int):
        self._num_demos = num_demos
        
    def split_demo_inputs(self, full_demo_inputs: str):
        sep_demo_inputs = re.findall(pattern=r"tance \d:\s+Q:\s*(.*?)(?:\n*New ins|\Z)", string=full_demo_inputs, flags=re.DOTALL)
        if len(sep_demo_inputs) != self._num_demos:
            raise ValueError(f"Number of demos in full_demo_inputs ({len(sep_demo_inputs)}) does not match num_demos ({self._num_demos})")
        return sep_demo_inputs
    
    def parse_pred(self, pred: str) -> str:
        return re.split(pattern=r"[,.;: ]", string=pred)[0]
    
    def extract_pred(self, full_res: str, use_cot: bool = False, use_parse_pred: bool = True):
        if use_cot:
            pred = re.findall(pattern=r".*(?:[Tt]herefore, the correct answer is:*|the correct answer is therefore|so the correct answer is)(.*)", string=full_res, flags=re.DOTALL)
            if len(pred) != 1:
                raise ValueError(f"Number of predictions in full_res ({len(pred)}) is not 1")
            else:
                pred = pred[0].strip()
            return self.parse_pred(pred) if use_parse_pred else pred
        else:
            pred = re.findall(pattern=r"A:\W*(\w+?)\W*\Z", string=full_res)
            if len(pred) != 1:
                raise ValueError(f"Number of predictions in full_res ({len(pred)}) is not 1")
            return pred[0]

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--split_demo_inputs", action="store_true")
    parser.add_argument("--extract_pred", action="store_true")
    parser.add_argument("--use_cot", action="store_true")
    return parser.parse_args()

# unit tests
if __name__ == "__main__":
    args = parse_args()
    
    num_demos = 3
    prompt_parser = PromptParser(num_demos)
    
    if args.split_demo_inputs:
        full_demo_inputs = Path("./log/model_outputs/stream/self-icl-no-cot-diverse-temp075-class/boolean_expressions/demo-inputs/0.txt").read_text()
        print(full_demo_inputs)
        demo_inputs = prompt_parser.split_demo_inputs(full_demo_inputs)
        print(demo_inputs)
    elif args.extract_pred:
        full_outputs = Path("./log/model_outputs/stream/standard-zero-shot-cot-class/boolean_expressions/6.txt").read_text()
        print(full_outputs)
        pred = prompt_parser.extract_pred(full_outputs, use_cot=args.use_cot)
        print(pred)