import os
import re
import json
from typing import List, Dict
from pathlib import Path

def raw_text_to_shots(text: str) -> List[Dict[str, str]]:
    num_shots = 3
    q_prefix = "Q: "
    a_prefix = "A: "
    cot_prompt = "Let's think step by step."
    ans_prompt = "So the answer is "
    Qs = re.findall(pattern=f"{q_prefix}(.*?){a_prefix + cot_prompt}", string=text, flags=re.DOTALL)
    As = re.findall(pattern=f"{ans_prompt}(.+)\\.", string=text)
    cots = re.findall(pattern=f"{a_prefix + cot_prompt}(.*?){ans_prompt}", string=text, flags=re.DOTALL)
    assert len(Qs) == len(As) == len(cots) == num_shots
    shots = []
    for Q, A, cot in zip(Qs, As, cots):
        shot = {
            "Q": Q.strip(),
            "A": A.strip(),
            "cot": cot.strip()
        }
        shots.append(shot)
    return shots

def raw_cot_prompts_to_parsed_prompts(cot_prompt_path: Path, save_path: Path) -> None:
    for filename in os.listdir(cot_prompt_path):
        text = (cot_prompt_path / filename).read_text()
        shots = raw_text_to_shots(text)
        (save_path / (filename.rstrip(".txt") + ".json")).write_text(json.dumps(shots, indent=4))

if __name__ == "__main__":
    cot_prompt_path = Path("./cot-prompts")
    save_path = Path("./cot-prompts-parsed")
    save_path.mkdir(exist_ok=True)
    raw_cot_prompts_to_parsed_prompts(cot_prompt_path, save_path)