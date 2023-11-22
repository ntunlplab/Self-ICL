# Self-ICL: Zero-Shot In-Context Learning with Self-Generated Demonstrations

This is the official repository of our paper [Self-ICL: Zero-Shot In-Context Learning with Self-Generated Demonstrations](https://arxiv.org/pdf/2305.15035.pdf), *EMNLP* 2023.

*TL;DR*: This work presents Self-ICL, a prompting framework bootstrapping LLMsʼ intrinsic task understanding ability to perform in-context learning via self-generated pseudo-demonstrations.

---

## The Self-ICL prompting framework

Given the ```[task_description]``` and a corresponding ```[test_input]```, Self-ICL consists of three steps:

1. Construction of pseudo-inputs.
   - [Prompt template](prompt/step-1.txt) (the same prompt is use in both direct prompting and chain-of-thought prompting).
   - Prompt the LLM to generate ```[num_shot]``` pseudo-inputs, conditioned on the ```[task_description]``` and ```[test_input]```.

3. Construction of pseudo-labels.
   - [Prompt template for direct prompting](prompt/direct/step-2.txt).
   - [Prompt template for chain-of-thought prompting](prompt/cot/step-2.txt).
   - Collect the ```[num_shot]``` pseudo-inputs generated in step 1 and predict their pseudo-labels by zero-shot prompting the LLM.

5. In-context learning with pseudo-demonstrations.
   - [Prompt template for direct prompting](prompt/direct/step-3.txt).
   - [Prompt template for chain-of-thought prompting](prompt/cot/step-3.txt).
   - Collect the pseudo-labels generated in step 2, and construct ```[num_shot]``` pseudo-demonstrations (*i.e.*, pseudo-input-label pairs).
   - Concatenate ```[task_description]```, pseudo-demonstrations, and ```[test_input]``` and prompt the LLM to perform ICL.
  
<details>
<summary>Reproduce paper experiments</summary>
  
*WIP, refactoring code...*

</details>
