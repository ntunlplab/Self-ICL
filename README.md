# Self-ICL: Zero-Shot In-Context Learning with Self-Generated Demonstrations

This is the official repository of our paper [Self-ICL: Zero-Shot In-Context Learning with Self-Generated Demonstrations](https://arxiv.org/pdf/2305.15035.pdf), *EMNLP* 2023.

*TL;DR*: This work presents Self-ICL, a prompting framework bootstrapping LLMsʼ intrinsic task understanding ability to perform in-context learning via self-generated pseudo-demonstrations.

<p align="center">
   <img src="https://github.com/ntunlplab/Self-ICL/assets/106149032/a0fbf92e-63ca-4d3d-a7eb-83400221feab" width=75% height=75%>
</p>

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
  
## Steps to Reproduce Paper Experiments
One may follow the steps below to reproduce the experiments, using the `text-bison-001` column in Table 3 as an example.

### Setup API Keys
Set the corresponding environment variables to your API keys. For example, `GOOGLE_API_KEY` for PaLM-2, and `OPENAI_API_KEY` for GPT models.
```
export GOOGLE_API_KEY=<your_api_key>
```

### Configure the Experiment Settings
Set the configuration file in `./configs` to your experiment settings. For example, the `./configs/config_template_standard.yml` file records the settings for running the **ZS-Direct** prompting method using the `text-bison-001` API endpoint.

### Run the Prompting Script
Run the following script to run different prompting methods and log output to `log_path` in the YAML file:
```
python experiment.py \
  --config_path ./configs/config_template_standard.yml \
  --label_type "class"
```
Note that one may need to configure other command line arguments in other experiments (please refer to experiment.py for more details).

### Run the Evaluation Script
After finishing the previous step, you may evaluate the performance by running:
```
python experiment.py \
  --config_path ./configs/config_template_standard.yml \
  --label_type "class" \
  --eval
```
The evaluation results would be stored in the `log_path` along with model outputs from the previous step.
