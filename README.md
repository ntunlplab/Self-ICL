# Self-ICL: Zero-Shot In-Context Learning with Self-Generated Demonstrations

## Data
### BIG-Bench Hard 
- [BBH data](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh)
- [BBH task description](https://github.com/nlplab-best-team/Self-ICL/tree/main/bbh)

## Experiment

### Model
- text-davinci-003

### Hyper-parameter
- ```max_tokens``` = 1024
- ```temperature``` = 0.0
- ```top_p``` = 1.0 or 0.0(?)

### Main Table 1
- Self-ICL w/ different demos (input, label)
  - (Self-ICL, Standard Zero-Shot)
  - (Self-ICL, Zero-Shot CoT)
  - (Self-ICL, Random)
- Zero-Shot Baseline
  - Zero-Shot
  - Zero-Shot CoT
- Other Baseline
  - Standard ICL: (Real Data, Golden)

* Classification tasks only (test sample size = 250)

|        | No-CoT    | CoT       |
|--------|-----------|-----------|
| Zero-Shot | 50.81% | 53.22% |
| Self-ICL | 53.93%\*\*| 55.54%\*|

\* p < 0.01; \*\* p < 0.001 (Self-ICL vs. Zero-Shot)

### Main Table 2
Stream vs Batch

### Ablation
- Number of demos: 0, 1, 2, 3
- Number of test input, i.e., batch size: 1, 2, 3

### Cost (in USD)
- Direct prompting
  - ZS-Direct: $15.27
  - Self-ICL (1-shot): $27.29
  - Self-ICL (3-shot): $118.35
  - Self-ICL (no-diverse): $135.58
  - Self-ICL (random): $51.27
  - Self-ICL (batch): $63.15
- CoT prompting:
  - ZS-CoT: $28.71
  - Self-ICL (3-shot): $203.10