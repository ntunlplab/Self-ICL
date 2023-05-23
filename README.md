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

* Classification tasks only (test sample size = 100)

|        | No-CoT    | CoT       |
|--------|-----------|-----------|
| Zero-shot | 50.39% | 52.48% |
| Self-ICL | 53.78% | 55.04% |

### Main Table 2
Stream vs Batch

### Ablation
- Number of demos: 0, 1, 2, 3
- Number of test input, i.e., batch size: 1, 2, 3
