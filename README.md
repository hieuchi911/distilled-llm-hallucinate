# On the effect of logit-based Knowledge distillation on hallucination in LLMs
A hallucination evaluation kit on distilled transformers/ LLMs

## Setup
1. Clone the repository
```bash
git clone https://github.com/hieuchi911/distilled-llm-hallucinate.git
cd distilled-llm-hallucinate
```
2. Install the required packages
```bash
conda env create -f environment.yml
conda activate distilled-llm-hallucinate
```

## Run evaluation

### 1. Evaluate hallucination on the generative QA dataset
You can run the following command to evaluate distilled student models (specified in [models.json](models.json)) on the hallucination benchmarks `hotpot_qa` and `truthful_qa`:
```bash
-d "truthful_qa" -sub "generation" -ms "./models.json" -db -dm "./data_map.py" -bs 32 -o "./eval_results"
```
where:
- `-d`     :        The path to the dataset used to evaluate the student models: `hotpot_qa` or `truthful_qa`
- `-sub`   :      The corresponding subset of the dataset: `'distractor'` for `hotpot_qa` or `'generation'` for `truthful_qa`
- `-ms`    :       The path to the json file that specifies student model paths
- `-dm`    :       The path to file that contains the data mappers
- `-a`     :        Whether to use average length of all answers in the dataset as the number of new tokens generate - if `False` then max length of all answers in the dataset
- `-db`    :       Whether or not to enable debug mode with toy dataset
- `-bs`    :       Batch size for the data loader
- `-o`     :        The path to the folder to store evaluation output