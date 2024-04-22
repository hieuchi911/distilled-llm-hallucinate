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
You can run the python file [main.py](main.py) to evaluate distilled student models on the corresponding hallucination benchmarks with the following flags:

- `-t`     : The type of task that the student models are finetuned in: `summ`, `qa`
- `-d`     : The dataset used to evaluate the student models: `hotpot_qa` , `truthful_qa`, `cnn_dm`, or `qmsum`
- `-sub`   : The corresponding subset of the dataset: `'distractor'` for `hotpot_qa`, `'generation'` for `truthful_qa`, `'1.0.0'` for `cnn_dm`, or `'default'` for `qmsum`
- `-max`   : The maximum length of tokenized context
- `-ms`    : The path to the json file that specifies student model huggingface paths
- `-dm`    : The path to the file that contains the data mappers
- `-a`     : Whether to use average length of all answers in the dataset as the number of new tokens to generate - if `False` then max length of all answers in the dataset is used
- `-db`    : Whether or not to enable debug mode with toy dataset
- `-bs`    : Batch size for the data loader
- `-o`     : The path to the folder to store evaluation output

For example, to evaluate student models specialized in QA tasks using `hotpot_qa` benchmark, run:
```bash
python main.py -t qa -d hotpot_qa -sub distractor -ms ./models_qa.json -dm ./data_map.py -bs 20 -o ./eval_results
```