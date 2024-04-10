import importlib
from pathlib import Path
from typing import Sequence

from transformers import PreTrainedTokenizer

def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)
    return module


def calculate_avg_tokens(text_list: Sequence[str], tokenizer: PreTrainedTokenizer, avg: bool) -> int:
    total_tokens = 0
    max_lens = []
    for text in text_list:
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)
        max_lens.append(len(tokens))

    avg_tokens = total_tokens / len(text_list)
    max_tokens = max(max_lens)
    return int(avg_tokens) if avg else max_tokens
