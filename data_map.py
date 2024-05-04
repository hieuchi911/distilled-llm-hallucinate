PROMPT_KEYWORDS = {
    "qa": "Answer:",
    "summ": "Summary:"
}

def add_prefix_truthful_qa(example):
    question = example["question"].strip()
    concatenated_input = f"Context: No context\nQuestion: {question}\nAnswer:"
    target_answer = example["best_answer"]

    return {"input_seq": concatenated_input, "label": target_answer}

def add_prefix_hotpot_qa(example):
    question = example["question"].strip()
    context_sentences = [sent for sent_group in example["context"]["sentences"] for sent in sent_group]
    context_titles = example["context"]["title"]

    context_str = " ".join([f"{sent}" for title, sent in zip(context_titles, context_sentences)])

    concatenated_input = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
    target_answer = example["answer"]

    return {"input_seq": concatenated_input, "label": target_answer}

def add_prefix_cnn_dm(example):
    context = example["article"].strip()
    concatenated_input = f"Dialogue: {context}\nSummary:"
    target_answer = example["highlights"]

    return {"input_seq": concatenated_input, "label": target_answer}

def add_prefix_qmsum(example):
    context = example["input"].strip()
    concatenated_input = f"Dialogue: {context}\nSummary:"
    target_answer = example["output"]

    return {"input_seq": concatenated_input, "label": target_answer}
