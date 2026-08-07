# benchmark_local_llm.py

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_accuracy(predictions, references):
    """
    Compute the accuracy of the predictions against the references.
    """
    correct = 0
    total = len(predictions)
    for pred, ref in zip(predictions, references):
        if pred == ref:
            correct += 1
    return correct / total

def run_benchmark(model_name, dataset_path):
    """
    Run the benchmarking process for the given model and dataset.
    """
    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load dataset
        dataset = load_dataset('json', data_files=dataset_path)
        inputs = dataset['train']['input']
        references = dataset['train']['reference']

        # Tokenize inputs
        inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(**inputs)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Compute accuracy
        accuracy = compute_accuracy(predictions, references)
        logger.info(f"Accuracy: {accuracy:.4f}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    model_name = "Llama-3-8B"
    dataset_path = "path/to/dataset.json"
    run_benchmark(model_name, dataset_path)