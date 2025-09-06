
# constraint_bart.py (Updated to use Longformer Extractive Output)

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import spacy

# Load English NER model
nlp = spacy.load("en_core_web_sm")

# Load BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def extract_constraints_from_extractive(extractive_sentences, max_entities=5):
    """
    Extracts named entities from the extractive summary output (Longformer).
    extractive_sentences: list of strings (output from Longformer extractive stage)
    """
    combined_text = " ".join(extractive_sentences)
    doc = nlp(combined_text)
    entities = list(set([ent.text for ent in doc.ents]))
    return entities[:max_entities]

def generate_constrained_prompt(text, constraints):
    """
    Prepend constraints as a guided prompt to the input text.
    """
    if not constraints:
        return text
    constraint_str = "Summarize the document focusing on: " + ", ".join(constraints) + ". "
    return constraint_str + text

def summarize_with_extractive_constraints(original_text, extractive_sentences, 
                                           max_length=200, min_length=50, beam_size=4):
    """
    Generate summary using BART with constraints taken from Longformer extractive output.
    """
    constraints = extract_constraints_from_extractive(extractive_sentences)
    prompt_text = generate_constrained_prompt(original_text, constraints)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=beam_size,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary, constraints
