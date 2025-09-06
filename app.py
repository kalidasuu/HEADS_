import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import LongformerModel, LongformerTokenizer
from sentence_transformers import SentenceTransformer, util
import os
import nltk
from nltk.tokenize import sent_tokenize
import hashlib
import jsonlines
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from rouge_score import rouge_scorer
import bert_score
from tqdm import tqdm
import spacy
from typing import Dict, Any

# Ensure necessary NLTK and SpaCy models are downloaded
@st.cache_resource
def download_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
download_nltk_punkt()

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.info("Downloading 'en_core_web_sm' model. This will happen only once.")
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")
nlp = load_spacy_model()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Models and Tokenizers with Caching ---
@st.cache_resource
def load_longformer_model(checkpoint_path: str):
    class LongformerExtractiveSummarizationModel(nn.Module):
        def __init__(self, pos_weight=None):
            super(LongformerExtractiveSummarizationModel, self).__init__()
            self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(self.longformer.config.hidden_size, 1)
            self.pos_weight = pos_weight if pos_weight is not None else torch.tensor(1.0)
        def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, labels=None):
            outputs = self.longformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            sequence_output = outputs.last_hidden_state
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)
            logits = logits.squeeze(-1)
            loss = None
            if labels is not None:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
                loss = loss_fct(logits, labels.float())
            return (loss, logits) if loss is not None else logits
    model = LongformerExtractiveSummarizationModel()
    model_file = None
    if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
        model_file = "pytorch_model.bin"
    elif os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
        model_file = "model.safetensors"
   
    if not model_file:
        st.error(f"No valid model file found in {checkpoint_path}. Expected 'pytorch_model.bin' or 'model.safetensors'.")
        st.stop()
   
    if model_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(checkpoint_path, model_file))
    else:
        state_dict = torch.load(os.path.join(checkpoint_path, model_file), map_location=torch.device('cpu'))
   
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

@st.cache_resource
def load_all_models():
    CHECKPOINT_PATH = "./HEADS/Longformer"
    tokenizer_long = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    sent_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    model_bart.to(device)
    longformer_model = load_longformer_model(CHECKPOINT_PATH)
   
    return tokenizer_long, sent_model, tokenizer_bart, model_bart, longformer_model

tokenizer_long, sent_model, tokenizer_bart, model_bart, longformer_model = load_all_models()

# ==============================================================================
# 1. Helper Functions
# ==============================================================================
def split_sentences(text: str) -> list[str]:
    """Splits text into sentences using SpaCy."""
    return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

def create_document_chunks(document: str, tokenizer, max_length: int = 512, overlap: int = 50) -> list[str]:
    """Creates overlapping document chunks for factual consistency check."""
    tokens = tokenizer.tokenize(document)
    chunks = []
    doc_max_length = max_length - 50
    start = 0
    while start < len(tokens):
        end = min(start + doc_max_length, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))
        if end == len(tokens):
            break
        start += (doc_max_length - overlap)
    return chunks

def calculate_text_stats(text: str) -> tuple[int, int]:
    """Calculates the number of sentences and words in the input text."""
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    # Split on whitespace and filter out empty strings or punctuation-only tokens
    words = [word for word in text.split() if word.strip()]
    num_words = len(words)
    return num_sentences, num_words

def calculate_default_sentences(num_sentences_total: int) -> int:
    """Calculates the default number of sentences based on extraction ratio."""
    if num_sentences_total < 100:  # Short input (~1 page or less)
        extraction_ratio = 0.3  # 30%
    elif 50 <= num_sentences_total < 200:  # Medium input
        extraction_ratio = 0.2  # 20%
    else:  # Long input (20+ pages, ~300+ sentences)
        extraction_ratio = 0.1  # 10%
    return max(1, int(num_sentences_total * extraction_ratio))

# ==============================================================================
# 2. Summary Generation Functions
# ==============================================================================
def generate_extractive_summary(document: str, progress_bar, num_sentences: int = None) -> str:
    """
    Generates an extractive summary using a fine-tuned Longformer model.
    Includes progress bar updates.
    Optionally takes a user-specified number of sentences.
    """
    if not document or not document.strip():
        return "Input document is empty."
   
    paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
    if not paragraphs:
        document_sentences = sent_tokenize(document)
    else:
        document_sentences = [sent for para in paragraphs for sent in sent_tokenize(para)]
   
    if not document_sentences:
        return "No sentences found in the document."
   
    num_sentences_total = len(document_sentences)
    
    # Determine top_k based on user input or default logic
    if num_sentences is not None:
        top_k = num_sentences
    else:
        top_k = calculate_default_sentences(num_sentences_total)
   
    progress_bar.progress(5, text="5% - Stage 1: Encoding sentences...")
    sentence_embeddings = sent_model.encode(document_sentences, convert_to_numpy=True)
   
    all_chunks_tokens = []
    all_chunks_attention = []
    all_chunks_global_attention = []
   
    current_input_ids = [tokenizer_long.cls_token_id]
    current_attention_mask = [1]
   
    CHUNK_SIZE = 4096
   
    # Simple chunking for processing
    flat_sentences = sent_tokenize(document)
   
    progress_bar.progress(15, text="15% - Stage 1: Creating model chunks...")
    for sent in flat_sentences:
        sentence_tokens = tokenizer_long.encode(sent, add_special_tokens=False)
        if len(current_input_ids) + len(sentence_tokens) + 1 > CHUNK_SIZE:
            padding_length = CHUNK_SIZE - len(current_input_ids)
            current_input_ids += [tokenizer_long.pad_token_id] * padding_length
            current_attention_mask += [0] * padding_length
            global_attention_mask = [0] * CHUNK_SIZE
            global_attention_mask[0] = 1
            all_chunks_tokens.append(current_input_ids)
            all_chunks_attention.append(current_attention_mask)
            all_chunks_global_attention.append(global_attention_mask)
           
            current_input_ids = [tokenizer_long.cls_token_id]
            current_attention_mask = [1]
       
        current_input_ids.extend(sentence_tokens)
        current_attention_mask.extend([1] * len(sentence_tokens))
   
    if len(current_input_ids) > 1:
        current_input_ids.append(tokenizer_long.sep_token_id)
        current_attention_mask.append(1)
        padding_length = CHUNK_SIZE - len(current_input_ids)
        current_input_ids += [tokenizer_long.pad_token_id] * padding_length
        current_attention_mask += [0] * padding_length
        global_attention_mask = [0] * CHUNK_SIZE
        global_attention_mask[0] = 1
        all_chunks_tokens.append(current_input_ids)
        all_chunks_attention.append(current_attention_mask)
        all_chunks_global_attention.append(global_attention_mask)
    if not all_chunks_tokens:
        return ""
   
    progress_bar.progress(25, text="25%")
    input_ids_tensor = torch.tensor(all_chunks_tokens).to(device)
    attention_mask_tensor = torch.tensor(all_chunks_attention).to(device)
    global_attention_mask_tensor = torch.tensor(all_chunks_global_attention).to(device)
   
    all_logits = []
    for i in range(len(all_chunks_tokens)):
        with torch.no_grad():
            logits = longformer_model(
                input_ids=input_ids_tensor[i:i+1],
                attention_mask=attention_mask_tensor[i:i+1],
                global_attention_mask=global_attention_mask_tensor[i:i+1]
            )
            all_logits.append(logits.squeeze(0).cpu().numpy())
           
    aggregated_scores = []
    current_token_idx = 0
    for chunk, att_mask in zip(all_logits, all_chunks_attention):
        effective_len = sum(att_mask)
        if effective_len > 2:
            chunk_scores = torch.sigmoid(torch.tensor(chunk[1:effective_len-1]))
            aggregated_scores.extend(chunk_scores.tolist())
    sentence_scores = [0.0] * len(document_sentences)
    current_token_idx = 0
    for i, sentence in enumerate(document_sentences):
        sentence_tokens = tokenizer_long.encode(sentence, add_special_tokens=False)
        end_token_idx = current_token_idx + len(sentence_tokens)
        if end_token_idx <= len(aggregated_scores):
            sentence_logits = aggregated_scores[current_token_idx:end_token_idx]
            sentence_scores[i] = max(sentence_logits) if sentence_logits else 0.0
            current_token_idx = end_token_idx
        else:
            # Handle edge cases where token counts don't perfectly align
            sentence_scores[i] = 0.0
   
    selected_indices = np.argsort(sentence_scores)[-top_k:][::-1]
    predicted_sentences = [document_sentences[i] for i in sorted(selected_indices)]
   
    progress_bar.progress(35, text="50%")
    return " ".join(predicted_sentences)

def rephrase_text(input_text: str) -> str:
    """Rephrases text using a BART model with a fixed sentence ratio."""
    sentences = split_sentences(input_text)
    input_count = len(sentences)
    if input_count == 0:
        return input_text
   
    target_count = max(1, int(input_count * 0.85))
    rephrased_sentences = []
    for sentence in sentences[:target_count]:
        inputs = tokenizer_bart(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        # Dynamic length based on input sentence length
        max_len = max(len(inputs.input_ids[0]) + 5, 20)
        outputs = model_bart.generate(
            **inputs,
            max_length=max_len,
            min_length=5,
            num_beams=6,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        rephrased_sentence = tokenizer_bart.decode(outputs[0], skip_special_tokens=True)
        rephrased_sentences.append(rephrased_sentence)
   
    return " ".join(rephrased_sentences)

def process_summary(summary: str, progress_bar) -> str:
    """
    Processes the extractive summary into an abstractive one in chunks.
    Includes progress bar updates.
    """
    sentences = split_sentences(summary)
    chunks = []
    current_chunk = []
    current_length = 0
   
    CHUNK_SIZE = 1024  # A more appropriate chunk size for BART
   
    for sentence in sentences:
        sentence_tokens = tokenizer_bart.encode(sentence, add_special_tokens=False)
        if current_length + len(sentence_tokens) > CHUNK_SIZE:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence_tokens)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence_tokens)
   
    if current_chunk:
        chunks.append(" ".join(current_chunk))
   
    final_summary_parts = []
    # Start abstractive progress from 40% up to 90%
    progress_start = 40
    progress_range = 50
    progress_step_size = progress_range / len(chunks) if chunks else 0
   
    for i, chunk in enumerate(chunks):
        current_progress = progress_start + int(progress_step_size * i)
        progress_bar.progress(current_progress)
        final_summary_parts.append(rephrase_text(chunk))
       
    final_summary = " ".join(final_summary_parts)
    progress_bar.progress(95, text="95% - Finalizing summary...")
    return final_summary

# ==============================================================================
# 3. Streamlit App Layout
# ==============================================================================
# Custom CSS for a professional look with a subtle gradient background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right bottom, #ece9e6, #ffffff);
        color: #333;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .main-header {
        color: #2c3e50; /* Darker blue-grey for main header */
        font-size: 2.8em;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5em;
        padding-top: 20px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        color: #34495e; /* Slightly lighter blue-grey for sub-headers */
        font-size: 1.6em;
        font-weight: 600;
        border-bottom: 2px solid #bdc3c7; /* Light grey border */
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #3498db; /* Blue button */
        color: white;
        font-size: 1.2em;
        padding: 12px 25px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2980b9; /* Darker blue on hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
    }
    .stTextArea label {
        font-size: 1.2em;
        font-weight: bold;
        color: #2c3e50;
    }
    .stSelectbox label, .stNumberInput label {
        font-size: 1.2em;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-container {
        padding: 20px;
        border-radius: 12px;
        background-color: #ecf0f1; /* Light grey for metric background */
        margin-top: 25px;
        border: 1px solid #dfe6e9;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-item {
        margin-bottom: 10px;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #f0f2f6;
    }
    .stProgress .st-ds {
        background-color: #2ecc71; /* Green for progress bar */
    }
    .stProgress .st-da {
        background-color: #dcdcdc; /* Lighter grey for progress bar track */
    }
    .stMarkdown h2.sub-header {
        margin-top: 40px;
    }
    .stats-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header with emoji
st.markdown("<h1 class='main-header'>HEADS: Hybrid Extractive Abstractive Document Summarizer ✍️</h1>", unsafe_allow_html=True)

# Main app description

# Initialize session state for text input
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# User input
with st.container():
    st.markdown("<div style='padding-top: 30px;'></div>", unsafe_allow_html=True)
    user_input = st.text_area("Input Document", height=350, placeholder="Paste your article, report, or text here...", key="user_input")
    
    # Calculate and display text statistics
    if user_input and user_input.strip():
        num_sentences, num_words = calculate_text_stats(user_input)
        st.markdown(f"""
        <div class='stats-container'>
            <strong>Input Statistics:</strong> {num_sentences} sentences, {num_words} words
        </div>
        """, unsafe_allow_html=True)
        max_sentences = num_sentences
        default_sentences = calculate_default_sentences(num_sentences)
    else:
        max_sentences = 100  # Default max for number input when no text is provided
        default_sentences = 1  # Default value when no text is provided
        st.markdown(f"""
        <div class='stats-container'>
            <strong>Input Statistics:</strong> 0 sentences, 0 words
        </div>
        """, unsafe_allow_html=True)
    
    # Dropdown for selecting summary sentence option
    sentence_option = st.selectbox(
        "Number of Sentences in Summary",
        ["Let the model decide", "Custom number"],
        help="Choose 'Let the model decide' to use automatic selection (30% for short, 20% for medium, 10% for long documents) or 'Custom number' to specify the exact number of sentences."
    )
    
    # Show number input only if "Custom number" is selected
    num_sentences = None
    if sentence_option == "Custom number":
        num_sentences = st.number_input(
            "Custom Number of Sentences",
            min_value=1,
            #max_value=max_sentences,
            value=default_sentences,
            step=1,
            help=f"Specify the desired number of sentences in the summary (max {max_sentences}). Default is {default_sentences} based on automatic selection."
        )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_button = st.button("Generate Summary")

# --- Summary Generation and Display ---
if generate_button:
    if user_input:
        # Validate input length
        if len(user_input.strip()) < 50:
            st.warning("Please enter at least 50 characters to generate a meaningful summary.")
        else:
            # Validate custom sentence count
            num_sentences_total = len(sent_tokenize(user_input))
            if sentence_option == "Custom number" and num_sentences > num_sentences_total:
                st.error(f"Requested number of sentences ({num_sentences}) exceeds the number of sentences in the input ({num_sentences_total}). Please choose a number between 1 and {num_sentences_total}.")
            else:
                progress_text = "Starting summarization..."
                progress_bar = st.progress(0, text=progress_text)
                
                # Using st.empty() to update the spinner message without recreating it
                spinner_message = st.empty()
                spinner_message.info("Processing your document...")
                try:
                    # Stage 1: Extractive Summary
                    spinner_message.info("Stage 1: Extracting summary sentences...")
                    extractive_summary = generate_extractive_summary(user_input, progress_bar, num_sentences)
                    if extractive_summary in ["Input document is empty.", "No sentences found in the document.", ""]:
                        st.error(extractive_summary or "Failed to generate summary due to invalid input.")
                        progress_bar.empty()
                        st.stop()
                    
                    spinner_message.info("Stage 2: Rephrasing summary...")
                    
                    # Stage 2: Rephrase with BART
                    final_summary = process_summary(extractive_summary, progress_bar)
                    
                    progress_bar.progress(100, text="100% - Summary complete! 🎉")
                    spinner_message.success("Summary generation successful!")
                    # Display the final summary
                    st.markdown("---")
                    st.markdown("<h2 class='sub-header'>Generated Summary ✨</h2>", unsafe_allow_html=True)
                    st.markdown(f"<div style='background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>{final_summary}</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    spinner_message.error(f"An error occurred during summarization: {e}")
                    progress_bar.empty()
    else:
        st.warning("Please enter some text to summarize.")