# GPT2-Entropy-Surprisal

Uses logits from GPT-2 to compute word-by-word entropy, surprisal, and cosine similarity from text via a Gradio interface.

Authors: Nikki Fackler Kaye & Zachary Gordon (zgordo on Github)

## Files

- `app.py` — Gradio interface
- `library.py` — GPT-2 computation logic
- `requirements.txt` — minimal dependencies
- `requirements-full.txt` — full environment (for reproducibility)

## Input format

Provide a **tab-delimited text file** with at least the following colums:

- `code` identifier (e.g., text id)
- `text` string of text

## Run

Install dependencies
Run python app.py
Navigate to local URL in browser
Upload input file
Specify settings
- **Strip Punctuation**: if enabled, punctuation is removed from the target word before computation.
- Output file path
- GPT-2 model
Click **Run**

## Outputs

The output file contains:

- `Word_ID` - text + word position
- `Text_ID` - index of the input text (row number)
- `word_location` - position of word in text
- `target` - observed word
- `predicted_word` - model's top-predicted next token
- `entropy` - Shannon entropy of the next-token distribution
- `surprisal` - negative log probability of the target word, summed across GPT-2 subtokens for multiple token words
- `similarity` - cosine similarity between the embedding of predicted_word and the embedding of the first token of the target word
- `actual_word_decoded` - reconstructed word follwing tokenization
- `num_tokens` - number of tokens in target word
- `text_history` - context preceding target word
- `code` - identifier copied from input



