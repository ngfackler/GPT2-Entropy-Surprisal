from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import pandas as pd
import re

def cosine_similarity(vec1, vec2):
    return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))

def run_model(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits
    last_word_logits = predictions[0, -1, :]
    probs = F.softmax(last_word_logits, dim=-1)
    return probs

def get_entropy_and_surprisal(text, actual_next_word, model, tokenizer, code):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    next_word_token_ids = tokenizer.encode(actual_next_word, add_special_tokens=False)
    num_tokens = len(next_word_token_ids)
    probs = run_model(model, input_ids)
    #calculate surprisal
    combined_surprisal = -torch.log(probs[next_word_token_ids[0]])
    if num_tokens > 1:
        text_and_partial_next_word = text + ' '
        for i in range(num_tokens-1):
            partial_next_word = tokenizer.decode(next_word_token_ids[i])
            text_and_partial_next_word += partial_next_word
            text_and_partial_input_ids = tokenizer.encode(text_and_partial_next_word, return_tensors='pt')
            temp_token_probs = run_model(model, text_and_partial_input_ids)
            surprisal = -torch.log(temp_token_probs[next_word_token_ids[i+1]])
            combined_surprisal += surprisal
    #calculate entropy
    entropy = -torch.sum(probs * torch.log(probs))    
    
    actual_next_word_embedding = model.base_model.wte.weight[next_word_token_ids[0]]
    top_probs, top_indices = torch.topk(probs, 10)
    top_prediction_id = top_indices[0].item()
    top_prediction_embedding = model.base_model.wte.weight[top_prediction_id]

    #calculate similarity
    similarity = cosine_similarity(actual_next_word_embedding, top_prediction_embedding)

    return {
        'entropy': entropy.item(),
        'surprisal': combined_surprisal.item(),
        'similarity': similarity.item(),
        'predicted_word': tokenizer.decode([top_prediction_id]),
        'actual_next_word_test': tokenizer.decode(next_word_token_ids),
        'num_tokens' : num_tokens,
        'code' : code
    }

def run_program(input_path, model_name, strip_punct):
    texts = pd.read_csv(input_path, sep='\t')    

    #load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    rows = []

    for i in range(len(texts['Text'])):
        text = texts['Text'][i]
        words = text.split()

        for j in range(1, len(words)):
            text_history = ' '.join(words[0:j])

            if strip_punct:
                actual_word_no_punc = re.sub(r'[^\w\s]', '', words[j])
            else:
                actual_word_no_punc = words[j]

            if actual_word_no_punc == "":
                continue

            result = get_entropy_and_surprisal(
                text_history,
                ' ' + actual_word_no_punc,
                model,
                tokenizer,
                texts['coding'][i]
            )

            predicted = '' if result['predicted_word'] == '\n' else result['predicted_word']

            row_data = {
                'Word_ID': f'T{i+1}W{j}',
                'Text_ID': i+1,
                'word_location': j,
                'target': ' ' + actual_word_no_punc,
                'predicted_word': predicted,
                'entropy': result['entropy'],
                'surprisal': result['surprisal'],
                'similarity': result['similarity'],
                'actual_word_test': result['actual_next_word_test'],
                'num_tokens': result['num_tokens'],
                'text_history': text_history,
                'code': result['code'],
            }

            rows.append(row_data)

    return pd.DataFrame(rows)

        

