import json

def align_tokens_to_text(tokens, text):
    spans = []
    current_pos = 0
    for token in tokens:
        pos = text.find(token, current_pos)
        if pos == -1:
            pos = text.lower().find(token.lower(), current_pos)
            
        if pos != -1:
            start_char = pos
            end_char = pos + len(token)
            spans.append((start_char, end_char))
            current_pos = end_char
        else:
            spans.append(None)
    return spans

def extend_span(text, start, end):
    # Extend end to include trailing '+' or '#' (like C++ or C#)
    while end < len(text) and text[end] in ('+', '#'):
        end += 1
    return start, end

def check_match(tokens, text):
    matches = 0
    current_pos = 0
    for token in tokens[:20]:
        pos = text.find(token, current_pos)
        if pos == -1:
            pos = text.lower().find(token.lower(), current_pos)
        if pos != -1:
            matches += 1
            current_pos = pos + len(token)
    return matches >= min(10, len(tokens[:20]) * 0.7)

def main():
    print("Loading data...")
    with open('for_train (1).json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    with open('train_dataset1.json', 'r', encoding='utf-8') as f:
        tok_data = json.load(f)
        
    print(f"Raw items: {len(raw_data)}, Tokenized items: {len(tok_data)}")
    
    results = []
    i_raw = 0
    i_tok = 0
    matched_count = 0
    
    while i_raw < len(raw_data) and i_tok < len(tok_data):
        raw_item = raw_data[i_raw]
        tok_item = tok_data[i_tok]
        
        if check_match(tok_item['tokenized_text'], raw_item['text']):
            tokens = tok_item['tokenized_text']
            ner = tok_item['ner']
            text = raw_item['text']
            
            token_spans = align_tokens_to_text(tokens, text)
            
            entities = []
            seen_spans = {} # to track occurrences of the same token span
            
            for start_tok_idx, end_tok_idx, label in ner:
                if start_tok_idx >= len(token_spans) or end_tok_idx >= len(token_spans):
                    continue
                    
                start_span = token_spans[start_tok_idx]
                end_span = token_spans[end_tok_idx]
                
                if start_span and end_span:
                    start_char = start_span[0]
                    end_char = end_span[1]
                    
                    # Heuristic to fix duplicate token indices from buggy dataset generator
                    span_key = (start_tok_idx, end_tok_idx)
                    if span_key in seen_spans:
                        # We saw this exact token span before. Let's find the NEXT occurrence of this string in the text.
                        last_end_char = seen_spans[span_key]
                        extracted_str = text[start_char:end_char]
                        next_pos = text.find(extracted_str, last_end_char)
                        if next_pos != -1:
                            start_char = next_pos
                            end_char = next_pos + len(extracted_str)
                    
                    seen_spans[span_key] = end_char
                    
                    # Post-processing: extend trailing characters like '+' and '#' for C++ and C#
                    start_char, end_char = extend_span(text, start_char, end_char)
                    
                    entities.append([start_char, end_char, label])
            
            # remove true duplicates just in case
            unique_entities = []
            for e in entities:
                if e not in unique_entities:
                    unique_entities.append(e)
            
            out_item = raw_item.copy()
            out_item['label'] = unique_entities
            results.append(out_item)
            
            i_raw += 1
            i_tok += 1
            matched_count += 1
        else:
            i_raw += 1
            
    print(f"Matched {matched_count} items. Missing {len(tok_data) - matched_count} tokenized items.")
    
    with open('converted_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("Saved to converted_dataset.json")

if __name__ == "__main__":
    main()
