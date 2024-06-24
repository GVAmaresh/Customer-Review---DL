from models import grammer_correction, response, text_split, sentimental_analysis, text_comparator

def process_models(input):
    split_output = [grammer_correction(text) for text in text_split(input) ]
    output = {response(i) : [i, sentimental_analysis(i)] for i in split_output}
    return output

def compare_text(texts, text):
    similarities = {}
    for i in texts:
        value = text_comparator(i, text)
        similarities[i] = value
    max_text = max(similarities, key=similarities.get, default=None)
    max_value = similarities[max_text] if max_text is not None else None
    
    return max_text if max_value and max_value > 0.9 else None
    