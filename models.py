# --------------------------------------------------------------------------
# ------------------------ Grammer Checker --------------------------------
# --------------------------------------------------------------------------
from happytransformer import HappyTextToText, TTSettings
import torch
import nltk
nltk.download('punkt')
from fastapi.responses import JSONResponse

happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
if torch.cuda.is_available():
    happy_tt.model.to("cuda")
    happy_tt.tokenizer.to("cuda")

args = TTSettings(num_beams=5, min_length=1)


def grammer_correction(text):
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    args = TTSettings(num_beams=5, min_length=1)
    print("HERE")
    result = happy_tt.generate_text(f"grammar: {text}", args=args)
    print("HERE2")
    return result.text

output = grammer_correction("He are moving here.")
# print(output)


# --------------------------------------------------------------------------
# ------------------------ Split Text -------------------------------------
# --------------------------------------------------------------------------
from transformers import T5Tokenizer, T5ForConditionalGeneration



def text_split(input=""):
    checkpoint = "unikei/t5-base-split-and-rephrase"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    complex_tokenized = tokenizer(
        input,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    print("error1")
    print("---------------------")
    # print(dir(model))
    simple_tokenized = model.generate(
        complex_tokenized["input_ids"],
        attention_mask=complex_tokenized["attention_mask"],
        max_length=256,
        num_beams=5,
    )
    print("error2")
    simple_sentences = tokenizer.batch_decode(
        simple_tokenized, skip_special_tokens=True
    )
    sentences = [s.strip() for s in simple_sentences[0].split(".") if s.strip()]
    return sentences

output = text_split("She did not cheat on the test, for it was the wrong thing to do.")
# print(output)
# --------------------------------------------------------------------------
# -------------------------- Text Comparator ------------------------------
# --------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sentences = ["That is a happy person", "That is a very happy person"]


def text_comparator(sentence_1, sentence_2):
    print("new error 1")
    
    sentences = [sentence_1, sentence_2]

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(sentences)
    print("new error 2")
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    print(similarity[0][0])
    return similarity[0][0]

output = text_comparator("I love her", "She loves him")
# print(output)

# --------------------------------------------------------------------------
# -------------------------- Multi Labeling ------------------------------
# --------------------------------------------------------------------------

# import numpy as np
# import tensorflow as tf
# import random
# from tensorflow.keras.models import load_model 
# import nltk
# from nltk.stem.lancaster import LancasterStemmer
# import json
# import pandas as pd
# import keras
# import pickle

# stemmer = LancasterStemmer()

# with open("labeling_models/words.pkl", "rb") as words_file:
#     words = pickle.load(words_file)
# # words = pd.read_pickle(r'labeling_models/words.pkl')

# with open("labeling_models/classes.pkl", "rb") as classes_file:
#     classes = pickle.load(classes_file)
# model = load_model('new_model/model.h5')
# # with open("new_model/model.h5", "rb") as model_file:
# #     model = pickle.load(model_file)
# # model = tf.keras.models.load_model('new_model/my_model.keras')
# def response(user_input):
#     user_input_words = nltk.word_tokenize(user_input)
#     user_input_words = [stemmer.stem(word.lower()) for word in user_input_words]
#     input_bag = [0] * len(words)

#     for s in user_input_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 input_bag[i] = 1

#     input_bag = np.array(input_bag).reshape(1, -1)

#     results = model.predict(input_bag)

#     result_index = np.argmax(results)
#     tag = classes[result_index]

#     with open("./dataset.json") as json_data:
#         intents = json.load(json_data)

#     for intent in intents["intents"]:
#         if intent["tag"] == tag:
#             response = random.choice(intent["responses"])
#             break

#     return response

# output = response("The dosa was tasty")
# print(output)

# response(
#     "All around lovely brunch spot! Cute and quaint decor, very friendly and accommodating host and wait staff, delicious coffee, BYOB and endless food options. It was very difficult to choose one thing, but the combinations are unique and flavorful- far from routine. The meals are reasonably portioned and packed with flavor. I appreciate the time out into each dish as opposed to hugeee portions of hastily made eggs or pancakes. There are so many elements to the dishes. I had the gypsy eggs! Will have took back to try other options! Expect to wait to be seated and don't go with a very large group as seating will be difficult- in my mind it's worth the wait and the wait shows that many others feel the same! Have heard from multiple sources that this was a spot not to miss and I 100% concur. Thank you for great food, friendly atmosphere and BYOB"
# )

# --------------------------------------------------------------------------
# -------------------------- Sentimental Analysis ------------------------
# --------------------------------------------------------------------------

# from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification
# from transformers import AutoTokenizer, AutoConfig
# import numpy as np
# from scipy.special import softmax

# def preprocess(text):
#     new_text = []
#     for t in text.split(" "):
#         t = '@user' if t.startswith('@') and len(t) > 1 else t
#         t = 'http' if t.startswith('http') else t
#         new_text.append(t)
#     return " ".join(new_text)
# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# config = AutoConfig.from_pretrained(MODEL)

# model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# def sentimental_analysis(text):
#     text = preprocess(text)
#     encoded_input = tokenizer(text, return_tensors='pt')
#     output = model(**encoded_input)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#     ranking = np.argsort(scores)
#     ranking = ranking[::-1]
#     highest_score = -np.inf
#     highest_label = None

#     for i in range(scores.shape[0]):
#         l = config.id2label[ranking[i]]
#         s = scores[ranking[i]]
#         rounded_score = np.round(float(s), 4)
#         if rounded_score > highest_score:
#             highest_score = rounded_score
#             highest_label = l

#     return highest_label
from pysentimiento import create_analyzer
def sentimental_analysis(text):
    analyzer = create_analyzer(task="sentiment", lang="en")
    output = analyzer.predict(text)
    highest_prob_sentiment = max(output.probas, key=output.probas.get)
    return highest_prob_sentiment
# sentimental_analysis("I am happy")
output = sentimental_analysis("The waiter took long time to serve")
# print(output)


# ///////////////////////////////////////////////////////////////////////////////////
import numpy as np
import tensorflow as tf
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import warnings
from tensorflow.keras.models import load_model 
import pandas as pd
from datasets import load_dataset

stemmer = LancasterStemmer()

# model = tf.keras.models.load_model("my_model.keras")

# with open("classes.h5", "rb") as classes_file:
#     classes = pickle.load(classes_file)

# with open("words.h5", "rb") as words_file:
#     words = pickle.load(words_file)

with open("labeling_models/words.pkl", "rb") as words_file:
    words = pickle.load(words_file)
# words = pd.read_pickle(r'labeling_models/words.pkl')

with open("labeling_models/classes.pkl", "rb") as classes_file:
    classes = pickle.load(classes_file)
model = load_model('new_model/model.h5')

def preprocess_input(input_text):
    input_words = nltk.word_tokenize(input_text)
    input_words = [stemmer.stem(word.lower()) for word in input_words]
    input_bag = [0] * len(words)

    for s in input_words:
        for i, w in enumerate(words):
            if w == s:
                input_bag[i] = 1

    input_bag = np.array(input_bag).reshape(1, -1)
    return input_bag

def response(input_text):
    input_bag = preprocess_input(input_text)
    predictions = model.predict(input_bag)
    result_index = np.argmax(predictions)
    predicted_class = classes[result_index]
    return predicted_class

input_text = "The service at this restaurant was amazing!"
predicted_class = response(input_text)
# print(predicted_class)