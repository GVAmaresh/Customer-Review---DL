#Amazon Food Review
# !pip install kaggle
import json, os

# api_token = {"username":"latestpurpose","key":"446a29c857a2c0256ae815ec34677a95"}
# !mkdir -p ~/.kaggle

# with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as file:
#     json.dump(api_token, file)

# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# !kaggle datasets download -d snap/amazon-fine-food-reviews
# !unzip amazon-fine-food-reviews.zip

import pandas as pd
food_review = pd.read_csv('Reviews.csv')
food_review = food_review.rename(columns={'Text': 'text'})
food_review = food_review["text"].head(200)
food_review


# Restaurant Reviews
# !pip install datasets
from datasets import load_dataset
import pandas as pd


dataset = load_dataset("vincha77/filtered_yelp_restaurant_reviews")
train_texts = dataset['train']['text'][:200]

restaurant_review = pd.DataFrame({'text': train_texts})



import pandas as pd
from datasets import load_dataset

food_review = pd.read_csv('Reviews.csv')
food_review = food_review.rename(columns={'Text': 'text'})
food_review = food_review.head(200) 
food_review['dataset'] = 'food_review'

dataset = load_dataset("vincha77/filtered_yelp_restaurant_reviews")
restaurant_reviews = dataset['train'][:200]

restaurant_review = pd.DataFrame(restaurant_reviews)
restaurant_review['dataset'] = 'restaurant_review'
restaurant_review = restaurant_review[['text', 'dataset']]  

merged_df = pd.concat([food_review[['text', 'dataset']], restaurant_review[['text', 'dataset']]])

merged_df.reset_index(drop=True, inplace=True)

print(merged_df)
import numpy as np
import tensorflow as tf
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import pickle
import warnings
import nltk
nltk.download('punkt')


warnings.filterwarnings("ignore")
stemmer = LancasterStemmer()

words = []
classes = []
documents = []
ignore_words = ["?"]

for idx, row in merged_df.iterrows():
    w = nltk.word_tokenize(row['text'])
    words.extend(w)
    documents.append((w, row['dataset']))

    if row['dataset'] not in classes:
        classes.append(row['dataset'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

training = []
output = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

tf.compat.v1.reset_default_graph()

input_data = tf.keras.Input(shape=(len(train_x[0]),), dtype=tf.float32)
output_data = tf.keras.Input(shape=(len(train_y[0]),), dtype=tf.float32)

hidden_layer1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(input_data)
hidden_layer2 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(hidden_layer1)
output_layer = tf.keras.layers.Dense(len(train_y[0]), activation=tf.nn.softmax)(
    hidden_layer2
)

model = tf.keras.Model(inputs=input_data, outputs=output_layer)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Training....")
model.fit(train_x, train_y, epochs=200, batch_size=8)

with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("classes.pkl", "wb") as classes_file:
    pickle.dump(np.array(classes), classes_file)

with open("words.pkl", "wb") as words_file:
    pickle.dump(np.array(words), words_file)

print("Model trained and saved.")
