import tensorflow as tf
from tensorflow.keras import layers
import json
import re
import string

BATCH_SIZE = 256
VOCAB_SIZE = 50000
MAX_LEN = 128

with open("recipes.txt", 'r', encoding='utf-8') as f:
    recipes = [line for line in f]

def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s

text_data = [pad_punctuation(x) for x in recipes]

text_ds = (
    tf.data.Dataset.from_tensor_slices(text_data)
    .batch(BATCH_SIZE)
    .shuffle(1000)
)

vectorize_layer = layers.TextVectorization(
    standardize="lower",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN + 1,
)

vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()

vocab_json = json.dumps(vocab)

with open("vocab.json", "w") as json_file:
    json_file.write(vocab_json)