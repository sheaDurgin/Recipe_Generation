import tensorflow as tf
from tensorflow.keras import layers
import json
import re
import string

BATCH_SIZE = 64
VOCAB_SIZE = 15000
MAX_LEN = 256

with open("full_format_recipes.json") as json_data:
    recipes = json.load(json_data)

filtered_data = [
    "Recipe for " + x["title"] + " | " + " ".join(x["directions"])
    for x in recipes
    if "title" in x
    and x["title"] is not None
    and "directions" in x
    and x["directions"] is not None
]

def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s

text_data = [pad_punctuation(x) for x in filtered_data]

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

# Convert the vocabulary to JSON
vocab_json = json.dumps(vocab)

# Write the JSON to a file
with open("vocab.json", "w") as json_file:
    json_file.write(vocab_json)