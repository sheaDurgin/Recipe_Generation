import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
import numpy as np
import json
import tkinter as tk

class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, model, top_k=10):
        self.index_to_word = index_to_word
        self.model = model
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y, att = self.model.predict(x, verbose=0)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append(
                {
                    "prompt": start_prompt,
                    "word_probs": probs,
                    "atts": att[0, :, -1, :],
                }
            )
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
        return start_prompt

model = load_model("recipe_generator", compile=False)

with open("vocab.json", "r") as json_file:
    vocab = json.load(json_file)

text_generator = TextGenerator(vocab, model)

def generate_recipe():
    prompt = entry.get()
    prompt = f"Recipe for {prompt} |"
    info = text_generator.generate(prompt, max_tokens=256, temperature=0.5)
    text_area.delete("1.0", tk.END)
    text_area.insert(tk.END, info)

root = tk.Tk()
root.title("Recipe Generator")

entry = tk.Entry(root, width=50)
entry.pack()

generate_button = tk.Button(root, text="Generate", command=generate_recipe)
generate_button.pack()

text_area = tk.Text(root, height=10, width=80)
text_area.pack()

root.mainloop()
