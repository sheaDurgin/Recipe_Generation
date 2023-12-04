import tensorflow as tf
from tensorflow.keras.models import load_model
from recipe_classes import TextGenerator, TransformerBlock, TokenAndPositionEmbedding
import json
import tkinter as tk

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

entry = tk.Entry(root, width=80)
entry.pack()

generate_button = tk.Button(root, text="Generate", command=generate_recipe)
generate_button.pack()

text_area = tk.Text(root, height=10, width=80)
text_area.pack()

root.mainloop()
