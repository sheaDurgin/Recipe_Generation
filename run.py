import tkinter as tk
from recipe_generator import text_generator

def generate_recipe():
    prompt = entry.get()
    prompt = f"Recipe for {prompt} |"
    info = text_generator.generate(prompt, max_tokens=128, temperature=0.7)
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
