import tkinter as tk
import tkinter as tk
from tkinter import ttk
from recipe_generator import text_generator

def generate_recipe():
    prompt = entry.get()
    prompt = f"Recipe for {prompt} |"
    info = text_generator.generate(prompt, max_tokens=128, temperature=0.7)
    text_area.delete("1.0", tk.END)
    text_area.insert(tk.END, info)


# sample/mock function
# def generate_recipe():
#     prompt = entry.get()
#     # Mock response
#     info = "This is a sample recipe response for " + prompt
#     text_area.delete("1.0", tk.END)
#     text_area.insert(tk.END, info)

root = tk.Tk()
root.title("Recipe Generator")

#window to be 500x500 ?
root.geometry("500x500")



#recipe generator :D
title = tk.Label(root, text="👨🏻‍🍳 Recipe Generator 👨🏻‍🍳", foreground = "pink", font=("Lucida Grande", 24))
title.pack(pady=10)

# label/prompt
label = tk.Label(root, text="Type in a dish name to get started with a recipe", font=("Lucida Grande", 18))
label.pack(pady=10)

# Entry widget
entry = tk.Entry(root, width=30, font=("Lucida Grande", 18))
entry.pack(pady=10)

# Button
style = ttk.Style()
style.configure("TButton", background="#ADD8E6", foreground="pink", font=('Lucida Grande', 14))

# Creating a themed Button
generate_button = ttk.Button(root, text="Generate Recipe and Get Cookin'!", command=generate_recipe, style="TButton")
generate_button.pack(padx=20, pady=20)

# Text area
text_area = tk.Text(root, width=60, height=35, bg="#000080", fg="#FFC0CB", padx=10, pady=10)
text_area.pack(pady=10)

root.mainloop()
