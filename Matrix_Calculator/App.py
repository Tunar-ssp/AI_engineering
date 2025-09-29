from Calculator import gaussian_elimination
import tkinter as tk
import numpy as np
def create_matrix_entries(n):
    for widget in window.winfo_children():
        if isinstance(widget, tk.Frame):
            widget.destroy()
    matrix_frame = tk.Frame(window)
    matrix_frame.pack(pady=10)
    entries = []
    for i in range(n):
        row_entries = []
        for j in range(n + 1): 
            e = tk.Entry(matrix_frame, width=5)
            e.grid(row=i, column=j, padx=2, pady=2)
            row_entries.append(e)
        entries.append(row_entries)
    calc_btn = tk.Button(matrix_frame, text="Calculate", command=lambda: calculate(entries))
    calc_btn.grid(row=n, column=0, columnspan=n+1, pady=10)

def calculate(entries):
    matrix = []
    for row in entries:
        matrix.append([float(e.get()) for e in row])
    matrix = np.array(matrix)
    A= matrix[:, :-1]
    B=np.reshape(matrix[:, -1],(-1,1))
    A.astype('float64')
    B.astype('float64')
    print(gaussian_elimination(A, B))


def on_submit():
    try:
        num_vars = int(num_variable_entry.get())
        if num_vars <= 0:
            raise ValueError("Number must be positive")
        error_label.config(text="", fg="black")
        create_matrix_entries(num_vars)
    except ValueError as e:
        error_label.config(text=f"Error: {e}", fg="red")

window = tk.Tk()
window.title("Matrix calculator")
window.geometry("500x500")
window.resizable(False,False)

label = tk.Label(window, text="enter number of variables")
label.pack(pady=10)

num_variable_entry = tk.Entry(window)
num_variable_entry.pack(pady=10)

submit_button = tk.Button(window, text="Enter", command=on_submit)
submit_button.pack(pady=10)

error_label = tk.Label(window, text="")
error_label.pack(pady=10)



window.mainloop()