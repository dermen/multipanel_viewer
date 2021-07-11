import tkinter as tk


class LabeledEntry(tk.Frame):

    def __init__(self, master, labeltext,  vartype=str, init_value=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.label = tk.Label(self, text=labeltext)
        self.label.pack(side=tk.LEFT)
        if vartype==str:
            self.variable = tk.StringVar()
        elif vartype==int:
            self.variable = tk.IntVar()
        elif vartype==float:
            self.variable=tk.DoubleVar()

        self.variable.set(init_value)
        self.entry = tk.Entry(self, textvariable=self.variable)
        self.entry.pack(side=tk.LEFT)