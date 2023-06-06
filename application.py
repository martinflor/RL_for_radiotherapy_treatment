import customtkinter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from scipy.stats import norm
import statistics
import time
import pickle
import webbrowser
import os
import pandas as pd

from model.environment import GridEnv
from pages.simulationPage import SimulationPage
from pages.helpPage import help_page
from pages.agent import *
from pages.RT_plots import *
from pages.Sidebar import SidebarSettings
from pages.treatment_tab import TreatmentTab


import warnings
warnings.filterwarnings("ignore")

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

dir_path = os.path.dirname(os.path.realpath(__file__))

def int_from_str(r):
    return ''.join(x for x in r if x.isdigit())


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("RL and RT")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        taskbar_height = screen_height - self.winfo_rooty()
        
        self.geometry("%dx%d+0+0" % (screen_width, 760))

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # create Sidebar
        self.sidebar = SidebarSettings(self, self.simulation, self.quit_page)
        
        # EPL LOGO
        epl = customtkinter.CTkImage(light_image=Image.open("images/EPL.jpg"),
                                  dark_image=Image.open("images/EPL.jpg"),
                                  size=(150, 80))
        button_epl = customtkinter.CTkButton(self, text= '', 
                                                image=epl, fg_color='transparent')
        button_epl.place(relx=1, rely=1, anchor='se')
        
        # AUTHORS
        self.author_label = customtkinter.CTkLabel(self, text="Author: Florian Martin")
        self.author_label.place(relx=0.16, rely=.975, anchor='sw')
        
        self.supervisor_label = customtkinter.CTkLabel(self, text='Supervisors: Mélanie Ghislain, Manon Dausort, Damien Dasnoy-Sumell, Benoît Macq')
        self.supervisor_label.place(relx=0.16, rely=1.0, anchor='sw')
        
        # PROGRESS BAR
        
        self.progressbar_1 = customtkinter.CTkProgressBar(self)
        self.progressbar_1.place(relx= 0.17, rely=0.83, relwidth=0.82)
        self.progressbar_1.configure(mode="indeterminnate")
        self.progressbar_1.start()
        

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=550, command=self.sidebar.update_helpbox)
        self.tabview.place(relx= 0.17, rely=0.025, relwidth=0.82, relheight=0.8)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.tabview.add("Treatment")
        self.tabview.add("Nutrients")
        self.tabview.add("Cell cycle")
        self.tabview.add("Radiosensitivity")
        self.tabview.add("Classifier")
        self.tabview.tab("Treatment").grid_columnconfigure(0, weight=1)  
        self.tabview.tab("Nutrients").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Cell cycle").grid_columnconfigure(0, weight=1)
        
        # Classifier
        
        self.str_classifier = customtkinter.StringVar(value="CNN")
        self.ckeck_cnn = customtkinter.CTkRadioButton(self.tabview.tab("Classifier"), text="CNN Classifier",
                                     variable=self.str_classifier, value="CNN")

        self.ckeck_rf = customtkinter.CTkRadioButton(self.tabview.tab("Classifier"), text="Random Forest Classifier",
                                     variable=self.str_classifier, value="RF")
        
        self.ckeck_cnn.place(relx=0.65, rely=0.025, relwidth=0.3, relheight=0.1)
        self.ckeck_rf.place(relx=0.16, rely=0.025, relwidth=0.3, relheight=0.1)
        
        classifier = customtkinter.CTkImage(light_image=Image.open("images/ROC_curve_with_thresholds_cnn.jpg"),
                                  dark_image=Image.open("images/ROC_curve_with_thresholds_cnn.jpg"),
                                  size=(620, 450))
        classifier_roc = customtkinter.CTkButton(self.tabview.tab("Classifier"), text= '', 
                                                image=classifier, fg_color='transparent', state='disabled')
        classifier_roc.place(relx=0.51, rely=0.1, relwidth=0.47, relheight=0.99)
        
        classifier_rf = customtkinter.CTkImage(light_image=Image.open("images/ROC_curve_with_thresholds_rf.jpg"),
                                  dark_image=Image.open("images/ROC_curve_with_thresholds_rf.jpg"),
                                  size=(600, 450))
        classifier_roc_rf = customtkinter.CTkButton(self.tabview.tab("Classifier"), text= '', 
                                                image=classifier_rf, fg_color='transparent', state='disabled')
        classifier_roc_rf.place(relx=0.02, rely=0.1, relwidth=0.47, relheight=0.99)
        
        # NUTRIENTS
        self.fields = ('Average healthy glucose absorption', 'Average cancer glucose absorption',
                  'Average healthy oxygen consumption', 'Average cancer oxygen consumption')
        
        self.default = [.36, .54, 20, 20]
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 6), frameon=False)
        self.fig.patch.set_alpha(0)
        plot_data(self.default, self.ax1, self.ax2, self.fig)
        
        canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Nutrients"))
        canvas.draw()
        canvas.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas.get_tk_widget().place(relx=0.45, rely=0.01, relwidth=0.55, relheight=0.99)
        
        self.labels = [customtkinter.CTkLabel(self.tabview.tab("Nutrients"), width=60, text=field+": ", anchor='nw', font=customtkinter.CTkFont(size=16, weight="bold")) for field in self.fields]
        self.entries = [customtkinter.CTkEntry(self.tabview.tab("Nutrients")) for _ in range(4)]
        
        init_x, init_y = 0.05, 0.1
        change_x, change_y = 0.25, 0.1
        
        for idx, ent in enumerate(self.entries):
            self.entries[idx].insert(1, str(self.default[idx]))
            self.entries[idx].bind('<KeyRelease>', self.update_plot_dist)
            self.labels[idx].place(relx=init_x, rely=init_y+change_y*idx)
            self.entries[idx].place(relx=init_x+change_x, rely=init_y+change_y*idx)

        self.table = make_table(self.default, menu=self.tabview.tab("Nutrients"), fields=self.fields)
        self.table.place(relx = 0.1, rely = 0.55)
        
        # Cell Cycle
        self.data = [11, 8, 4, 1, 24]
        self.fig2, self.ax = plt.subplots(2,1, figsize=(6, 9), frameon=False)
        self.fig2.patch.set_alpha(0)
        
        canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tabview.tab("Cell cycle"))
        canvas2.draw()
        canvas2.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas2.get_tk_widget().place(relx=0.45, rely=0.01, relwidth=0.55, relheight=0.99)
        
        
        self.fields_cc = ('Gap 1 (G1)', 'Synthesis (S)', 'Gap 2 (G2)', 'Mitosis (M)', 'Cell Cycle Duration')
        self.labels_cc = [customtkinter.CTkLabel(self.tabview.tab("Cell cycle"), width=60, text=field+": ", anchor='nw', font=customtkinter.CTkFont(size=16, weight="bold")) for field in self.fields_cc]
        self.entries_cc = [customtkinter.CTkEntry(self.tabview.tab("Cell cycle")) for _ in range(len(self.fields_cc))]
        
        init_x, init_y = 0.05, 0.1
        change_x, change_y = 0.25, 0.075
        
        for idx, ent in enumerate(self.entries_cc):
            self.entries_cc[idx].insert(1, str(self.data[idx]))
            self.entries_cc[idx].bind('<KeyRelease>', self.update_plot_pie)
            self.labels_cc[idx].place(relx=init_x, rely=init_y+change_y*idx)
            self.entries_cc[idx].place(relx=init_x+change_x, rely=init_y+change_y*idx)
        
        self.entry_text = tk.StringVar()
        self.entries_cc[idx].configure(textvariable=self.entry_text)
        plot_pie(self.data, self.ax, self.fig2, self.entries_cc, self.entry_text)
        
        
        # Radiosensitivity
        
        self.radio = [1, .75, 1.25, 1.25, .75, 0.96875]
        self.fig_radio, self.ax_radio = plt.subplots(1, 1, figsize=(12,9))
        self.fig_radio.patch.set_alpha(0)
        
        canvas_radio = FigureCanvasTkAgg(self.fig_radio, master=self.tabview.tab("Radiosensitivity"))
        canvas_radio.draw()
        canvas_radio.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas_radio.get_tk_widget().place(relx=0.45, rely=0.01, relwidth=0.55, relheight=0.99)
        
        self.fields_radio = ('Gap 1 (G1)', 'Synthesis (S)', 'Gap 2 (G2)', 'Mitosis (M)', 'Quiescent (G0)', 'Sum of radiosensitivities')
        self.labels_radio = [customtkinter.CTkLabel(self.tabview.tab("Radiosensitivity"), width=60, text=field+": ", anchor='nw', font=customtkinter.CTkFont(size=16, weight="bold")) for field in self.fields_radio]
        self.entries_radio = [customtkinter.CTkEntry(self.tabview.tab("Radiosensitivity")) for _ in range(len(self.fields_radio))]
        
        init_x, init_y = 0.05, 0.1
        change_x, change_y = 0.25, 0.075
        
        for idx, ent in enumerate(self.entries_radio):
            self.entries_radio[idx].insert(1, str(self.radio[idx]))
            self.entries_radio[idx].bind('<KeyRelease>', self.update_plot_radio)
            self.labels_radio[idx].place(relx=init_x, rely=init_y+change_y*idx)
            self.entries_radio[idx].place(relx=init_x+change_x, rely=init_y+change_y*idx)
        
        self.entry_text2 = tk.StringVar()
        self.entries_radio[idx].configure(textvariable=self.entry_text2)
        plot_radio(self.radio, self.ax_radio, self.entries_radio, self.entry_text2, self.fig_radio, self.fields_radio)
 
        # Automatic Robust Mode
        self.str_robust = customtkinter.StringVar()
        self.checkbox_robust = customtkinter.CTkCheckBox(self, text="Automatic Robust Decision Making",
                                     variable=self.str_robust, onvalue="on", offvalue="off")
        self.checkbox_robust.place(relx=0.185, rely=0.85, relwidth=0.3, relheight=0.08)
        
        # TREATMENT
        self.treatment_tab = TreatmentTab(master=self.tabview.tab("Treatment"))
    
    
    def get_values(self, value_type='float', fields=None, entries=None):
        if value_type not in ['float', 'int']:
            raise ValueError("Invalid value_type. Accepted values are 'float' and 'int'.")
    
        if fields is None or entries is None:
            raise ValueError("Both fields and entries should be provided.")
    
        values = []
    
        for idx, field in enumerate(fields):
            if value_type == 'float':
                values.append(float(entries[idx].get()))
            elif value_type == 'int':
                values.append(int(entries[idx].get()))
    
        return values

    def update_plot_dist(self, event):
        values = self.get_values(value_type='float', fields=self.fields, entries=self.entries)
        self.after(500, plot_data, values, self.ax1, self.ax2, self.fig)
        update_table(values, self.table)
        
    def update_plot_radio(self, event):
        self.after(500, plot_radio, self.get_values(value_type='float', fields=self.fields_radio, entries=self.entries_radio), self.ax_radio, self.entries_radio, self.entry_text2, self.fig_radio, self.fields_radio)
        
    def update_plot_pie(self, event):
        self.after(500, plot_pie, self.get_values(value_type='int', fields=self.fields_cc, entries=self.entries_cc), self.ax, self.fig2, self.entries_cc, self.entry_text)
        
    def simulation(self):
        file_name = self.treatment_tab.combobox_1.get()
        file_list = pd.concat([list_agent(), list_agent2(), list_agent3()])
        path = file_list[file_list["name"]==file_name]["path"].iloc[0]
        
        params = self.get_values(value_type='float', fields=self.fields, entries=self.entries) + [file_name, path, self.get_values(value_type='int', fields=self.fields_cc, entries=self.entries_cc), self.get_values(value_type='float', fields=self.fields_radio, entries=self.entries_radio), self.str_robust.get(), self.str_classifier.get()]
        self.simulation = SimulationPage(self, params)
        
    def quit_page(self):
        self.quit()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()