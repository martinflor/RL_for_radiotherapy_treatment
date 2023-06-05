# -*- coding: utf-8 -*-
"""
Created on Fri May  5 18:39:24 2023

@author: Florian Martin
"""

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

from pages.RT_plots import *
from pages.agent import *


DOSE = 60.0
DURATION = 400.0
SURVIVAL = 92.5

class TreatmentTab(customtkinter.CTkFrame):
    
    def __init__(self, master):
        self.master = master
        
        # TREATMENT
        
        self.tabview_combobox_left = customtkinter.CTkTabview(self.master, width=550, command=self.update_tabview_agent_left)
        self.tabview_combobox_left.place(relx= 0.01, rely=0.12, relwidth=0.4, relheight=0.17)
        
        self.tabview_combobox_left.add("Non-Robust")
        self.tabview_combobox_left.add("Robust Cell Cycle")
        self.tabview_combobox_left.add("Robust Radiosensitivity")
        
        self.tabview_combobox_right = customtkinter.CTkTabview(self.master, width=550, command=self.update_tabview_agent_right)
        self.tabview_combobox_right.place(relx= 0.425, rely=0.12, relwidth=0.4, relheight=0.17)
        
        self.tabview_combobox_right.add("Non-Robust")
        self.tabview_combobox_right.add("Robust Cell Cycle")
        self.tabview_combobox_right.add("Robust Radiosensitivity")

        # Classify Agents
        init_x, init_y = 0.835, 0.26
        step = 0.075
        self.str_best_agents = customtkinter.StringVar()
        self.checkbox_best_agents = customtkinter.CTkCheckBox(self.master, text="Only Best Agents", command=self.update_tabview_agent_left,
                                     variable=self.str_best_agents, onvalue=True, offvalue=False, font=customtkinter.CTkFont(size=16, weight="bold"))
        self.checkbox_best_agents.place(relx=init_x, rely=init_y, relwidth=0.35, relheight=0.1)
        
        
        self.str_100TCP_agents = customtkinter.StringVar()
        self.checkbox_100TCP_agents = customtkinter.CTkCheckBox(self.master, text="100% TCP Agents", command=self.update_tabview_agent_left,
                                     variable=self.str_100TCP_agents, onvalue=True, offvalue=False, font=customtkinter.CTkFont(size=16, weight="bold"))
        self.checkbox_100TCP_agents.place(relx=init_x, rely=init_y+step, relwidth=0.35, relheight=0.1)
        
        
        self.str_low_dose_agents = customtkinter.StringVar()
        self.checkbox_low_dose_agents = customtkinter.CTkCheckBox(self.master, text=f"Low Dose (<{DOSE})", command=self.update_tabview_agent_left,
                                     variable=self.str_low_dose_agents, onvalue=True, offvalue=False, font=customtkinter.CTkFont(size=16, weight="bold"))
        self.checkbox_low_dose_agents.place(relx=init_x, rely=init_y+2*step, relwidth=0.35, relheight=0.1)
        
        
        self.str_low_duration_agents = customtkinter.StringVar()
        self.checkbox_low_duration_agents = customtkinter.CTkCheckBox(self.master, text=f"Low Duration (<{DURATION})", command=self.update_tabview_agent_left,
                                     variable=self.str_low_duration_agents, onvalue=True, offvalue=False, font=customtkinter.CTkFont(size=16, weight="bold"))
        self.checkbox_low_duration_agents.place(relx=init_x, rely=init_y+3*step, relwidth=0.35, relheight=0.1)
        
        
        self.str_high_survival_agents = customtkinter.StringVar()
        self.checkbox_high_survival_agents = customtkinter.CTkCheckBox(self.master, text=f"High Survival (>{SURVIVAL})", command=self.update_tabview_agent_left,
                                     variable=self.str_high_survival_agents, onvalue=True, offvalue=False, font=customtkinter.CTkFont(size=16, weight="bold"))
        self.checkbox_high_survival_agents.place(relx=init_x, rely=init_y+4*step, relwidth=0.35, relheight=0.1)
        
        self.popup = None
        self.agent_perfs = customtkinter.CTkButton(self.master, text="Agent's Performances", command=self.agents_performances)
        self.agent_perfs.place(relx=init_x, rely=init_y+6*step, relwidth=0.15, relheight=0.05)
        
        # LISTING IN COMBOBOXES
        lst = list_agent()["name"]
        
        self.combobox_label = customtkinter.CTkLabel(self.master, text="Selected RL Agent:", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.combobox_label.place(relx=0.015, rely=0.05, relwidth=0.25, relheight=0.05)
        self.combobox_1 = customtkinter.CTkComboBox(self.master,
                                                    values=lst, command=self.update_description)
        self.combobox_1.place(relx=0.072, rely=0.215, relwidth=0.275, relheight=0.06)
        
        self.combobox_label2 = customtkinter.CTkLabel(self.master, text="Compare RL Agent:", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.combobox_label2.place(relx=0.415, rely=0.05, relwidth=0.25, relheight=0.05)
        
        self.combobox_2 = customtkinter.CTkComboBox(self.master,
                                                    values=lst, command=self.update_description)
        self.combobox_2.place(relx=0.49, rely=0.215, relwidth=0.275, relheight=0.06)
        
        self.update_description(1)
        
    def update_description(self, event):
        agent_name1 = self.combobox_1.get()
        print(agent_name1)
        agent_name2 = self.combobox_2.get()
        description(self.master, agent_name1, agent_name2)
        if self.popup is not None:
            self.update_agents_performances()
        
    def update_tabview_agent_left(self):
        tab = self.tabview_combobox_left.get()
        state = self.checkbox_best_agents.get()
        params = [self.checkbox_100TCP_agents.get(), self.checkbox_low_dose_agents.get(), self.checkbox_low_duration_agents.get(), self.checkbox_high_survival_agents.get()]
        update_tabview_agent(tab, state, params, self.combobox_1)
        self.update_description(1)
        
        
    def update_tabview_agent_right(self):
        tab = self.tabview_combobox_right.get()
        update_tabview_agent(tab, 0, [False, False, False, False], self.combobox_2)
        self.update_description(1)
        
        
    def agents_performances(self):
        if self.popup is None:
            self.popup = tk.Toplevel()
            self.popup.title("Agents' Performances")
            screen_width = self.popup.winfo_screenwidth()
            screen_height = self.popup.winfo_screenheight()
            taskbar_height = screen_height - self.popup.winfo_rooty()
        
            self.popup.geometry("%dx%d+0+0" % (screen_width, 760))
            self.popup.protocol("WM_DELETE_WINDOW", self.on_popup_close)
            
            self.tabview_tt = customtkinter.CTkTabview(self.popup, width=550)
            self.tabview_tt.place(relx=0.05, rely=0.12, relwidth=0.9, relheight=.82)
    
            self.tabview_tt.add("Performances")
            self.tabview_tt.add("Agent's q-table")
            
            self.tabview_tt.tab("Performances").grid_columnconfigure(0, weight=1)  
            self.tabview_tt.tab("Agent's q-table").grid_columnconfigure(0, weight=1)


            self.fig_box, self.axes = plt.subplots(3,1, figsize=(24,20))
            self.fig_box.patch.set_alpha(0)
            self.canvas_box = FigureCanvasTkAgg(self.fig_box, master=self.tabview_tt.tab("Performances"))
            self.canvas_box.draw()
            self.canvas_box.get_tk_widget().config(highlightthickness=0, borderwidth=0)
            self.canvas_box.get_tk_widget().place(relx=0.01, rely=0.01, relwidth=0.99, relheight=0.99)
            self.states_label = customtkinter.CTkLabel(self.popup, text="Number of unexplored states by the agent : /", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
            self.states_label.place(relx=0.015, rely=0.01, relwidth=0.35, relheight=0.1)
    
            self.fig_table, self.axes_table = plt.subplots(4, 1, constrained_layout=True, figsize = (16,12))
            self.fig_table.patch.set_alpha(0)
            self.canvas_table = FigureCanvasTkAgg(self.fig_table, master=self.tabview_tt.tab("Agent's q-table"))
            self.canvas_table.draw()
            self.canvas_table.get_tk_widget().config(highlightthickness=0, borderwidth=0)
            self.canvas_table.get_tk_widget().place(relx=0.01, rely=0.01, relwidth=0.99, relheight=0.99)
  
            self.update_agents_performances()
            
    def on_popup_close(self):
        self.popup.destroy()
        self.popup = None
    
    def update_agents_performances(self):
        file_name = self.combobox_1.get()
        file_list = pd.concat([list_agent(), list_agent2(), list_agent3()])
        path = file_list[file_list["name"]==file_name]["path"].iloc[0]
        tmp_dict, path_q_table = get_agent_property(file_name, path)
        fractions, duration, survival = tmp_dict["fractions"], tmp_dict["duration"], tmp_dict["survival"]
        
        boxplot_agent(self.axes, self.fig_box, fractions, duration, survival, file_name)
        q_table_agent(file_name, path, self.axes_table, self.states_label, self.fig_table)