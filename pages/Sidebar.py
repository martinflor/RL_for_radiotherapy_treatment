# -*- coding: utf-8 -*-
"""
Created on Fri May  5 18:02:22 2023

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

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class SidebarSettings(customtkinter.CTkFrame):
    def __init__(self, master, simulation, quit_page):
        self.master = master
        self.simulation = simulation
        self.quit_page = quit_page
        self.sidebar_frame = customtkinter.CTkFrame(self.master, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Reinforcement Learning \n and \n Radiotherapy", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Start", command=self.simulation, width=190)
        self.sidebar_button_1.grid(row=1, column=0, padx=10, pady=10)
        
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Quit", width=190, fg_color="transparent", text_color=("gray10", "#DCE4EE"), border_width=2, command=self.quit_page)
        self.sidebar_button_4.grid(row=2, column=0, padx=10, pady=10)
        
        # create textbox
        self.textbox = customtkinter.CTkTextbox(self.sidebar_frame)
        self.textbox.place(relx=0.05, rely=0.25, relwidth=0.9, relheigh=0.45)
        
        with open(dir_path + '\\misc\\treatment_help.txt', 'r') as file:
            treatment_file = file.readlines()
            
        with open(dir_path + '\\misc\\cell_cycle_help.txt', 'r') as file:
            cell_cycle_file = file.readlines()
            
        with open(dir_path + '\\misc\\nutrients_help.txt', 'r') as file:
            nutrients_file = file.readlines()
        
        with open(dir_path + '\\misc\\radiosensitivity_help.txt', 'r') as file:
            radio_file = file.readlines()
            
        treatment_file = ''.join(line for line in treatment_file)
        cell_cycle_file = ''.join(line for line in cell_cycle_file)
        nutrients_file = ''.join(line for line in nutrients_file)
        radio_file = ''.join(line for line in radio_file)
                                                            
        
        self.texts = {"Nutrients" : "Help Box\n\n\n " + nutrients_file,
                      "Treatment" : "Help Box\n\n\n " + treatment_file,
                      "Cell cycle" : "Help Box\n\n\n " + cell_cycle_file,
                      "Radiosensitivity" : "Help Box\n\n\n " + radio_file,
                      "Classifier" : "Help Box\n\n\n "}
        self.textbox.insert('0.0', self.texts["Treatment"])
        self.textbox.configure(state="disabled", wrap="word")

        
        # GITHUB ICON
        
        github = customtkinter.CTkImage(light_image=Image.open("images/github.png"),
                                  dark_image=Image.open("images/github.png"),
                                  size=(30, 30))
        button_github = customtkinter.CTkButton(self.sidebar_frame, text= 'GITHUB', 
                                                image=github, fg_color='transparent', text_color=('black', 'white'),
                                                command=self.open_github)
        
        button_github.place(relx=0.025, rely=0.75, relwidth=0.9, relheight=0.05)
        
        # LINKEDIN ICON
        
        linkedin = customtkinter.CTkImage(light_image=Image.open("images/linkedin.png"),
                                  dark_image=Image.open("images/linkedin.png"),
                                  size=(30, 30))
        button_linkedin = customtkinter.CTkButton(self.sidebar_frame, text= 'LinkedIn', 
                                                image=linkedin, fg_color='transparent', text_color=('black', 'white'),
                                                command=self.open_linkedin)
        button_linkedin.place(relx=0.025, rely=0.825, relwidth=0.9, relheight=0.05)
        
        # APPEARANCE MODE
        
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], width=190,
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))
        
    def update_helpbox(self):
        self.textbox.configure(state="normal", wrap="word")
        tab = self.master.tabview.get()
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", self.texts[tab])
        self.textbox.configure(state="disabled", wrap="word")
        
    def open_github(self):
        webbrowser.open_new_tab('https://github.com/martinflor/master_thesis_RL')

    def open_linkedin(self):
        webbrowser.open_new_tab('https://www.linkedin.com/in/florian-martin-554350239/')

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
        
        
class SidebarSimulation(customtkinter.CTkFrame):
    def __init__(self, master, focus_plot, save_plot_hd, save_anim, pause_simulation, quit_page):
        self.master = master 
        self.focus_plot = focus_plot
        self.save_plot_hd = save_plot_hd
        self.save_anim = save_anim
        self.pause_simulation = pause_simulation
        self.quit_page = quit_page

        self.sidebar_frame = customtkinter.CTkFrame(self.master, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Reinforcement Learning \n and \n Radiotherapy", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        
        # ZOOM ON PLOT BUTTON
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Zoom", command=self.focus_plot, width=190)
        self.sidebar_button_1.grid(row=1, column=0, padx=10, pady=10)
        
        # SAVE PLOT BUTTON
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Save Plot", width=190, command=self.save_plot_hd)
        self.sidebar_button_2.grid(row=2, column=0, padx=10, pady=10)      
        
        # SAVE ANIMATION BUTTON
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Save Animation", width=190, command=self.save_anim)
        self.sidebar_button_3.grid(row=3, column=0, padx=10, pady=10)
        
        # PAUSE SIMULATION BUTTON
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Pause", width=190, command=self.pause_simulation)
        self.sidebar_button_4.grid(row=4, column=0, padx=10, pady=10)
        
        # QUIT BUTTON
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Quit", fg_color="transparent", width=190, border_width=2, text_color=("gray10", "#DCE4EE"), command=self.quit_page)
        self.sidebar_button_5.grid(row=5, column=0, padx=10, pady=10)
        
        # HELP BOX
        self.textbox = customtkinter.CTkTextbox(self.sidebar_frame)
        self.textbox.grid(row=6, column=0, rowspan=2, padx=20, pady=10, sticky="nsew")
        self.textbox.insert("0.0", "Help Box\n\n ZOOM BUTTON \n \n Zoom or Unzoom on the three bottom plots \n \n \n SAVE PLOT BUTTON \n \n Saving the entire graphic. The graphic will be found in the folder SAVE. \n \n \n SAVE ANIMATION BUTTON \n \n Saving the entire animation from the beggining to the current time step. It is deprecated to use this button without using the PAUSE button before. Be careful, it might take some minutes before finishing the save, do not close the window during this time. The animation will be found in the folder SAVE. \n \n \n PAUSE BUTTON \n \n Stop the simulation until the user press the button again.")
        self.textbox.configure(state="disabled", wrap="word")
        
        # LIGHT/DARK MODE
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], width=190,
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))  
        
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)