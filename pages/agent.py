# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:50:09 2023

@author: Florian Martin
"""

import customtkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import matplotlib.animation as animation 
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from scipy.stats import norm
import statistics
import threading
import time
import pickle
import os
import pandas as pd

from model.environment import GridEnv
from model.cell import HealthyCell, CancerCell, OARCell, Cell
from pages.helpPage import help_page

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DOSE = 60.0
DURATION = 400.0
SURVIVAL = 92.5


def q_table_agent(name, path, axes_table, states_label, fig_table):
    
    def get_q_color(value, vals):
        if all(x==max(vals) for x in vals):
            return "grey", 0.5
        if value == max(vals):
            return "green", 1.0
        else:
            return "red", 0.3

    if 'baseline' not in path:
        try:
            q_table = np.load(path +  f'\\q_table_{int_from_str(path)}.npy', allow_pickle=False)
        except:
            q_table = np.load(path + f'\\q_table_{name}.npy', allow_pickle=False)
        
        axes_table[0].clear()
        axes_table[1].clear()
        axes_table[2].clear()
        axes_table[3].clear()
        
        axes_table[0].set_title("Action 1 : 1 Gray")
        axes_table[1].set_title("Action 2 : 2 Grays")
        axes_table[2].set_title("Action 3 : 3 Grays")
        axes_table[3].set_title("Action 4 : 4 Grays")
        
        count = 0
        for x, x_vals in enumerate(q_table):
                for y, y_vals in enumerate(x_vals):
                    axes_table[0].scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
                    axes_table[1].scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
                    axes_table[2].scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])
                    axes_table[3].scatter(x, y, c=get_q_color(y_vals[3], y_vals)[0], marker="o", alpha=get_q_color(y_vals[3], y_vals)[1])
                
                    if all(x==y_vals[0] for x in y_vals):
                        count += 1
                        
        states_label.configure(text=f"Number of unexplored states : {count}")
        fig_table.canvas.draw()

def boxplot_agent(axes, fig_box, fractions, duration, survival, name):
    
    axes[0].clear()
    axes[1].clear()
    axes[2].clear()
    
    split = name.split('_')
    if len(split) == 6:
        radio = (11*float(split[0])+8*float(split[1])+4*float(split[2])+float(split[3]))/24
        name = f'Radiosensitivity of {radio:.5f}'
    
    fig_box.suptitle(name)

    # Create a DataFrame for each list
    data_fractions = pd.DataFrame({"Values": fractions})
    data_fractions["Type"] = "Fractions [-]"
    
    data_duration = pd.DataFrame({"Values": duration})
    data_duration["Type"] = "Duration \n [hours]"
    
    data_survival = pd.DataFrame({"Values": survival})
    data_survival["Type"] = "Survival [-]"
    
    # Combine the three DataFrames
    data = pd.concat([data_fractions, data_duration, data_survival], ignore_index=True)
    
    # Loop through each subplot and create the boxplot with scatter points
    for i, data_type in enumerate(["Fractions [-]", "Duration \n [hours]", "Survival [-]"]):
        sns.boxplot(x="Values", y="Type", orient='h', data=data[data["Type"] == data_type], ax=axes[i], palette="Set2", width=0.5)
        sns.stripplot(x="Values", y="Type", orient='h', data=data[data["Type"] == data_type], ax=axes[i], color=".25")
        axes[i].set_ylabel("")

    fig_box.canvas.draw()

def int_from_str(r):
    return ''.join(x for x in r if x.isdigit())

def get_agent(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def list_agent(TCP=False, low_dose=False, low_duration=False, high_survival=False):
    lst = [('Baseline', dir_path + '\\TabularAgentResults\\results_baseline.pickle')]
    filename = dir_path + "\\TabularAgentResults\\"
    list_dir = [(f.name, f.path) for f in os.scandir(filename) if f.is_dir()]
    
    for name, path in list_dir:
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        for i in range(len(subfolders)):
            lst.append((name + ' ' + int_from_str(subfolders[i]), subfolders[i]))
            
    df = list_to_dataframe(lst)
    return constraint_agent(df, TCP, low_dose, low_duration, high_survival) 

def list_agent2(TCP=False, low_dose=False, low_duration=False, high_survival=False):
    
    filename = dir_path + "\\TabularAgentRobustCellCycle\\"
    list_dir = [(f.name, f.path) for f in os.scandir(filename) if f.is_dir()]
            
    df = list_to_dataframe(list_dir) 
    return constraint_agent(df, TCP, low_dose, low_duration, high_survival) 

def list_agent3(TCP=False, low_dose=False, low_duration=False, high_survival=False):
    
    filename = dir_path + "\\TabularAgentRobustRadio\\"
    list_dir = [(f.name, f.path) for f in os.scandir(filename) if (f.is_dir()) and ('more' not in f.name) and ('pycache' not in f.name) and ('SARSAgent' not in f.name) and ('QAgent' not in f.name)]
    
    folders = [(f.name, f.path) for f in os.scandir(filename) if (f.is_dir()) and ('pycache' not in f.name) and ('more' in f.name)]
    
    for name, path in folders:
        tmp_lst = os.scandir(path)
        for file in tmp_lst:
            list_dir.append((file.name, file.path))
        
    df = list_to_dataframe(list_dir)
    return constraint_agent(df, TCP, low_dose, low_duration, high_survival) 

def list_to_dataframe(list_dir):
    
    dict_ = {"name" : [], "path" : [], "TCP" : [], "fractions" : [], "doses" : [], "duration" : [], "survival" : [],
             "mean_fractions" : [], "mean_doses" : [], "mean_duration" : [], "mean_survival" : []}
    
    for name, path in list_dir:
        tmp_dict, _ = get_agent_property(name, path)
        dict_["name"].append(name)
        dict_["path"].append(path)
        dict_["TCP"].append(tmp_dict["TCP"])
        for key in tmp_dict.keys():
            if key in ["fractions", "doses", "duration", "survival"]:
                mean, std = np.mean(tmp_dict[key]), np.std(tmp_dict[key])
                dict_[key].append(f"{mean:.3f}" + ' ' + u"\u00B1" + ' ' + f"{std:.3f}")
                dict_[f"mean_{key}"].append(mean)
        
    return pd.DataFrame.from_dict(dict_)

def constraint_agent(df, TCP, low_dose, low_duration, high_survival):
    if TCP:
        df = df[df["TCP"] == 100.0]
    if low_dose:
        df = df[df["mean_doses"] < DOSE]
    if low_duration:
        df = df[df["mean_duration"] < DURATION]
    if high_survival:
        df = df[df["mean_survival"] > SURVIVAL]
        
    return df


    
def get_agent_property(file_name, file_path):
    name, path = file_name, file_path
    if name == 'Baseline':
        return (get_agent(path), None)
    else:
        try:
            tmp_dict = get_agent(path + f'\\results_{int_from_str(path)}.pickle')
            path_q_table = path + f'\\q_table_{int_from_str(path)}'
        except:
            tmp_dict = get_agent(path + f'\\results_{name}.pickle')
            path_q_table = path + f'\\q_table_{name}'
            
        return (tmp_dict, path_q_table)

def data_table(file_name1, file_name2):
    
    file_list = pd.concat([list_agent(), list_agent2(), list_agent3()])
    
    df  = file_list[file_list["name"] == 'Baseline']
    df1 = file_list[file_list["name"] == file_name1]
    df2 = file_list[file_list["name"] == file_name2]
    
    data = [('TCP', "{0}".format(df1["TCP"].iloc[0]), "{0}".format(df2["TCP"].iloc[0]), "{0}".format(df["TCP"].iloc[0]))]
    
    for col in df.columns:
        if col in ["fractions", "doses", "duration", "survival"]:
            data.append((col.capitalize(), df1[col].iloc[0], df2[col].iloc[0], df[col].iloc[0]))
        
    return data
    
            
    
def description(menu, file_name1, file_name2):
       
    agent_frame = customtkinter.CTkFrame(menu, fg_color='transparent')
    agent_frame.place(relx=0.01, rely=0.38, relwidth=0.775, relheight=0.45)
    

    headers = ['', file_name1, file_name2, 'Baseline']
    data = data_table(file_name1, file_name2)

    padx_value = 23
    pady_value = 8
    for j, header in enumerate(headers):
                label = customtkinter.CTkLabel(agent_frame, text=header, font=('Arial', 24))
                label.grid(row=0, column=j*4, padx=padx_value, pady=pady_value)
    
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            label = customtkinter.CTkLabel(agent_frame, text=cell, font=('Arial', 18))
            label.grid(row=i+1, column=j*4, padx=padx_value, pady=pady_value)
      
    
def update_tabview_agent(tab, state_best_agent, params, combobox_widget):

    agents = []
    TCP=params[0]
    low_dose=params[1]
    low_duration=params[2]
    high_survival=params[3]

    if tab == 'Non-Robust':
        df = list_agent(TCP, low_dose, low_duration, high_survival)
        agents = df["name"]
        if state_best_agent:
            agents = [agent for agent in agents if (('SARSAgent 17' in agent) and ('Exp' not in agent)) or ('Baseline' in agent)]
    elif tab == 'Robust Cell Cycle':
        df = list_agent2(TCP, low_dose, low_duration, high_survival)
        agents = df["name"]
        if state_best_agent:
            agents = [agent for agent in agents if ('16_cc_14' in agent) or ('18_cc_18' in agent) or ('20_cc_18' in agent)]
    elif tab == 'Robust Radiosensitivity': 
        agents = list_agent3(TCP, low_dose, low_duration, high_survival)["name"]

    combobox_widget.configure(values=agents)