# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:06:10 2023

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
from pages.agent import *


def plot_pie(data, ax, fig2, entries_cc, entry_text):
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct
    
    default = [11, 8, 4, 1, 24]
    labels = ['Gap 1', 'Synthesis', 'Gap 2', 'Mitosis']
    colors = sns.color_palette('pastel')[0:4]
    ax[0].clear()
    ax[1].clear()
    ax[0].pie(default[:-1], labels = labels, colors = colors, autopct=make_autopct(default[:-1]))
    ax[1].pie(data[:-1], labels = labels, colors = colors, autopct=make_autopct(data[:-1]))
    plt.tight_layout()
    
    entries_cc[-1].configure(state='normal')
    entry_text.set(str(sum(data[:-1])))
    entries_cc[-1].configure(state='disabled')
    
    fig2.canvas.draw()
    
def plot_data(values, ax1, ax2, fig):
    ax1.clear()
    ax2.clear()

    f1 = lambda x, y: min(x, y)
    f2 = lambda x, y: max(x, y)

    # Plot between -10 and 10 with .001 steps.
    x_axis = np.arange(-0.5, 2.5, 0.0001)

    # Calculating mean and standard deviation
    mean = 1
    sd = 1 / 3

    normal = norm.pdf(x_axis, mean, sd)
    results = []

    for i in range(len(x_axis)):
        results.append(max(0, min(2, normal[i])))

    ax1.plot(x_axis, np.array(results) * values[1], label="Cancer Cells")
    ax1.plot(x_axis, np.array(results) * values[0], label="Healthy Cells")
    ax1.set_ylabel("Glucose Absorption")
    ax1.grid(alpha=0.5)
    ax1.legend()

    ax2.plot(x_axis, np.array(results) * values[3], label="Cancer Cells")
    ax2.plot(x_axis, np.array(results) * values[2], label="Healthy Cells")
    ax2.set_ylabel("Oxygen Consumption")
    ax2.grid(alpha=0.5)
    ax2.legend()

    # Set the background color of the plot to transparent
    ax1.set_facecolor('none')
    ax2.set_facecolor('none')

    # Remove the border of the plot
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    fig.canvas.draw()
    
def plot_radio(radiosensitivities, ax_radio, entries_radio, entry_text2, fig_radio, fields_radio):
    ax_radio.clear()
    alpha_norm_tissue = 0.15
    beta_norm_tissue = 0.03
    dose = np.linspace(0,10, 1000)
    f = lambda x, stage : np.exp(radiosensitivities[stage] * (-alpha_norm_tissue*x - beta_norm_tissue * (x ** 2)))
    
    for idx, rad in enumerate(radiosensitivities[:-1]):
        survival = f(dose, idx)
        ax_radio.plot(dose, survival, label=f"radiosensitivity of {rad} ({fields_radio[idx]})")
        ax_radio.set_xlabel("Radiation dose (Gy)")
        ax_radio.set_ylabel("Surviving fraction")
    
    ax_radio.legend()
    
    sum_radio = (11*radiosensitivities[0]+8*radiosensitivities[1]+4*radiosensitivities[2]+radiosensitivities[3])/24
    
    entries_radio[-1].configure(state='normal')
    entry_text2.set(str(sum_radio))
    entries_radio[-1].configure(state='disabled')
    
    fig_radio.canvas.draw()
    
def make_table(values, menu, fields):
    table_frame = customtkinter.CTkFrame(menu, fg_color='transparent')
    table_frame.grid(row=len(fields)+1, column=1, padx=10, pady=10, sticky='n')
    
    
    padx_table = 25 
    headers = ['Parameter', 'Value']
    
    rows = [('Quiescent (Glucose)', f"{2*24*values[0]:.3f}"), 
            ('Quiescent (Oxygen)', f"{2*24*values[2]:.3f}"),
            ('Critical (Glucose)', f"{(3/4)*24*values[0]:.3f}"),
            ('Critical (Oxygen)', f"{(3/4)*24*values[2]:.3f}")]
    
    for j, header in enumerate(headers):
        label = customtkinter.CTkLabel(table_frame, text=header, font=customtkinter.CTkFont(size=18, weight="bold"))
        label.grid(row=0, column=j, padx=padx_table, pady=5)
    
    for row, data in enumerate(rows):
        for col, item in enumerate(data):
            label = customtkinter.CTkLabel(table_frame, text=item, font=('Arial', 14))
            label.grid(row=row+1, column=col, padx=padx_table, pady=5, sticky='w')
            
    return table_frame

def update_table(values, table):
    # delete the children of the table_frame widget
    for widget in table.winfo_children():
        widget.destroy()
        
    padx_table = 25 
    headers = ['Parameter', 'Value']
    
    rows = [('Quiescent (Glucose)', f"{2*24*values[0]:.3f}"), 
            ('Quiescent (Oxygen)', f"{2*24*values[2]:.3f}"),
            ('Critical (Glucose)', f"{(3/4)*24*values[0]:.3f}"),
            ('Critical (Oxygen)', f"{(3/4)*24*values[2]:.3f}")]
    
    for j, header in enumerate(headers):
        label = customtkinter.CTkLabel(table, text=header, font=customtkinter.CTkFont(size=18, weight="bold"))
        label.grid(row=0, column=j, padx=padx_table, pady=5)
    
    for row, data in enumerate(rows):
        for col, item in enumerate(data):
            label = customtkinter.CTkLabel(table, text=item, font=('Arial', 14))
            label.grid(row=row+1, column=col, padx=padx_table, pady=5, sticky='w')