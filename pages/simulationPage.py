import customtkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
import torch

from model.environment import GridEnv
from model.cell import HealthyCell, CancerCell, OARCell, Cell
from pages.helpPage import help_page
from pages.agent import *
from pages.auto_robust import auto_robust_agent_selection
from pages.Sidebar import SidebarSimulation
from pages.treatment_tab import TreatmentTab
from model.CNN.CNN_classifier import BinaryClassifier


import warnings
warnings.filterwarnings("ignore")

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
large = 22; med = 16; small = 12

params = {'axes.titlesize': large,
          'legend.fontsize': small,
          'figure.figsize': (5, 7),
          'axes.labelsize': small,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')

dir_path = os.path.dirname(os.path.realpath(__file__))
        

class SimulationPage(customtkinter.CTkFrame):
    def __init__(self, master=None, params=None):
        self.master = master
        self.stop_event = threading.Event()
        self.is_paused = False  
        self.q_table = None
        
        for i in self.master.winfo_children():
            i.destroy()
            
        self.params = params
        self.name = self.params[4]
        self.previous_name = self.params[4]
        self.path = self.params[5]
        if self.params[8] == "on":
            self.auto_robust = True
            print("Auto-Robust Mode On")
            self.auto_robust_agent = auto_robust_agent_selection(self.name)
        else:
            self.auto_robust = False
            
        if self.params[9] == 'RF':
            print("RF classifier selected")
        else:
            print("CNN classifier selected")

            
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        taskbar_height = screen_height - self.master.winfo_rooty()
        
        self.master.geometry("%dx%d+0+0" % (screen_width, 760))
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        # Sidebar Inputs
        self.sidebar = SidebarSimulation(self.master, self.focus_plot, self.save_plot_hd, self.save_anim, self.pause_simulation, self.quit_page)
        self.save = False
        self.focus = False
        self.agent_on_plot_bool = False
        
        
        init = 0.05
        step = 0.085
        self.cancer_label = customtkinter.CTkLabel(self.master, text=f"Number of Cancer Cells \n \t 0 Cells", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.cancer_label.place(relx=0.86, rely=init, relwidth=0.125, relheight=0.05)
        
        self.healthy_label = customtkinter.CTkLabel(self.master, text=f"Number of Healthy Cells \n \t 0 Cells", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.healthy_label.place(relx=0.86, rely=init+step, relwidth=0.125, relheight=0.05)
        
        
        self.total_dose_label = customtkinter.CTkLabel(self.master, text=f"Total dose administered \n \t 0 Gy", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.total_dose_label.place(relx=0.86, rely=init+2*step, relwidth=0.125, relheight=0.05)
        
        self.retrain_early = customtkinter.CTkLabel(self.master, text=f"Re-train Early : \n    Not Ready", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.retrain_early.place(relx=0.86, rely=init+3*step, relwidth=0.125, relheight=0.05)
        
        self.predict_tcp = customtkinter.CTkLabel(self.master, text=f"Predicted TCP : /", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.predict_tcp.place(relx=0.86, rely=init+4*step, relwidth=0.125, relheight=0.05)
        
        self.agent_on_plot_bool = customtkinter.StringVar(value=False)
        self.agent_plot = customtkinter.CTkCheckBox(self.master, text="Agents on Plot",
                                     variable=self.agent_on_plot_bool, onvalue=True, offvalue=False, font=customtkinter.CTkFont(size=14, weight="bold"))
        
        #self.agent_plot = customtkinter.CTkButton(self.master, text="Agents on Plot", command=self.agent_on_plot)
        self.agent_plot.place(relx=0.86, rely=init+5*step, relwidth=0.125, relheight=0.035)
        
        self.env_summary = customtkinter.CTkButton(self.master, text="Simulation Summary", command=self.environment_summary)
        self.env_summary.place(relx=0.86, rely=init+5.5*step, relwidth=0.125, relheight=0.035)
        
        self.retrain_button = customtkinter.CTkButton(self.master, text="Retrain", command=self.retrain_agent)
        self.retrain_button.place(relx=0.86, rely=init+6*step, relwidth=0.125, relheight=0.035)
            
        self.textbox_agent = customtkinter.CTkTextbox(self.master)
        self.textbox_agent.place(relx=0.86, rely=init+7*step, relwidth=0.125, relheight=0.15)
        
        
        if self.auto_robust:
            self.retrain_button.configure(state="disabled")
            self.textbox_agent.insert("0.0", f"Automatic Robust Mode Activated")
        else:
            self.textbox_agent.insert("0.0", f"Manual Mode Activated")
            
        self.count_agent = 1
        self.textbox_agent.configure(state="disabled", wrap="word")
        
 
        
        self.structures = []
        self.simulate()
        
    def simulate(self):
        self.environment = GridEnv(reward="dose", sources = 100,
                 average_healthy_glucose_absorption = self.params[0],
                 average_cancer_glucose_absorption = self.params[1],
                 average_healthy_oxygen_consumption = self.params[2],
                 average_cancer_oxygen_consumption = self.params[3],
                 cell_cycle = self.params[6],
                 radiosensitivities=self.params[7])
    
        self.environment.reset()
        self.environment.go(steps=1)
        self.structures.append(self.environment)
        self.idx = 0
        self.speed = 250
        self.start_hour = None

        x_pos = 0.2
        self.fig, axs = plt.subplots(2, 3, figsize = (16,16))
        canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().place(relx = x_pos, rely = 0.025, relwidth=0.65, relheight=0.775)
        
        
        self.cell_plot = axs[0][0]
        self.cell_density_plot = axs[0][1]
        self.glucose_plot = axs[0][2]
    
        self.dose_plot   = axs[1][0]
        self.healthy_plot = axs[1][1]
        self.cancer_plot  = axs[1][2]
    
    
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # CELL DENSITY COLORBAR
        self.div = make_axes_locatable(self.cell_density_plot)
        self.cax = self.div.append_axes('right', '5%', '5%')
        data = np.zeros((self.environment.xsize, self.environment.ysize))
        im = self.cell_density_plot.imshow(data)
        self.cb = self.fig.colorbar(im, cax=self.cax)
        
        # GLUCOSE COLORBAR
        self.div2 = make_axes_locatable(self.glucose_plot)
        self.cax2 = self.div2.append_axes('right', '5%', '5%')
        data2 = np.zeros((self.environment.xsize, self.environment.ysize))
        im2 = self.glucose_plot.imshow(data2)
        self.cb2 = self.fig.colorbar(im2, cax=self.cax2)
    
        self.nb = self.environment.nb
        self.time_arr    = self.environment.time_arr
        self.healthy_arr = self.environment.healthy_arr
        self.cancer_arr  = self.environment.cancer_arr
        self.dose_arr    = self.environment.dose_arr
        self.total_dose = self.environment.total_dose
        self.glucose_arr = self.environment.glucose_arr
        self.oxygen_arr = self.environment.oxygen_arr
        self.grid_arr = self.environment.grid_arr
        self.density_arr = self.environment.density_arr
    
        # Simulation Buttons
        
        
        self.slider = customtkinter.CTkSlider(self.master, from_=0, to=1550, width=1100)
        self.slider.bind("<ButtonRelease-1>", self.move_slider)
        self.slider.place(relx=x_pos, rely=0.875, anchor='nw')
        self.label_slider = customtkinter.CTkLabel(self.master, text='Simulation Time:')
        self.label_slider.place(relx=x_pos-0.025, rely=0.85, anchor='w')
        
        self.slider_speed = customtkinter.CTkSlider(self.master, from_=1, to=5, width=1100, command=self.move_slider_speed)
        self.slider_speed.place(relx=x_pos, rely=0.95, anchor='nw')
        self.label_slider_speed = customtkinter.CTkLabel(self.master, text='Simulation Speed:')
        self.label_slider_speed.place(relx=x_pos-0.025, rely=0.925, anchor='w')
        
        
        self.slider_speed.set(3)
        
                
        self.agents = []
        self.agent_changes = []
        
        # call the update function after a delay
        self.update()
        
    def environment_summary(self):
        pass
        
        
    def focus_plot(self):
        self.focus = not self.focus
        self.update_plot(self.idx-1)
        if self.sidebar.sidebar_button_1.cget('text') =="Zoom":
            self.sidebar.sidebar_button_1.configure(text="Unzoom")
        else:
            self.sidebar.sidebar_button_1.configure(text="Zoom")
    
    def save_plot(self):
        self.save = True
        self.update_plot(self.idx)
        self.save = False
        
    def save_anim(self):
        self.anim = animation.FuncAnimation(self.fig, self.update_plot, 
							frames=len(self.structures), interval=100, repeat = False)
        
        self.anim.save('save/animated_env.gif', writer='imagemagick') 
      
    def pause_simulation(self):
        self.is_paused = not self.is_paused
        if self.sidebar.sidebar_button_4.cget('text') == "Pause":
            self.sidebar.sidebar_button_4.configure(text="Continue")
        else:
            self.sidebar.sidebar_button_4.configure(text="Pause")
        
    def stop_simulation(self):
        self.stop_event.set()
        self.is_running = False
    
    def quit_page(self):
        for i in self.master.winfo_children():
            i.destroy()
        self.master.quit()
        self.master.destroy()
        
    def load_q_table(self):
        
        if self.previous_name != self.name:
            self.previous_name = self.name
            self.agents.append(self.name)
            self.agent_changes.append(self.environment.time)
        
        try:
            self.q_table = np.load(self.path +  f'\\q_table_{int_from_str(self.path)}.npy', allow_pickle=False)
        except:
            self.q_table = np.load(self.path + f'\\q_table_{self.name}.npy', allow_pickle=False)
            
        self.textbox_agent.configure(state="normal")
        text = self.textbox_agent.get("0.0", "end")  
        self.textbox_agent.delete("0.0", "end")
        self.textbox_agent.insert("0.0", text + "\n" + f"New Agent : {self.name}")
        self.textbox_agent.configure(state="disabled")
    
    def choose_action(self, state):
        if self.name == "Baseline":
            return 1.0
        else:
            if self.q_table is None:
                self.load_q_table()
            
            actions = np.argwhere(self.q_table[state]==np.max(self.q_table[state])).flatten()
            return np.random.choice(actions)
            
    def update(self):
        if not self.stop_event.is_set():
            if not self.is_paused:
                if not self.environment.inTerminalState():
                    self.environment.go(steps=1)
                    self.slider.set(self.idx)
                    if (self.start_hour is not None):
                        if ((self.idx-(self.start_hour-24))%24 == 0):
                            print(self.environment.time)
                            state = self.environment.convert(self.environment.observe())
                            self.action = self.choose_action(state)
                            reward = self.environment.act(self.action)
                    elif (self.start_hour is None):
                        if (CancerCell.cell_count > 9000):
                            self.start_hour = self.environment.time
                            print(f'Start of radiotherapy at time {self.start_hour}')
                            print(f'Number of Cancer Cells : {CancerCell.cell_count}')
                            print(f'Number of Healthy Cells : {HealthyCell.cell_count}')
                        elif (self.environment.time > 326):
                            self.start_hour = 350
                            print(f'Start of radiotherapy at time 350')
                            print(f'Number of Cancer Cells : {CancerCell.cell_count}')
                            print(f'Number of Healthy Cells : {HealthyCell.cell_count}')
                        
                        if self.start_hour is not None:
                            self.textbox_agent.configure(state="normal")
                            text = self.textbox_agent.get("0.0", "end")  
                            self.textbox_agent.delete("0.0", "end")
                            self.textbox_agent.insert("0.0", text + "\n" + f"Radiotherapy starts at t = {self.start_hour}")
                            self.textbox_agent.configure(state="disabled")
                    
                    self.update_plot(self.idx)
                    
                else:
                    self.update_plot(self.idx)
                    self.idx -= 1
                    self.save_plot()
                    self.save_plot_hd()
                    self.stop_simulation()
                    print("End of simulation")
                    return

            self.master.after(self.speed, self.update)
        
    def update_plot(self, idx):
        
        self.time_arr    = self.environment.time_arr
        self.healthy_arr = self.environment.healthy_arr
        self.cancer_arr  = self.environment.cancer_arr
        self.dose_arr    = self.environment.dose_arr
        self.total_dose = self.environment.total_dose
        self.glucose_arr = self.environment.glucose_arr
        self.oxygen_arr = self.environment.oxygen_arr
        self.grid_arr = self.environment.grid_arr
        self.density_arr = self.environment.density_arr
        
        try:
            self.structures[idx] = self.environment
        except:
            self.structures.append(self.environment)

        
        self.plot_data(idx)
        
    def move_slider(self, event):
        tmp = self.speed
        self.speed = 0
        self.idx = int(self.slider.get())

        if self.idx > len(self.structures):
            self.idx = len(self.structures)
        self.slider.set(self.idx)
        self.environment = self.structures[self.idx-1]
        self.speed = tmp
        self.update_plot(self.idx)
        
    def move_slider_speed(self, value):
        self.slider_speed.set(int(value))
        lst = [1000, 500, 250, 100, 2]
        self.speed = lst[int(value)-1]
        
    def plot_data(self, i):
    
        self.fig.suptitle('Cell proliferation at t = ' + str(i))    
    
        # plot cells
        self.cell_plot.clear()
        self.cell_plot.imshow(self.grid_arr[i], cmap='coolwarm')
        self.cell_plot.set_xticks([])
        self.cell_plot.set_yticks([])
        self.cell_plot.set_title("Cells", fontsize=20, pad=10)
        
        # plot cell density
        self.cell_density_plot.clear()
        im = self.cell_density_plot.imshow(self.density_arr[i], cmap='coolwarm')
        self.fig.colorbar(im, cax=self.cax)
        self.cell_density_plot.set_xticks([])
        self.cell_density_plot.set_yticks([])
        self.cell_density_plot.set_title("Cell Density", fontsize=20, pad=10)
        
        # plot glucose
        self.cax2.cla()
        self.glucose_plot.clear()
        im2 = self.glucose_plot.imshow(self.glucose_arr[i], cmap='YlOrRd')
        self.fig.colorbar(im2, cax=self.cax2)
        self.glucose_plot.set_xticks([])
        self.glucose_plot.set_yticks([])
        self.glucose_plot.set_title("Glucose Concentration", fontsize=20, pad=10)
        
        # plot dose
        self.dose_plot.clear()
        self.dose_plot = axes_off(self.dose_plot)
        plt.text(0.5, 1.08, "Radiation Dose",
             horizontalalignment='center',
             fontsize=20,
             transform = self.dose_plot.transAxes)
        self.dose_plot.plot(self.time_arr[:i+1], self.dose_arr[:i+1])
        
        # plot healthy cells
        self.healthy_plot.clear()
        self.healthy_plot = axes_off(self.healthy_plot)
        plt.text(0.5, 1.08, "Healthy Cells",
             horizontalalignment='center',
             fontsize=20,
             transform = self.healthy_plot.transAxes)
        self.healthy_plot.plot(self.time_arr[:i+1], self.healthy_arr[:i+1], label="Healthy", color="b")
        
        # plot cancer cells
        self.cancer_plot.clear()
        self.cancer_plot = axes_off(self.cancer_plot)
        plt.text(0.5, 1.08, "Cancer Cells",
             horizontalalignment='center',
             fontsize=20,
             transform = self.cancer_plot.transAxes)
        self.cancer_plot.plot(self.time_arr[:i+1], self.cancer_arr[:i+1], label="Cancer", color="r")
        
        if (self.agent_on_plot_bool.get()) and (self.start_hour is not None):
            for idx, item in enumerate(self.agent_changes):
                x_max, y_max = self.time_arr[-1], np.max(self.cancer_arr[:self.start_hour])
                self.cancer_plot.axvline(x=item, color='b')
                plt.text(item+0.005*x_max, 0.95*y_max, f'{self.agents[idx]}', rotation=90, fontsize=16)
        
        if not self.focus:
            self.cancer_plot.set_xlim(0, self.time_arr[-1])
            self.healthy_plot.set_xlim(0, self.time_arr[-1])
            self.dose_plot.set_xlim(0, self.time_arr[-1])
        
        if self.save:
            plt.savefig("save/simulation.svg")
    
        self.idx += 1
        
        self.cancer_label.configure(text=f"Number of Cancer Cells \n \t {int(self.cancer_arr[i])} Cells")
        self.healthy_label.configure(text=f"Number of Healthy Cells \n \t {int(self.healthy_arr[i])} Cells")
        #self.dose_label.configure(text=f"Last dose administered \n \t {self.dose_arr[i]-self.dose_arr[i-1]} Gy")
        self.total_dose_label.configure(text=f"Total dose administered \n \t {self.dose_arr[i]} Gy")
        start_hour = next(idx for idx, item in enumerate(self.dose_arr) if item != 0) - 1
        if self.dose_arr[i] != 0:
            if (i-start_hour > 5):
                if self.params[9] == 'RF':
                    self.predict_classifier(self.environment.cancer_arr)
                else:
                    self.predict_cnn(self.action)
                    
                
        self.fig.canvas.draw()
        
    def predict_cnn(self, action):
        i = self.environment.time
        start_hour = self.environment.start_time
        def get_cnn():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = BinaryClassifier(3,3).to(device)
            if (i-start_hour) < 30:
                model.load_state_dict(torch.load('model\\CNN\\cnn_dose1_74%.pt'))
                return model
            elif (i-start_hour) < 50:
                model.load_state_dict(torch.load('model\\CNN\\cnn_dose2_77%.pt'))
                return model
            elif (i-start_hour) < 75:
                model.load_state_dict(torch.load('model\\CNN\\cnn_dose3_82%.pt'))
                return model
            elif (i-start_hour) < 100:
                model.load_state_dict(torch.load('model\\CNN\\cnn_dose4_86%.pt'))
                return model
            else:
                model.load_state_dict(torch.load('model\\CNN\\cnn_dose5_87%.pt'))
                return model

        model = get_cnn()
        output = model(torch.tensor(np.expand_dims(np.array(self.environment.grid_arr[i-1]).transpose(2,0,1), axis=0), dtype=torch.float32), torch.tensor(np.expand_dims([start_hour, action+1, self.environment.total_dose],axis=0), dtype=torch.float32))
        _, y_pred = torch.max(output,1)
        print(y_pred)
        
        if not y_pred: # Predict if tcp is 100.0 or not (binary classification)
            self.retrain_early.configure(text=f"Re-train Early : \n     Yes")
        else:
            self.retrain_early.configure(text=f"Re-train Early : \n     No")
        
        if self.auto_robust:
            if self.environment.count_dose == 2:
                self.environment.count_dose = 0
                self.name, self.path = self.auto_robust_agent.update_agent(y_pred, 
                                                                           100.0,
                                                                           100.0,
                                                                           self.name,
                                                                           self.path)
                if self.name != 'Baseline':
                    self.load_q_table()
        
    def predict_classifier(self, ccells):
        i = self.environment.time
        start_hour = self.environment.start_time
        if (i-start_hour < 50):
            with open('model\\RF\\classifier_5h_86.5%.pickle', 'rb') as file:
                clf = pickle.load(file)
        if (i-start_hour < 120):
            with open('model\\RF\\classifier_50h_91%.pickle', 'rb') as file:
                clf = pickle.load(file)
        if (i-start_hour < 150):
            with open('model\\RF\\classifier_120h_92.3%.pickle', 'rb') as file:
                clf = pickle.load(file)
        if (i-start_hour >= 150):
            with open('model\\RF\\classifier_150h_93.5%.pickle', 'rb') as file:
                clf = pickle.load(file)
        
        data = {"mean" : [],
                "median" : [],
                "max" : [],
                "min" : [],
                }
        
        def extract_feature(ccells):
            ccells = pd.Series(data=ccells)
            
            data["mean"].append(ccells.mean())
            data["median"].append(ccells.median())
            data["max"].append(ccells.max())
            data["min"].append(ccells.min())
            
        start_hour = next(idx for idx, item in enumerate(self.dose_arr) if item != 0) - 1
        end_hour   = next(idx for idx, item in enumerate(self.dose_arr) if np.isnan(item)) - 1
        init_ccells = self.cancer_arr[start_hour]

        percent_ccells = 100*self.cancer_arr[start_hour:end_hour]/init_ccells

        extract_feature(percent_ccells)
        data = pd.DataFrame.from_dict(data)
        y_pred = clf.predict(data)
        y_pred_proba = clf.predict_proba(data)
        
        if not y_pred[0]: # Predict if tcp is 100.0 or not (binary classification)
            self.retrain_early.configure(text=f"Re-train Early : \n     Yes ({100*np.max(y_pred_proba):.1f}%)")
        else:
            self.retrain_early.configure(text=f"Re-train Early : \n     No ({100*np.max(y_pred_proba):.1f}%)")
            
        with open('model/regressor.pickle', 'rb') as file:
                    reg = pickle.load(file)
                    
        def predict_with_confidence(rf, X):
            tree_predictions = np.array([tree.predict(X) for tree in rf.estimators_])
            mean_predictions = np.mean(tree_predictions, axis=0)
            percent_predictions = 100*sum(tree_predictions==mean_predictions)/len(tree_predictions)
            return mean_predictions, percent_predictions
        
        y_pred_tcp, y_pred_tcp_percent = predict_with_confidence(reg, data)
        self.predict_tcp.configure(text=f"Predicted TCP : \n    {y_pred_tcp[0]:.1f}%,    ({y_pred_tcp_percent[0]:.1f})")
        
        if self.auto_robust:
            if self.environment.count_dose == 2:
                self.environment.count_dose = 0
                self.name, self.path = self.auto_robust_agent.update_agent(y_pred[0], 
                                                                           y_pred_tcp[0],
                                                                           100*np.max(y_pred_proba),
                                                                           self.name,
                                                                           self.path)
                if self.name != 'Baseline':
                    self.load_q_table()
                
        
    def save_plot_hd(self):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        i = self.idx - 1
        font = 65
        
        fig, ax = plt.subplots(1, 1, figsize = (16,16))
        ax.imshow(self.grid_arr[i], cmap='coolwarm')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Cells \n", fontsize=font, pad=10)
        plt.savefig(f"save/{i}cells.svg")
        
        fig, ax = plt.subplots(1, 1, figsize = (16,16))
        ax = axes_off(ax)
        ax.set_xlabel("\n Time [hours]", fontsize=40)
        ax.set_ylabel("# Cancer Cells \n", fontsize=40)
        plt.text(0.5, 1.08, f"Cancer Cells after {i} hours",
             horizontalalignment='center',
             fontsize=font,
             transform = ax.transAxes)
        
        np.save("save/cancer_arr_high_nutrients", self.cancer_arr[:i+1])
        
        
        ax.plot(self.time_arr[:i+1], self.cancer_arr[:i+1], label="Cancer", color="r")
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.savefig(f"save/{i}_cancer_cells.svg")
        
        fig, ax = plt.subplots(1, 1, figsize = (16,16))
        ax = axes_off(ax)
        ax.set_xlabel("\n Time [hours]", fontsize=40)
        ax.set_ylabel("# Healthy Cells \n", fontsize=40)
        plt.text(0.5, 1.08, f"Healthy Cells after {i} hours",
             horizontalalignment='center',
             fontsize=font,
             transform = ax.transAxes)
        ax.plot(self.time_arr[:i+1], self.healthy_arr[:i+1], label="Healthy", color="b")
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.savefig(f"save/{i}_healthy_cells.svg")
        
        
        # CELL DENSITY
        fig, ax = plt.subplots(1, 1, figsize = (16,16))
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        data = np.zeros((self.environment.xsize, self.environment.ysize))
        im = ax.imshow(data)
        cb = fig.colorbar(im, cax=cax)
        
        
        im = ax.imshow(self.density_arr[i], cmap='coolwarm')
        cbar = fig.colorbar(im, cax=cax)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(40)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Cell Density \n", fontsize=font, pad=10)
        plt.savefig(f"save/{i}_density.svg")
        
        # GLUCOSE HEATMAP
        fig, ax = plt.subplots(1, 1, figsize = (16,16))
        div2 = make_axes_locatable(ax)
        cax2 = div2.append_axes('right', '5%', '5%')
        data2 = np.zeros((self.environment.xsize, self.environment.ysize))
        im2 = ax.imshow(data2)
        cb2 = fig.colorbar(im2, cax=cax2)
        
        
        cax2.cla()
        im2 = ax.imshow(self.glucose_arr[i], cmap='YlOrRd')
        cbar = fig.colorbar(im2, cax=cax2)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(40)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Glucose Concentration \n", fontsize=font, pad=10)
        plt.savefig(f"save/{i}glucose_concentration.svg")
        
    def retrain_agent(self):
        if not self.is_paused:
            self.pause_simulation()
        # Create a new Toplevel window
        self.popup = tk.Toplevel()
        self.popup.title("Modify Parameters")
        self.popup.geometry("1600x850")
        
        self.treatment_tab = TreatmentTab(master=self.popup)
    
        # Function to handle the submit button click event in the self.popup
        def submit():
            file_name = self.treatment_tab.combobox_1.get()
            file_list = pd.concat([list_agent(), list_agent2(), list_agent3()])
            path = file_list[file_list["name"]==file_name]["path"].iloc[0]
                    
            self.name = file_name
            self.path = path
            self.load_q_table()
            self.pause_simulation()
            self.popup.destroy()
    
        # Add submit button and bind it to the submit function
        submit_button = customtkinter.CTkButton(self.popup, text="Submit", command=submit)
        submit_button.place(relx=0.015, rely=0.05, relwidth=0.25, relheight=0.05)

    
def int_from_str(r):
    return ''.join(x for x in r if x.isdigit())

def axes_off(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.tick_params(axis='both', which='both', length=0)
  
  return ax




