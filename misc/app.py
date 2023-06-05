import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk  
from tkinter import ttk
from tkinter import *
from tkinter.ttk import *
import sv_ttk
import numpy as np
from scipy.stats import norm
import statistics
import threading
import time
import pickle
from environment import GridEnv
import os

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


class Application(tk.Frame):
    def __init__(self, master=None):
        self.master = master
        self.master.geometry("1440x600+0+0")
        #self.master.attributes('-fullscreen', True)
        self.master.bind('<Escape>', lambda event: self.quit_page())
    
        self.fields = ('Average healthy glucose absorption', 'Average cancer glucose absorption',
                  'Average healthy oxygen consumption', 'Average cancer oxygen consumption')
        
        self.default = [.36, .54, 20, 20]
        self.welcome()
    
        
    def welcome(self):
        for i in self.master.winfo_children():
            i.destroy()
            
        tk.Frame.__init__(self, self.master)
        
        self.entries_xy = (25, 15)
        ents=self.makeform()
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        self.plot_data(self.default)
        
        canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().place(relx = 0.45, rely = 0.05, relwidth=0.5, relheight=0.55)
        
        quit_button = tk.Button(self.master, fg = 'black', background='white', text='Quit', width=25, command=self.quit_page)
        quit_button.place(relx=0.0, rely=1.0, anchor='sw')
    
        simulation_button = tk.Button(self.master, width=25, text='Simulation', command=self.simulation)
        simulation_button.place(relx=1.0, rely=1.0, anchor='se')
        
        self.entries = ents
        for field in self.fields:
            self.entries[field].bind('<KeyRelease>', self.update_plot)
            
        self.table = self.make_table(self.default)
        self.table.place(relx = 0.45, rely = 0.65, relwidth=0.5, relheight=0.3)
        
        
        self.var = tk.IntVar()
        self.radio = tk.Checkbutton(self.master, text="Radiation", variable=self.var, command=self.show_menu, font=('Arial', 18, 'bold'))
        self.radio.place(relx = 0.05, rely = 0.25, relwidth=0.1, relheight=0.1)
       
    def list_agent(self):
        
        lst = [('Baseline', '')]
        filename = dir_path + "\\TabularAgentResults\\"
        list_dir = [(f.name, f.path) for f in os.scandir(filename) if f.is_dir()]
        
        for name, path in list_dir:
            subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
            for i in range(len(subfolders)):
                lst.append((name + ' ' + int_from_str(subfolders[i]), subfolders[i]))
                
        return lst
        
    def show_menu(self):
        if self.var.get():
            menu = tk.Frame(self.master)
            menu.place(relx=0.05, rely=0.35, relwidth=0.15, relheight=0.45)
            
            relative_x, relative_y = 0.15, 0.05
            title_label = tk.Label(menu, text="Select a file", font=("Arial", 13, "bold"), padx=10, pady=10)
            title_label.place(relx=relative_x, rely=relative_y, relwidth=0.750, relheight=0.10)
            
            file_list = self.list_agent()
            
            listbox = tk.Listbox(menu)
            for file, _ in file_list:
                listbox.insert(tk.END, file)
            listbox.place(relx=relative_x+0.15, rely=relative_y+0.1, relwidth=0.5, relheight=0.5)

            scrollbar = tk.Scrollbar(menu, orient="vertical")
            scrollbar.config(command=listbox.yview)
            scrollbar.place(relx=relative_x, rely=relative_y+0.1, relwidth=0.05, relheight=0.5)

            listbox.config(yscrollcommand=scrollbar.set)
            
            confirm_button = tk.Button(menu, text="Confirm", command=lambda: self.select_file(listbox.get(tk.ACTIVE), menu))
            confirm_button.place(relx=0, rely=0.8, anchor='sw')      
            button = tk.Button(menu, text="Close", command=lambda: self.quit_menu(menu))
            button.place(relx=1.0, rely=.8, anchor='se')
            
            show_button = tk.Button(menu, text="Description", command=lambda: self.description(listbox.get(tk.ACTIVE)))
            show_button.place(relx=0.68, rely=.8, anchor='se')
            
            
            self.selected_file = tk.StringVar()
            self.selected_file_label = tk.Label(menu, text="No file has been selected", font=("Arial", 12), pady=10)
            self.selected_file_label.place(relx=1, rely=.925, anchor='se')
    
    def description(self, file_name):
        
        self.agent_frame = tk.Frame(self.master)
        self.agent_frame.place(relx=0.25, rely=0.4, relwidth=0.15, relheight=0.45)
        
        if file_name == 'Baseline':
            tcp = 100.0
            fractions = (34.9, 3.5623026261113755)
            doses = (69.8, 7.124605252222751)
            duration = (837.6, 85.495263026673)
            survival = (0.9873456049228684, 0.011230605729865561)
        else:
            file_list = self.list_agent()
            for name, path_ in file_list:
                if file_name == name:
                    path = path_
                
            tmp_dict = self.get_agent(path + f'\\results_{int_from_str(path)}.pickle')
            tcp = tmp_dict["TCP"]
            fractions = (np.mean(tmp_dict["fractions"]), np.std(tmp_dict["fractions"]))
            doses = (np.mean(tmp_dict["doses"]), np.std(tmp_dict["doses"]))
            duration = (np.mean(tmp_dict["duration"]), np.std(tmp_dict["duration"]))
            survival = (np.mean(tmp_dict["survival"]), np.std(tmp_dict["survival"]))

        # create table headers
        headers = ['Parameter', 'Value']
        for col, header in enumerate(headers):
            label = tk.Label(self.agent_frame, text=header, font=('Arial', 12, 'bold'))
            label.grid(row=0, column=col, padx=5, pady=5, sticky='w')
            
        # create table rows
        
        rows = [('TCP', f"{tcp}"), 
                ('Fractions', f"{fractions[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{fractions[1]:.3f}"), 
                ('Doses', f"{doses[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{doses[1]:.3f}"), 
                ('Duration', f"{duration[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{duration[1]:.3f}"),
                ('Survival', f"{survival[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{survival[1]:.3f}")]
        
        for row, data in enumerate(rows):
            for col, item in enumerate(data):
                label = tk.Label(self.agent_frame, text=item, font=('Arial', 12))
                label.grid(row=row+1, column=col, padx=5, pady=5, sticky='w')
        
    def get_agent(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)
        
    
    def select_file(self, file_name, menu):
        self.selected_file.set(file_name)
        self.selected_file_label.config(text="Selected file: " + file_name)
        
    def makeform(self):
        entries = {}
        x, y = 25, 15
        form_frame = tk.Frame(self.master)
        form_frame.place(relx=0.05, rely=0.05, relwidth=0.4, relheight=0.9)
        for idx, field in enumerate(self.fields):
            row = tk.Frame(form_frame)
            lab = tk.Label(row, width=40, text=field+": ", anchor='w', font=('Arial', 12, 'bold'))
            ent = tk.Entry(row)
            ent.insert(1, str(self.default[idx]))
            row.pack(side=tk.TOP, pady=5)
            lab.pack(side=tk.LEFT)
            ent.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            entries[field] = ent
        return entries

        
    def get_values(self):
        values = []
        for field in self.fields:
            values.append(float(self.entries[field].get()))
        return values
    
    def plot_data(self, values):
        self.ax1.clear()
        self.ax2.clear()
        
        f1 = lambda x, y : min(x,y)
        f2 = lambda x,y : max(x,y) 
        
        # Plot between -10 and 10 with .001 steps.
        x_axis = np.arange(-0.5, 2.5, 0.0001)
          
        # Calculating mean and standard deviation
        mean = 1
        sd = 1/3
        
        normal = norm.pdf(x_axis, mean, sd)
        results = []
        
        for i in range(len(x_axis)):
            results.append(max(0, min(2, normal[i])))
          
        self.ax1.plot(x_axis, np.array(results)*values[1],  label="Cancer Cells")
        self.ax1.plot(x_axis, np.array(results)*values[0], label="Healthy Cells")
        self.ax1.set_ylabel("Glucose Absorption")
        self.ax1.grid(alpha=0.5)
        self.ax1.legend()
        
        self.ax2.plot(x_axis, np.array(results)*values[3],  label="Cancer Cells")
        self.ax2.plot(x_axis, np.array(results)*values[2], label="Healthy Cells")
        self.ax2.set_ylabel("Oxygen Consumption")
        self.ax2.grid(alpha=0.5)
        self.ax2.legend()
        self.fig.canvas.draw()
        
    def make_table(self, values):
        table_frame = tk.Frame(self.master)
        table_frame.grid(row=len(self.fields)+1, column=1, padx=10, pady=10, sticky='n')
        
        # create table headers
        headers = ['Parameter', 'Value']
        for col, header in enumerate(headers):
            label = tk.Label(table_frame, text=header, font=('Arial', 12, 'bold'))
            label.grid(row=0, column=col, padx=5, pady=5, sticky='w')
            
        # create table rows
        rows = [('Quiescent Glucose Level', f"{2*24*values[0]:.3f}"), 
                ('Quiescent Oxygen Level', f"{2*24*values[2]:.3f}"), 
                ('Critical Glucose Level', f"{(3/4)*24*values[0]:.3f}"), 
                ('Critical Oxygen Level', f"{(3/4)*24*values[2]:.3f}")]
        for row, data in enumerate(rows):
            for col, item in enumerate(data):
                label = tk.Label(table_frame, text=item, font=('Arial', 12))
                label.grid(row=row+1, column=col, padx=5, pady=5, sticky='w')
                
        return table_frame
    
    def update_plot(self, event):
        self.master.after(500, self.plot_data, self.get_values())
        self.update_table(self.get_values())
    
    def update_table(self, values):
        # delete the children of the table_frame widget
        for widget in self.table.winfo_children():
            widget.destroy()
            
        # create table headers
        headers = ['Parameter', 'Value']
        for col, header in enumerate(headers):
            label = tk.Label(self.table, text=header, font=('Arial', 12, 'bold'))
            label.grid(row=0, column=col, padx=5, pady=5, sticky='w')
            
        # create table rows with updated values
        rows = [('Quiescent Glucose Level', f"{2*24*values[0]:.3f}"), 
                ('Quiescent Oxygen Level', f"{2*24*values[2]:.3f}"), 
                ('Critical Glucose Level', f"{(3/4)*24*values[0]:.3f}"), 
                ('Critical Oxygen Level', f"{(3/4)*24*values[2]:.3f}")]
        for row, data in enumerate(rows):
            for col, item in enumerate(data):
                label = tk.Label(self.table, text=item, font=('Arial', 12))
                label.grid(row=row+1, column=col, padx=5, pady=5, sticky='w')
                
    def quit_menu(self, menu):
        self.var.set(0)
        try:
            self.agent_frame.destroy()
        except:
            pass
        
        menu.destroy()

    def quit_page(self):
        self.master.quit()
        self.master.destroy()
        
    def simulation(self):
        if self.var.get():
            file_name = self.selected_file.get()
            file_list = self.list_agent()
            for name, path_ in file_list:
                if file_name == name:
                    path = path_
                    
            params = self.get_values() + [self.var.get(), file_name, path]
        else:
            params = self.get_values()
        self.simulation = SimulationPage(self.master, params)
        
        
        
        
        
        
        

class SimulationPage(tk.Frame):
    def __init__(self, master=None, params=None):
        self.master = master
        self.stop_event = threading.Event()
        self.is_paused = False  
        self.q_table = None
        
        for i in self.master.winfo_children():
            i.destroy()
            
        self.params = params
        if len(self.params) > 4:
            self.rad = True
            self.name = self.params[5]
            self.path = self.params[6]
        else:
            self.rad = False
            
        self.master.geometry("1440x600+0+0")
        
        self.structures = []
        
        self.simulate()
        
    def simulate(self):
        self.environment = GridEnv(reward="dose", sources = 100,
                 average_healthy_glucose_absorption = self.params[0],
                 average_cancer_glucose_absorption = self.params[1],
                 average_healthy_oxygen_consumption = self.params[2],
                 average_cancer_oxygen_consumption = self.params[3])
    
        self.environment.reset()
        self.environment.go(steps=1)
        self.idx = 0
        self.speed = 250
        
        self.structures.append(self.environment)
    
        self.fig, axs = plt.subplots(2, 3, figsize = (16,12))
        canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().place(relx = 0.1, rely = 0.025, relwidth=0.75, relheight=0.775)
    
        self.cell_plot = axs[0][0]
        self.cell_density_plot = axs[0][1]
        self.glucose_plot = axs[0][2]
        #self.cellular_model = axs[0][3]
    
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
        data = np.zeros((self.environment.xsize, self.environment.ysize))
        im = self.glucose_plot.imshow(data)
        self.cb2 = self.fig.colorbar(im, cax=self.cax2)
    
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
    
        incr = 0.05
        init_y = 0.77
    
        # add button to go back to Application page
        back_button = tk.Button(self.master, fg = 'black', background='white', text='Back to Application', width=20, command=self.go_back)
        back_button.place(relx=0, rely=0)
        
        quit_button = tk.Button(self.master, fg = 'black', background='white', text='Quit', width=20, command=self.quit_page)
        quit_button.place(relx=1.0, rely=0.0, anchor='ne')
        
        self.focus_button = tk.Button(self.master, fg = 'black', background='white', width=20, text='Zoom', command=self.focus_plot)
        self.focus_button.place(relx=0.86, rely=init_y-3*incr, anchor='nw')
        self.focus = False
        
        self.save_plot_button = tk.Button(self.master, fg = 'black', background='white', width=20, text='Save Env.', command=self.save_env)
        self.save_plot_button.place(relx=0.86, rely=init_y-2*incr, anchor='nw')
        
        self.save_plot_button = tk.Button(self.master, fg = 'black', background='white', width=20, text='Save Plot', command=self.save_plot)
        self.save_plot_button.place(relx=0.86, rely=init_y-incr, anchor='nw')
        self.save = False
        
        self.unpause_button = tk.Button(self.master, fg = 'black', width=20, background='white', text='Pause', command=self.pause_simulation)
        self.unpause_button.place(relx=0.86, rely=init_y, anchor='nw')
        
        # Simulation Buttons
        
        self.slider = tk.Scale(self.master, from_=0, to=1550, length=1100, fg = 'black', background='white',
                               orient=tk.HORIZONTAL, label="Simulation Time")
        self.slider.bind("<ButtonRelease-1>", self.move_slider)
        self.slider.place(relx=0.15, rely=0.85, anchor='nw')
        
        self.slider_speed = tk.Scale(self.master, from_=1, to=5, length=1100, fg = 'black', background='white',
                               orient=tk.HORIZONTAL, label="Simulation Speed", command=self.move_slider_speed)
        self.slider_speed.place(relx=0.15, rely=0.925, anchor='nw')
        
        self.slider_speed.set(3)
        
        # call the update function after a delay
        self.update()
        
        
    def focus_plot(self):
        self.focus = not self.focus
        self.update_plot(self.idx)
        if self.focus_button["text"] =="Zoom":
            self.focus_button.configure(text="Unzoom")
        else:
            self.focus_button.configure(text="Zoom")
    
    def save_plot(self):
        self.save = True
        self.update_plot(self.idx)
        self.save = False
        
    def save_env(self):
        with open("env_test.pickle", 'wb') as file_env:
            pickle.dump(self.structures, file_env)
            
    def load_env(self):
        with open('env_test.pickle', 'rb') as file:
                tmp_dict = pickle.load(file)
      
    def pause_simulation(self):
        self.is_paused = not self.is_paused
        if self.unpause_button["text"] == "Pause":
            self.unpause_button.configure(text="Continue")
        else:
            self.unpause_button.configure(text="Pause")
        
    def stop_simulation(self):
        self.stop_event.set()
        self.is_running = False

    def go_back(self):
        self.stop_simulation()
        self.back_page = Application(master=self.master)
    
    def quit_page(self):
        for i in self.master.winfo_children():
            i.destroy()
        self.master.quit()
        self.master.destroy()
        
    def load_q_table(self):
        self.q_table = np.load(self.path +  f'\\q_table_{int_from_str(self.path)}.npy', allow_pickle=False)
    
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
                    if (self.rad) and (self.idx > 350) and ((self.idx-326)%24 == 0):
                        state = self.environment.convert(self.environment.observe())
                        action = self.choose_action(state)
                        reward = self.environment.act(action)
                    self.update_plot(self.idx)

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
        self.idx = self.slider.get()

        if self.idx > len(self.structures):
            self.idx = len(self.structures)
        self.slider.set(self.idx)
        self.environment = self.structures[self.idx-1]
        self.speed = tmp
        
    def move_slider_speed(self, value):
        self.slider_speed.set(int(value))
        lst = [1000, 500, 250, 100, 2]
        self.speed = lst[int(value)-1]
        
    def plot_data(self, i):
        
        def axes_off(ax):
          ax.spines['top'].set_visible(False)
          ax.spines['right'].set_visible(False)
          ax.spines['bottom'].set_visible(False)
          ax.spines['left'].set_visible(False)
          ax.tick_params(axis='both', which='both', length=0)
          
          return ax
    
        self.fig.suptitle('Cell proliferation at t = ' + str(i))    
    
        # plot cells
        self.cell_plot.clear()
        self.cell_plot.set_title("Cells")
        self.cell_plot.imshow(self.grid_arr[i], cmap='coolwarm')
        self.cell_plot.set_xticks([])
        self.cell_plot.set_yticks([])
        
        # plot cell density
        self.cax.cla()
        self.cell_density_plot.clear()
        self.cell_density_plot.set_title("Cell Density")
        im = self.cell_density_plot.imshow(self.density_arr[i], cmap='coolwarm')
        self.fig.colorbar(im, cax=self.cax)
        self.cell_density_plot.set_xticks([])
        self.cell_density_plot.set_yticks([])
        
        # plot glucose
        self.cax2.cla()
        self.glucose_plot.clear()
        self.glucose_plot.set_title("Glucose Concentration")
        im2 = self.glucose_plot.imshow(self.glucose_arr[i], cmap='YlOrRd')
        self.fig.colorbar(im2, cax=self.cax2)
        self.glucose_plot.set_xticks([])
        self.glucose_plot.set_yticks([])
        
        # plot dose
        self.dose_plot.clear()
        self.dose_plot = axes_off(self.dose_plot)
        self.dose_plot.set_title("Radiation Dose")
        self.dose_plot.plot(self.time_arr[:i+1], self.dose_arr[:i+1])
        
        # plot healthy cells
        self.healthy_plot.clear()
        self.healthy_plot = axes_off(self.healthy_plot)
        self.healthy_plot.set_title("Healthy Cells")
        self.healthy_plot.plot(self.time_arr[:i+1], self.healthy_arr[:i+1], label="Healthy", color="b")
        
        # plot cancer cells
        self.cancer_plot.clear()
        self.cancer_plot = axes_off(self.cancer_plot)
        self.cancer_plot.set_title("Cancer Cells")
        self.cancer_plot.plot(self.time_arr[:i+1], self.cancer_arr[:i+1], label="Cancer", color="r")
        
        if not self.focus:
            self.cancer_plot.set_xlim(0, self.time_arr[-1])
            self.healthy_plot.set_xlim(0, self.time_arr[-1])
            self.dose_plot.set_xlim(0, self.time_arr[-1])
        
        if self.save:
            plt.savefig("test.svg")
    
        self.idx += 1
        self.fig.canvas.draw()
    
def int_from_str(r):
    return ''.join(x for x in r if x.isdigit())


if __name__ == '__main__':
    root=tk.Tk()
    #sv_ttk.set_theme("dark")
    app=Application(master=root)
    app.mainloop()


