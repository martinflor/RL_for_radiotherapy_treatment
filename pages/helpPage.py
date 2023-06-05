import customtkinter
import matplotlib.pyplot as plt
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

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

dir_path = os.path.dirname(os.path.realpath(__file__))

def int_from_str(r):
    return ''.join(x for x in r if x.isdigit())

class help_page(customtkinter.CTk):
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

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Reinforcement Learning \n and \n Radiotherapy", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Quit", width=190, fg_color="transparent", text_color=("gray10", "#DCE4EE"), border_width=2, command=self.quit)
        self.sidebar_button_4.grid(row=2, column=0, padx=10, pady=10)

        # EPL LOGO
        
        epl = customtkinter.CTkImage(light_image=Image.open("images/EPL.jpg"),
                                  dark_image=Image.open("images/EPL.jpg"),
                                  size=(150, 80))
        button_epl = customtkinter.CTkButton(self, text= '', 
                                                image=epl, fg_color='transparent')
        button_epl.place(relx=1, rely=1, anchor='se')
        
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
        
        # PROGRESS BAR
        
        self.progressbar_1 = customtkinter.CTkProgressBar(self)
        self.progressbar_1.place(relx= 0.17, rely=0.83, relwidth=0.82)
        self.progressbar_1.configure(mode="indeterminnate")
        self.progressbar_1.start()
        

        # TREATMENT
        
        self.tabview_combobox = customtkinter.CTkTabview(self, width=550, command=self.update_tabview_agent)
        self.tabview_combobox.place(relx= 0.17, rely=0.025, relwidth=0.82, relheight=0.8)
        
        self.tabview_combobox.add("Non-Robust")
        self.tabview_combobox.add("Robust Cell Cycle")
        self.tabview_combobox.add("Robust Radiosensitivity")
        
        self.tabview_tt = customtkinter.CTkTabview(self, width=550)
        self.tabview_tt.place(relx=0.45, rely=0.01, relwidth=0.55, relheight=.99)
        
        self.tabview_tt.add("Performances")
        self.tabview_tt.add("Agent's q-table")

        self.tabview_tt.tab("Performances").grid_columnconfigure(0, weight=1)  
        self.tabview_tt.tab("Agent's q-table").grid_columnconfigure(0, weight=1)
        
        values = self.list_agent()
        lst = [i for i, _ in values]
        
        self.combobox_label = customtkinter.CTkLabel(self, text="RL Agent:", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.combobox_label.place(relx=0.015, rely=0.05, relwidth=0.25, relheight=0.05)
        self.combobox_1 = customtkinter.CTkComboBox(self,
                                                    values=lst, command=self.update_description)

        self.combobox_1.place(relx=0.018, rely=0.22, relwidth=0.25, relheight=0.06)
        
        self.states_label = customtkinter.CTkLabel(self, text="Number of unexplored states by the agent : /", anchor="w", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.states_label.place(relx=0.015, rely=0.82, relwidth=0.35, relheight=0.05)
        
        self.str_agent = customtkinter.StringVar()
        self.checkbox_agent = customtkinter.CTkCheckBox(self, text="Only Best Agents", command=self.update_tabview_agent,
                                     variable=self.str_agent, onvalue="on", offvalue="off")
        self.checkbox_agent.place(relx=0.015, rely=0.88, relwidth=0.35, relheight=0.1)
        
        # Treatment : Performances
        
        self.fig_box, self.axes = plt.subplots(3,1, figsize=(24,20))
        self.fig_box.patch.set_alpha(0)
        canvas_box = FigureCanvasTkAgg(self.fig_box, master=self.tabview_tt.tab("Performances"))
        canvas_box.draw()
        canvas_box.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas_box.get_tk_widget().place(relx=0.01, rely=0.01, relwidth=0.99, relheight=0.99)
        
        # Treatment : Q-table
        
        self.fig_table, self.axes_table = plt.subplots(4, 1, constrained_layout=True, figsize = (16,12))
        self.fig_table.patch.set_alpha(0)
        canvas_table = FigureCanvasTkAgg(self.fig_table, master=self.tabview_tt.tab("Agent's q-table"))
        canvas_table.draw()
        canvas_table.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas_table.get_tk_widget().place(relx=0.01, rely=0.01, relwidth=0.99, relheight=0.99)

        self.update_description(1)
        
            
    def update_tabview_agent(self):
        tab = self.tabview_combobox.get()
        state = self.checkbox_agent.get()
        if tab == 'Non-Robust':
            values = self.list_agent()
            lst = [i for i, _ in values]
            if state == 'on':
                best_lst = [agent for agent in lst if (('SARSAgent 17' in agent) and ('Exp' not in agent)) or ('Baseline' in agent)]
                self.combobox_1.configure(values=best_lst)
            else:
                self.combobox_1.configure(values=lst)
        elif tab == 'Robust Cell Cycle':
            values = self.list_agent2()
            lst = [i for i, _ in values]
            if state == 'on':
                best_lst = [agent for agent in lst if ('16_cc_14' in agent) or ('18_cc_18' in agent) or ('20_cc_18' in agent)]
                self.combobox_1.configure(values=best_lst)
            else:
                self.combobox_1.configure(values=lst)
        elif tab == 'Robust Radiosensitivity':
            values = self.list_agent3()
            lst = [i for i, _ in values]
            self.combobox_1.configure(values=lst)
            
        self.update_description(1)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
        
    def open_github(self):
        webbrowser.open_new_tab('https://github.com/martinflor/master_thesis_RL')

    def open_linkedin(self):
        webbrowser.open_new_tab('https://www.linkedin.com/in/florian-martin-554350239/')
        
    def update_description(self, event):
        agent_name = self.combobox_1.get()  # get the selected agent name
        print(agent_name)
        self.description(self.tabview.tab("Treatment"), agent_name)
        

    def description(self, menu, file_name):
            
        self.agent_frame = customtkinter.CTkFrame(menu, fg_color='transparent')
        self.agent_frame.place(relx=0.01, rely=0.35, relwidth=0.4, relheight=0.45)
        
        tmp_dict = self.get_agent(dir_path + '\\TabularAgentResults\\results_baseline.pickle')
        tcp_baseline = tmp_dict["TCP"]
        fractions_baseline = (np.mean(tmp_dict["fractions"]), np.std(tmp_dict["fractions"]))
        doses_baseline = (np.mean(tmp_dict["doses"]), np.std(tmp_dict["doses"]))
        duration_baseline = (np.mean(tmp_dict["duration"]), np.std(tmp_dict["duration"]))
        survival_baseline = (np.mean(tmp_dict["survival"]), np.std(tmp_dict["survival"]))
        
        path = None
        
        if file_name == 'Baseline':
            tcp = tcp_baseline
            fractions = fractions_baseline
            doses = doses_baseline
            duration = duration_baseline
            survival = survival_baseline
        else:
            file_list = self.list_agent() + self.list_agent2() + self.list_agent3()
            names = [x[0] for x in file_list]
            idx = names.index(file_name)
            
            name, path = file_list[idx]
            try:
                tmp_dict = self.get_agent(path + f'\\results_{int_from_str(path)}.pickle')
                path_q_table = path + f'\\q_table_{int_from_str(path)}'
            except:
                tmp_dict = self.get_agent(path + f'\\results_{name}.pickle')
                path_q_table = path + f'\\q_table_{name}'
                
                
            
            tcp = tmp_dict["TCP"]
            fractions = (np.mean(tmp_dict["fractions"]), np.std(tmp_dict["fractions"]))
            doses = (np.mean(tmp_dict["doses"]), np.std(tmp_dict["doses"]))
            duration = (np.mean(tmp_dict["duration"]), np.std(tmp_dict["duration"]))
            survival = (np.mean(tmp_dict["survival"]), np.std(tmp_dict["survival"]))
    
        # create table headers
        headers = ['', file_name, 'Baseline']
            
        # create table rows
        
        data = [('TCP', f"{tcp}", f"{tcp_baseline}"), 
                ('Fractions', f"{fractions[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{fractions[1]:.3f}",
                 f"{fractions_baseline[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{fractions_baseline[1]:.3f}"), 
                ('Doses', f"{doses[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{doses[1]:.3f}",
                 f"{doses_baseline[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{doses_baseline[1]:.3f}"), 
                ('Duration', f"{duration[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{duration[1]:.3f}",
                 f"{duration_baseline[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{duration_baseline[1]:.3f}"),
                ('Survival', f"{survival[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{survival[1]:.3f}",
                 f"{survival_baseline[0]:.3f}" + ' ' + u"\u00B1" + ' ' + f"{survival_baseline[1]:.3f}")]

    
        padx_value = 30
        for j, header in enumerate(headers):
                    label = customtkinter.CTkLabel(self.agent_frame, text=header, font=('Arial', 18))
                    label.grid(row=0, column=j*4, padx=padx_value, pady=5)
        
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                label = customtkinter.CTkLabel(self.agent_frame, text=cell, font=('Arial', 14))
                label.grid(row=i+1, column=j*4, padx=padx_value, pady=5)
                
        self.boxplot_agent(tmp_dict["fractions"], tmp_dict["duration"], tmp_dict["survival"], file_name)
        if file_name != 'Baseline':
            self.q_table_agent(path_q_table)

                
    def get_agent(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def list_agent(self):
        
        lst = [('Baseline', '')]
        filename = dir_path + "\\TabularAgentResults\\"
        list_dir = [(f.name, f.path) for f in os.scandir(filename) if f.is_dir()]
        
        for name, path in list_dir:
            subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
            for i in range(len(subfolders)):
                lst.append((name + ' ' + int_from_str(subfolders[i]), subfolders[i]))
                
        return lst
    
    def list_agent2(self):
        
        filename = dir_path + "\\TabularAgentRobustCellCycle\\"
        list_dir = [(f.name, f.path) for f in os.scandir(filename) if f.is_dir()]
                
        return list_dir
    
    def list_agent3(self):
        
        filename = dir_path + "\\TabularAgentRobustRadio\\"
        list_dir = [(f.name, f.path) for f in os.scandir(filename) if f.is_dir()]
                
        return list_dir
    
    def q_table_agent(self, path):
        
        def get_q_color(value, vals):
            if all(x==max(vals) for x in vals):
                return "grey", 0.5
            if value == max(vals):
                return "green", 1.0
            else:
                return "red", 0.3

        q_table = np.load(path + '.npy', allow_pickle=False)
        
        self.axes_table[0].clear()
        self.axes_table[1].clear()
        self.axes_table[2].clear()
        self.axes_table[3].clear()
        
        self.axes_table[0].set_title("Action 1 : 1 Gray")
        self.axes_table[1].set_title("Action 2 : 2 Grays")
        self.axes_table[2].set_title("Action 3 : 3 Grays")
        self.axes_table[3].set_title("Action 4 : 4 Grays")
        
        count = 0
        for x, x_vals in enumerate(q_table):
                for y, y_vals in enumerate(x_vals):
                    self.axes_table[0].scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
                    self.axes_table[1].scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
                    self.axes_table[2].scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])
                    self.axes_table[3].scatter(x, y, c=get_q_color(y_vals[3], y_vals)[0], marker="o", alpha=get_q_color(y_vals[3], y_vals)[1])
                
                    if all(x==y_vals[0] for x in y_vals):
                        count += 1
                        
        self.states_label.configure(text=f"Number of unexplored states : {count}")
        self.fig_table.canvas.draw()
        
    def boxplot_agent(self, fractions, duration, survival, name):
        
        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[2].clear()
        
        self.fig_box.suptitle(name)

        # Create a DataFrame for each list
        data_fractions = pd.DataFrame({"Values": fractions})
        data_fractions["Type"] = "Fractions [-]"
        
        data_duration = pd.DataFrame({"Values": duration})
        data_duration["Type"] = "Duration \n [hours]"
        
        data_survival = pd.DataFrame({"Values": survival})
        data_survival["Type"] = "Survival [-]"
        
        # Combine the three DataFrames
        data = pd.concat([data_fractions, data_duration, data_survival], ignore_index=True)
        
        # Create the subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
        
        # Loop through each subplot and create the boxplot with scatter points
        for i, data_type in enumerate(["Fractions [-]", "Duration \n [hours]", "Survival [-]"]):
            sns.boxplot(x="Values", y="Type", orient='h', data=data[data["Type"] == data_type], ax=self.axes[i], palette="Set2", width=0.5)
            sns.stripplot(x="Values", y="Type", orient='h', data=data[data["Type"] == data_type], ax=self.axes[i], color=".25")
            self.axes[i].set_ylabel("")

        self.fig_box.canvas.draw()
    
    def quit_page(self):
        self.quit()
        self.destroy()