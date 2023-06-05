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


from model.environment import GridEnv
from model.cell import HealthyCell, CancerCell, OARCell, Cell
from pages.helpPage import help_page
from pages.agent import *
import warnings
warnings.filterwarnings("ignore")

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class auto_robust_agent_selection:
    
    def __init__(self, agent_name):
        
        self.agents = {1 : ("17", dir_path + "\\TabularAgentResults\\SARSAgent\\17"),
                       2 : ("20_cc_19", dir_path + "\\TabularAgentRobustCellCycle\\20_cc_19"),
                       3 : ("16_cc_14", dir_path + "\\TabularAgentRobustCellCycle\\16_cc_14")}
        
        self.current_stage = 0
        

        
    def update_agent(self, predicted_class, predicted_tcp, confidence_class, confidence_tcp, current_name, current_path):
        
        if not predicted_class and confidence_class > 59.0:
            if self.current_stage == 0:
                if predicted_tcp > 85.0:
                    self.current_stage = 1
                else:
                    self.current_stage = 2
            elif self.current_stage == 1:
                if predicted_tcp > 85.0:
                    self.current_stage = 2
                else:
                    self.current_stage = 3
            elif self.current_stage == 2:
                self.current_stage = 3
    
            return self.agents[self.current_stage]
        else:
            return (current_name, current_path)

            