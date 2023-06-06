# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:12:15 2022

@author: Florian Martin

Classification - Convolutionnal Neural Network

"""
                    
                    
import torch.nn as nn
import torch
import os
import numpy as np

                    
                    
class BinaryClassifier(nn.Module):
    def __init__(self, input_shape, num_additional_features):
        super().__init__()
        conv_layers = []
        
        # First Convolution Block
        self.conv1 = nn.Conv2d(input_shape, 8, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(8)
        conv_layers += [self.conv1, self.relu1, self.bn1]
        
        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.PReLU()
        self.bn2 = nn.BatchNorm2d(16)
        conv_layers += [self.conv2, self.relu2, self.bn2]
        
        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.PReLU()
        self.bn3 = nn.BatchNorm2d(32)
        conv_layers += [self.conv3, self.relu3, self.bn3]
        
        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.PReLU()
        self.bn4 = nn.BatchNorm2d(64)
        conv_layers += [self.conv4, self.relu4, self.bn4]
        
        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu5 = nn.PReLU()
        self.bn5 = nn.BatchNorm2d(128)
        conv_layers += [self.conv5, self.relu5, self.bn5]
        
        
        self.conv_fc = nn.Linear(in_features=128, out_features=128)
        
        # Dropout Layer
        self.drp = nn.Dropout2d(p=0.5)

        # Fully connected layer for additional information
        self.fc_additional = nn.Linear(in_features=num_additional_features, out_features=64)
        self.relu = nn.ReLU()

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=128 + 64, out_features=2)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

        self.initialize_weights()

    def forward(self, x, additional_info):
        x = self.conv(x)
        x = self.drp(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.conv_fc(x))
        


        additional_info = self.fc_additional(additional_info)
        additional_info = self.relu(additional_info)
        
        x = torch.cat((x, additional_info), dim=1)

        x = self.fc(x)

        return x

    def initialize_weights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight, a = 0.1)
                
                if m.bias is not None :
                    nn.init.constant_(m.bias, 0)
                    