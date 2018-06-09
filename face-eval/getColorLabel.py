#!/usr/bin/env python
import pylab
import numpy as np
import os
colorCounter = 8
color_list = pylab.cm.Set1(np.linspace(0, 1, 26))

def getColorLabel(name):
    print (name)
    global colorCounter, color_list
    if name.find("MTCNN_afw") != -1:
        color = 'black'
        label = "MTCNN_afw"
    elif name.find("SFD_afw") != -1:
        color = 'r'
        label = "SFD_afw"
    elif name.find("SSD_afw") != -1:
        color = 'b'
        label = "SSD_afw"
    elif name.find("MTCNN_pascal") != -1:
        color = 'r'
        label = "MTCNN_pascal"
    elif name.find("SFD_pascal") != -1:
        color = color_list[2]
        label = "SFD_pascal"
    elif name.find("SSD_pascal") != -1:
        color = color_list[3]
        label = "SSD_pascal"
    else:
        color = color_list[colorCounter]
        colorCounter = colorCounter + 2
        label = os.path.splitext(os.path.basename(name))[0]
        label = label.replace("_", " ")
    return [color, label]
