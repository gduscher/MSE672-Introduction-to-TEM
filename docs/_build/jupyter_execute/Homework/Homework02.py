#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **Chapter 2: [Introduction](../Introduction/CH1_00-Introduction.ipynb)** </font>
# 
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # Homework 2
# 
# <font size = "5"> Reading Microscopy Data</font>
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Homework/Homework02.ipynb)
# 
# part of 
# 
# <font size = "5"> **[MSE672:  Introduction to Transmission Electron Microscopy](../_MSE672_Intro_TEM.ipynb)**</font>
# 
# by Gerd Duscher, Spring 2023
# <br>
# Microscopy Facilities<br>
# Institute of Advanced Materials & Manufacturing<br>
# Materials Science & Engineering<br>
# The University of Tennessee, Knoxville
# 
# Background and methods to analysis and quantification of data acquired with transmission electron microscopes.
# 

# ## Load Packages
# 
# First we need to load the libraries we want to use. 
# Here we use:
# - numpy: numerical library
# - matplotlib: graphic library
# - pyTEMlib: TEM microsocpy library
# 
# All of those packages except pyTEMlib are provided by annaconda.
# ### Check Installed Packages

# In[ ]:


import sys
from pkg_resources import get_distribution, DistributionNotFound

def test_package(package_name):
    """Test if package exists and returns version or -1"""
    try:
        version = get_distribution(package_name).version
    except (DistributionNotFound, ImportError):
        version = '-1'
    return version

if test_package('pyTEMlib') < '0.2023.2.0':
    print('installing pyTEMlib')
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib -q')
# ------------------------------
print('done')


# ### Load the plotting and pyTEMlib packages

# In[ ]:


get_ipython().run_line_magic('matplotlib', ' notebook')
import matplotlib.pylab as plt
import numpy
get_ipython().run_line_magic('gui', 'qt')

import pyTEMlib
import pyTEMlib.file_tools  as ft     # File input/ output library


# For archiving reasons it is a good idea to print the version numbers out at this point
print('pyTEM version: ',pyTEMlib.__version__)
__notebook__='CH1_04-Reading_File'
__notebook_version__='2023_02_10'


# ## Open a file 
# 
# This function opens a hfd5 file in the pyNSID style which enables you to keep track of your data analysis.
# 
# Please see the **[Installation](../Introduction/CH1_02-Prerequisites.ipynb#TEM-Library)** notebook for installation.
# 
# 
# Please note that the plotting routine of ``matplotlib`` was introduced in **[Matplotlib and Numpy for Micrographs](../Introduction/CH1_03-Data_Representation.ipynb)** notebook.
# 
# In the first lab we used a sample with a carbon grid of a periodicity of 500 nm
# 
# Download your images from the [google drive for 2023 Lab Data](https://drive.google.com/drive/folders/1RHcwbrBsxxJB_5cGgsNKPKAXcw2Y5GAo?usp=share_link)
# 
# > You must log into Google with your UTK account to be able to read these data.
# >
# 
# Go to the folder of you data and select one

# In[ ]:


datasets = ft.open_file()
main_dataset = datasets['Channel_000']
view = main_dataset.plot()


# ## Determination of Magnification  
# 
# We plot the image in pixels and there will be a line to select the length of a feature.
# 
# Any Mouseclick will extend the line from the last click

# In[ ]:


fig = plt.figure()

plt.imshow(main_dataset.T)
plt.title(main_dataset.title)
ax = plt.gca()
start_x = 0

fixed_line = False
line = plt.plot([0, 100],[0,200], color = 'orange')


def on_click(event):
   
    if event.inaxes:
        (start_x, end_x), (start_y, end_y) = line[0].get_data()
        start_x = end_x
        start_y = end_y
        end_x = event.xdata
        end_y = event.ydata

        line[0].set_data([start_x, end_x],[start_y, end_y])

        plt.draw()
# mouse_reference = plt.connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_press_event', on_click)


# ## Length and Angle of Line 

# In[ ]:


line_coordinates = np.array(line[0].get_data()).T
vector = line_coordinates[0]-line_coordinates[1]
print(f' The line is {np.linalg.norm(vector):.2f}pixels long')
print(f' The angle is {np.degrees(np.arctan2(vector[1], vector[0]))%180:.2f} degrees')


# ## Second Image

# In[ ]:


second_datasets = ft.open_file()
second_dataset = second_datasets['Channel_000']
view = second_dataset.plot()


# ## Determination of Magnification of Image 2
# 
# We plot the image in pixels and there will be a line to select the length of a feature.
# 
# Any Mouseclick will extend the line from the last click

# In[ ]:


fig = plt.figure()

plt.imshow(second_dataset.T)
plt.title(second_dataset.title)
ax = plt.gca()
start_x = 0

fixed_line = False
line = plt.plot([0, 100],[0,200], color = 'orange')


def on_click(event):
   
    if event.inaxes:
        (start_x, end_x), (start_y, end_y) = line[0].get_data()
        start_x = end_x
        start_y = end_y
        end_x = event.xdata
        end_y = event.ydata

        line[0].set_data([start_x, end_x],[start_y, end_y])

        plt.draw()
# mouse_reference = plt.connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_press_event', on_click)


# ## Length and Angle of Line of Image 2

# In[ ]:


line_coordinates = np.array(line[0].get_data()).T
vector = line_coordinates[0]-line_coordinates[1]
print(f' The line is {np.linalg.norm(vector):.2f}pixels long')
print(f' The angle is {np.degrees(np.arctan2(vector[1], vector[0]))%180:.2f} degrees')


# # Question
# 
# - What are the pixel sizes in the two images
# - What is the relative change in magnification with respect to pixel and with respect to indicated Magnification
# 
# - What is the relative rotation

# In[ ]:




