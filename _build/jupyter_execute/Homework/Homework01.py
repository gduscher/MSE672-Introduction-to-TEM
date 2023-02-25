#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **Chapter 1: [Introduction](../Introduction/CH1_00-Introduction.ipynb)** </font>
# 
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # Homework 1
# 
# <font size = "5"> Test Python Environment</font>
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Homework/Homework01.ipynb)
# 
# part of 
# 
# <font size = "5"> **[MSE672:  Introduction to Transmission Electron Microscopy](../_MSE672_Intro_TEM.ipynb)**</font>
# 
# by Gerd Duscher, Spring 2022
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
# 
# All of those packages are provided by annaconda.

# In[ ]:


import numpy as np
import matplotlib.pylab as plt


# Now we run the next code cell to plot a graph.

# In[ ]:


# Generate plotting values
t = np.linspace(0, 2*np.pi, 200)
x = 16 * np.sin(t)**3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

plt.figure()
plt.plot(x,y, color='red', linewidth=2)
plt.text(-23, 0,  'I', ha='center', va='center', fontsize=206)

plt.text(20,0, 'MSE 672',va='center', fontsize=206)
plt.gca().set_aspect('equal')


# Who would have thought that the formula of love is based on trigonometry?

# In[ ]:




