#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **[MSE672: Introduction to TEM](https://gduscher.github.io/MSE672-Introduction-to-TEM/)** </font>
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# by 
# 
# Gerd Duscher
# 
# Materials Science & Engineering<br>
# Joint Institute of Advanced Materials<br>
# The University of Tennessee, Knoxville
# 
# # Test Notebook
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/notebooks/TestNotebook.ipynb")
#  
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
#     https://colab.research.google.com/github/gduscher/MSE672-Introduction-to-TEM/blob/main/notebooks/TestNotebook.ipynb)
# 

# First we need to load the libraries we want to use. All of those are installed in Google colab and annaconda.

# In[1]:


import numpy as np
import matplotlib.pylab as plt


# Now we run the next code cell to plot a graph.

# In[2]:


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




