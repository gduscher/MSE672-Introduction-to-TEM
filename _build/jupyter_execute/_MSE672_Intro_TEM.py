#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **MSE672:  Introduction to Transmission Electron Microscopy**</font>
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# by Gerd Duscher, Spring 2023
# 
# Microscopy Facilities<br>
# Institute of Advanced Materials & Manufacturing<br>
# Materials Science & Engineering<br>
# The University of Tennessee, Knoxville
# 
# 
# 
# # Foreword (ReadME)
# This course provides information on analysis of TEM data and the background of the nature of this information in the data.
# 
# Please note, that there is **no programming** required, but the user should not be shy in changing the values of parameters.
# 
# Therefore, two ways of usage are targeted, 
# * Beginner
#     * Please start at the beginning and advance in the order of the notebooks given here.
# * Advanced, Inpatient or both:
#     * Go to the pyTEMlib notebooks (and give feedback if links are missing, please).
# 
# 
# You can [download all files](https://github.com/gduscher/MSE672-Introduction-to-TEM/archive/main.zip)
# from GitHub as a zip file.

# 

# ## Libraries and Classes
# 
# The functions introduced in this book are also organized in a package. The package is named [pyTEMlib](https://github.com/gduscher/pyTEMlib) and can be downloaded from GitHub (no pip installation at this point in time).
# 
# 
# For a more modern programming approach they could also be grouped in classes, but classes  put another layer between notebook and code, which is desirable for abstraction but not necessarily for understanding.
# 
# So for the course of this book the functions will be made available in libraries, which can be wrapped in classes for a more monolithic program.
# 
# A graphical user interface (GUI) was consciously omitted in this book to encourage the user to mess around in the code.
# So please change parameters and see what happens.
# 
# ## A word of caution:
# A notebook can become confusing if one does not go through it in a sequential way, because the values of parameters can be changed at any stage in any code cell without the other cells having any knowledge about the order in which the cells are activated.
# 
# A program or function will not have that kind of confusing tendency, and therefore, once we understand a topic, a comprehensive function will be provided.
# 

# In[ ]:




