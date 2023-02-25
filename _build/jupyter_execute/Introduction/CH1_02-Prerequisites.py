#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **Chapter 1: [Introduction](CH1_00-Introduction.ipynb)** </font>
# 
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # Prerequisites
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Introduction/CH1_02-Prerequisites.ipynb)
# 
# 
# 
# part of 
# 
# <font size = "5"> **[MSE672:  Introduction to Transmission Electron Microscopy](../_MSE672_Intro_TEM.ipynb)**</font>
# 
# by Gerd Duscher, Spring 2023
# 
# Microscopy Facilities<br>
# Institute of Advanced Materials & Manufacturing<br>
# Materials Science & Engineering<br>
# The University of Tennessee, Knoxville
# 
# Background and methods to analysis and quantification of data acquired with transmission electron microscopes.
# 
# 

# ## Language
# The notebooks are all in python 3. 
# 
# At this point the common version is python 3.9

# cd
# ## Packages
# The idea behind any python program is to make use of the highly efficient libraries that already exist.
# 
# I use [anaconda3](https://www.anaconda.com/distribution/) (not miniconda) which is available for all major operating systems.
# 
# We use a few modules that come with every python installation, like:
# 
# * math
# * sys
# * os
# 
# We use mostly the common packages for scientific computing in python (all included in anaconda3)
# The most important ones are:
# * [Numpy](https://www.numpy.org/) - the numerical library
# * [Scipy](https://www.scipy.org/scipylib/index.html) the scientific library
# * [Matplotlib](https://www.matplotlib.org/) the interactive plotting library 
# 
# 
# These packages are expected to be installed on your computer to run the notebooks of this book.
# 
# For specialist applications we do not reinvent the wheel and use those on an as needed basis.
# 
# Example is the library to register a stack of images:
# * [SimpleITK](https://www.simpleitk.org)
# integration of AI in image analysis and for high performing computer algorithms
# * [pyNSID](https://pycroscopy.github.io/pyNSID/about.html)
# the atomistic simulation program is used for crystallographic data
# * [ase](https://wiki.fysik.dtu.dk/ase/)
# together with a symmetry package
# * [spglib](https://atztogo.github.io/spglib/)
# 
# 
# For dialogs, we use the capabilities provided by:
# * [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro)
# 
# All routines that are introduced in the notebooks are also available (for analysis) in the provided package  
# * [pyTEMlib](https://github.com/pycroscopy/pyTEMlib)
# 
# 
# If you install **[pyTEMlib](#TEM-Library)** with the code cell below all packages you need for this book will be installed.

# ## Basic Installation of Python Environment
# 
# I recommend installing the free **anaconda3** from this [link](https://www.anaconda.com/products/individual).
# 
# If you have an old version of anaconda, please reinstall the new version.
# The lecture is based on fairly new packages of scipy and numpy, dask, and h5py, which will create problems in old anaconda versions.
# 
# Once you have installed anaconda
# - open the anaconda PowerShell prompt and type:
# >conda install -c conda-forge pytemlib
# 
# - or type 
# >pip install pyTEMlib
# 
# Alternatively you can just run the code cell below. Be aware that the installation process may take a while.

# In[1]:


import sys
from pkg_resources import get_distribution, DistributionNotFound

def test_package(package_name):
    """Test if package exists and returns version or -1"""
    try:
        version = (get_distribution(package_name).version)
    except (DistributionNotFound, ImportError) as err:
        version = '-1'
    return version

# pyTEMlib setup ------------------
if test_package('pyTEMlib') < '0.2023.1.0':
    print('installing pyTEMlib')
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib -q')
# ------------------------------
print('done')


# ## Data Format
# All data in this course are stored in the data format of
# * [pyNSID](https://pycroscopy.github.io/pyNSID/about.html)
# 
# which is based on
# * [HDF5](https://www.h5py.org/)
# 
# 
# 
# ## Notebook preamble
# As a minimum Any notebook in this course has to have the following libraries loaded :

# In[1]:


# import matplotlib and numpy with this **magic** command
#                       use "inline" instead of "notebook" for non-interactive plots
#                       use "widget" instead of "notebook" for jupyter lab
get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pylab as plt
import numpy as np


# ## Numpy
# 
# The calculations depend on **Numpy** and an installation of that package that is compiled to include BLAS and LAPACK libraries will be much faster than the standalone version.
# For example the numpy installed on ubuntu with *> sudo apt-get install python3-numpy* or at windows you can install the numpy package from Gohlke's webpage which compiled against the MKL library of Intel. If you used anaconda3, everything is already optimized.
# 
# The command below lets you see what you have

# In[2]:


## What is numpy compiled against
np.__config__.show()
print('numpy version: ',np.version.version)
import scipy as sp
print('scipy version: ',sp.__version__)


# ## TEM Library
# 
# 
# You will have to run the code cell below **at least once** to install the library with the programs needed for the analysis of data.
# 
# The  code cell below will install pyTEMlib  directly from pypi

# In[3]:


import sys
from pkg_resources import get_distribution, DistributionNotFound

def test_package(package_name):
    """Test if package exists and returns version or -1"""
    try:
        version = get_distribution(package_name).version
    except (DistributionNotFound, ImportError):
        version = '-1'
    return version

# pyTEMlib setup ------------------
if test_package('pyTEMlib') <= '0.2023.1.0':
    print('installing pyTEMlib')
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib -q')
# ------------------------------
print('done')


# Now we load pyTEMlib to make it available for this notebook.

# In[4]:


import pyTEMlib
print(f'pyTEM version: {pyTEMlib.__version__}')


# ## Test
# Let's test if the installation was successful and plot a unit cell. You can rotate the plot around, zoom and select. Try it!

# In[5]:


import pyTEMlib.kinematic_scattering as ks # Import kinematic scattering library from pyTEMlib

# make a structure dictionary
atoms = ks.structure_by_name('Graphite')

atoms


# In[6]:


from ase.visualize import view
view(atoms*(4,4,1))


# Get used to changing parameters and type **silicon** instead of **Graphite** in the code cell above.

# ## Summary
# 
# We now have tested all tools to load data and save our analysis.
# 
# We are ready to go

# ## Navigation
# - <font size = "3">  **Back  [Python as it is used here](CH1_01-Introduction_Python.ipynb)** </font>
# - <font size = "3">  **Next: [Matplotlib and Numpy for Micrographs](CH1_03-Data_Representation.ipynb)** </font>
# - <font size = "3">  **Chapter 1: [Introduction](CH1_00-Introduction.ipynb)** </font>
# - <font size = "3">  **List of Content: [Front](../_MSE672_Intro_TEM.ipynb)** </font>
# 

# ## Appendix
# 
# I am using some extensions to jupyter notebooks which can be installed with the following cells
# 
# I like mostly the table of content extension that shows where in the notebook I am and lets me jump to different parts easily.
# 

# In[1]:


# Install a pip package in the current Jupyter kernel
import sys
get_ipython().system('{sys.executable} -m jupyter nbextension list')


# Use one of the following (uncomment pip and comment out the conda command line, if you are not in an anaconda environment)

# In[2]:


import sys

#!{sys.executable} -m pip install jupyter_contrib_nbextensions
get_ipython().system('conda install --yes --prefix {sys.prefix} jupyter_contrib_nbextensions')


# 

# In[3]:


get_ipython().system('{sys.executable} -m  jupyter nbextension enable toc2')


# In[4]:


get_ipython().run_cell_magic('javascript', '', '$(\'<div id="toc"></div>\').css({position: \'fixed\', top: \'120px\', left: 0}).appendTo(document.body);\n$.getScript(\'https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js\');\n')


# In[ ]:




