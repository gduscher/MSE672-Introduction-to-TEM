#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **Chapter 1: [Introduction](CH1_00-Introduction.ipynb)** </font>
# 
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # Open a Data File with Icons
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Introduction/CH1_01-Introduction_Python.ipynb)
# 
# 
# part of 
# 
# <font size = "5"> **[MSE672:  Introduction to Transmission Electron Microscopy](_MSE672_Intro_TEM.ipynb)**</font>
# 
# by Gerd Duscher, Fall 2022
# 
# Microscopy Facilities<br>
# Institute of Advanced Materials & Manufacturing<br>
# Materials Science & Engineering<br>
# The University of Tennessee, Knoxville
# 
# Background and methods to analysis and quantification of data acquired with transmission electron microscopes.
# 

# ## Load necessary Libraries

# In[1]:


get_ipython().run_line_magic('pylab', 'notebook')
get_ipython().run_line_magic('gui', 'qt')

import sys
sys.path.insert(0,'../../pyTEMlib/')
sys.path.insert(0,'../../sidpy/')

import sidpy

import pyTEMlib
import pyTEMlib.file_tools_qt
import pyTEMlib.file_tools as ft
sidpy.__version__


# ## Select and Plot a Data File

# In[3]:


plt.close('all')


# In[2]:


try:
    dataset.h5_dataset.file.close()
except:
    pass
file_viewer = pyTEMlib.file_tools_qt.FileIconDialog('../example_data/')
dataset = ft.open_file(file_viewer.file_name)
view = dataset.plot(verbose=True)


# In[ ]:


dataset.h5_dataset.file.close()


# In[4]:


dataset.plot()


# In[11]:


import sidpy.viz as viz


# In[26]:





# In[24]:


class CurveVisualizer(object):
    def __init__(self, dset, spectrum_number=0, figure=None, **kwargs):
                
        scale_bar = kwargs.pop('scale_bar', False)
        colorbar = kwargs.pop('colorbar', True)
        set_title = kwargs.pop('set_title', True)

        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        if figure is None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure

        self.dset = dset
        self.selection = []
        self.spectral_dims = []

        for dim, axis in dset._axes.items():
            if axis.dimension_type == sidpyDimensionType.SPECTRAL:
                self.selection.append(slice(None))
                self.spectral_dims.append(dim)
            else:
                if spectrum_number <= dset.shape[dim]:
                    self.selection.append(slice(spectrum_number, spectrum_number + 1))
                    self.spectral_dims.append(dim)
                else:
                    self.spectrum_number = 0
                    self.selection.append(slice(0, 1))
                    self.spectral_dims.append(dim)

        # Handle the simple cases first:
        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        self.dim = self.dset._axes[self.spectral_dims[0]]


        self.axis = self.fig.add_subplot(1, 1, 1, **fig_args)
        self.axis.plot(self.dim.values, self.dset, **kwargs)
        if set_title:
            self.axis.set_title(self.dset.title, pad=15)
        self.axis.set_xlabel(self.dset.labels[self.spectral_dims[0]])
        self.axis.set_ylabel(self.dset.data_descriptor)
        self.axis.ticklabel_format(style='sci', scilimits=(-2, 3))
        self.fig.canvas.draw_idle()



# In[21]:


sidpy.DimensionType


# In[10]:


fig, ax = plt.subplots(figsize=(1,1), nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot([0,1,2], [10,20,3])
plt.ioff()
fig.savefig('to.png')   # save the figure to file
plt.close(fig) 


# In[6]:


plt.figure()
x = [1, 2, 3, 4, 5]
y = [4, 6, 3, 7, 2]
plt.plot(x, y)
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("Matplotlib - Save plot but dont show")
plt.savefig("filename.png")
plt.show()


# In[ ]:




