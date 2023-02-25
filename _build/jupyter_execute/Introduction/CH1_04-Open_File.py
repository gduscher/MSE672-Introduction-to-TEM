#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **Chapter 1: [Introduction](CH1_00-Introduction.ipynb)** </font>
# 
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # Open DM3 Images, Spectra, Spectrum-Images and  Image-Stacks with pyNSID 
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Introduction/CH1_04-Open_File.ipynb)
#  
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
# ---
# Reading a dm file and translating the data in a **[pyNSID](https://pycroscopy.github.io/pyNSID/)** style hf5py file to be compatible with  the **[pycroscopy](https://pycroscopy.github.io/pycroscopy/)** package.
# 
# Because, many other packages and programs for TEM data manipulation are based on the ``hdf5`` file-formats it is relatively easy to convert back and forward between them.
# 
# 

# ## Import packages for figures and
# ### Check Installed Packages

# In[1]:


from pkg_resources import get_distribution, DistributionNotFound

def test_package(package_name):
    """Test if package exists and returns version or -1"""
    try:
        version = get_distribution(package_name).version
    except (DistributionNotFound, ImportError):
        version = '-1'
    return version

if test_package('pyTEMlib') < '0.2023.1.0':
    print('installing pyTEMlib')
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib -q')
# ------------------------------
print('done')


# ### Load the plotting and figure packages

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pylab as plt
import numpy as np

get_ipython().run_line_magic('gui', 'qt')

import pyTEMlib
import pyTEMlib.file_tools  as ft     # File input/ output library

import sidpy
import pyNSID
import h5py

# For archiving reasons it is a good idea to print the version numbers out at this point
print('pyTEM version: ',pyTEMlib.__version__)
__notebook__='CH1_04-Reading_File'
__notebook_version__='2021_12_14'


# ## Open a file 
# 
# This function opens a hfd5 file in the pyNSID style which enables you to keep track of your data analysis.
# 
# Please see the **[Installation](CH1_02-Prerequisites.ipynb#TEM-Library)** notebook for installation.
# 
# We want to consolidate files into one dataset that belongs together.  For example a spectrum image dataset consists of: 
# * Survey image, 
# * EELS spectra 
# * Z-contrast image acquired simultaneously with the spectra.
# 
# 
# So load the top dataset first in the above example the survey image.
# 
# Please note that the plotting routine of ``matplotlib`` was introduced in **[Matplotlib and Numpy for Micrographs](CH1_03-Data_Representation.ipynb)** notebook.
# 
# **Use the file p1-3hr.dm3 from TEM_data directory for a practice run**

# In[2]:


# ------ Input ------- #
load_example = False
# -------------------- #

# Open file widget and select file which will be opened in code cell below
if not load_example:
    drive_directory = ft.get_last_path()
    file_widget = ft.FileWidget(drive_directory)


# In[3]:


try:
    main_dataset.h5_dataset.file.close()
except:
    pass

if load_example:
    file_name = '../example_data/p1-3-hr3.dm3'
else:
    file_name = file_widget.file_name

datasets = ft.open_file()
main_dataset = datasets['Channel_000']

view = main_dataset.plot()


# In[ ]:





# ## Data Structure
# 
# The data themselves reside in a ``sidpy dataset`` which we name ``current_dataset``.

# The current_dataset has additional information stored as attributes which can be accessed through their name.

# In[4]:


print(main_dataset)
main_dataset


# In[5]:


print(f'size of current dataset is {main_dataset.shape}')


# The current_dataset has additional information stored as attributes which can be accessed through their name.
# 
# There are two dictionaries within that attributes:
# - **metadata**
# - **original_metadata**
# 
# which contain additional information about the data

# In[6]:


print('title: ', main_dataset.title)
print('data type: ', main_dataset.data_type)

for key in datasets:
    print(key)
    print(datasets[key].original_metadata.keys())
    
main_dataset.metadata  


# ## Data Structure
# The datasets variable is a dictionary (like a directory in a file system) which containes contains datasets.
# 
# Below I show how to access one of those datasets with a pull down menu.

# In[7]:


chooser = ft.ChooseDataset(datasets)


# In[8]:


current_dataset = chooser.dataset
view = current_dataset.plot()


# An important attribute in ``current_dataset`` is the ``original_metadata`` group, where all the original metadata of your file reside in the ``attributes``. This is usually a long list for ``dm3`` files.

# In[12]:


current_dataset.original_metadata.keys()


# The original_metadata attribute has all information stored from the orginal file. 
# > No information will get lost

# In[9]:


for key,value in current_dataset.original_metadata.items():
    print(key, value)
print(current_dataset.h5_dataset)    


# Any python object will provide a help.

# In[14]:


help(current_dataset)


# All attributes of a python object can be viewed with the * dir* command. 
# > As above: too much information for normal use, but it is there if needed.

# In[15]:


dir(current_dataset)


# ## Adding Data
# 
# To add another dataset that belongs to this measurement we will use the **h5_add_channel** from  **file_tools** in the  pyTEMlib package.
# 
# Here is how we add a channel there.
# 
# We can also add a new measurement group (add_measurement in pyTEMlib) for similar datasets.
# 
# This is equivalent to making a new directory in a file structure on your computer.

# In[16]:


datasets['Copied_of_Channel_000'] = current_dataset.copy()


# We use above functions to add the content of a (random) data-file to the current file.
# 
# This is important if you for example want to add a Z-contrast or survey-image to a spectrum image.
# 
# Therefore, these functions enable you to collect the data from different files that belong together.
# 

# In[17]:


datasets.keys()


# ## Adding additional information
# 
# Similarly, we can add a whole new measurement group or a structure group.
# 
# This function will be contained in the KinsCat package of pyTEMlib.
# 
# If you loaded the example image, with graphite and ZnO both are viewed in the [1,1,1] zone axis.
# 

# In[18]:


import pyTEMlib.kinematic_scattering as ks         # kinematic scattering Library
                             # with Atomic form factors from Kirkland's book
import ase

                                                                                 
graphite = ks.structure_by_name('Graphite')
print(graphite)


# In[19]:


current_dataset.structures['Crystal_000'] = graphite
                                                            
zinc_oxide = ks.structure_by_name('ZnO')
current_dataset.structures['ZnO'] =zinc_oxide               


# ## Keeping Track of Analysis and Results
# A notebook is notorious for getting confusing, especially if one uses different notebooks for different task, but store them in the same file.
# 
# If you like a result of your calculation, log it.
# 
# Use the datasets dictionary to add a analysed and/or modified dataset. Make sure the metadata contain all the necessary information, so that you will know later what you did.
# 
# The convention in this class will be to call the dataset **Log_000**.
# 

# In[20]:


new_dataset = current_dataset.T
new_dataset.metadata = {'analysis': 'Nothing', 'name': 'Nothing'}
datasets['Log_000'] = new_dataset


# ## An example for a log
# We log the Fourier Transform of the image we loaded
# 
# First we perform the calculation

# In[21]:


fft_image = current_dataset.fft().abs()
fft_image = np.log(60+fft_image)

view = fft_image.plot()


# Now that we like this we log it.
# 
# Please note that just saving the fourier transform would not be good as we also need the scale and such.

# In[22]:


fft_image.title = 'FFT Gamma corrected'
fft_image.metadata = {'analysis': 'fft'}
datasets['Log_001'] = fft_image

view = fft_image.plot()


# We added quite a few datasets to our dictionary. 
# 
# Let's have a look
# 

# In[23]:


chooser = ft.ChooseDataset(datasets)


# In[24]:


view = chooser.dataset.plot()


# ## Save Datasets to  hf5_file
# Write all datasets to one h5_file, which we then close immediatedly

# In[23]:


h5_group = ft.save_dataset(datasets, filename='./nix.hf5')


# Close the file

# In[24]:


h5_group.file.close()


# ## Open h5_file
# Open the h5_file that we just created

# In[27]:


datasets2= ft.open_file(filename='./nix-1.hf5')

chooser = ft.ChooseDataset(datasets2)


# ### Short check if we got the data right
# we print the tree and we plot the data

# In[28]:


view = chooser.dataset.plot()


# 
# ## Navigation
# - <font size = "3">  **Back  [Matplotlib and Numpy for Micrographs](CH1_03-Data_Representation.ipynb)** </font>
# - <font size = "3">  **Next: [Overview](CH1_06-Overview.ipynb)** </font>
# - <font size = "3">  **Chapter 1: [Introduction](CH1_00-Introduction.ipynb)** </font>
# - <font size = "3">  **List of Content: [Front](../_MSE672_Intro_TEM.ipynb)** </font>
# 

# In[ ]:




