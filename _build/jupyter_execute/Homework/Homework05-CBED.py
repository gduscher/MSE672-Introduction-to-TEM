#!/usr/bin/env python
# coding: utf-8

# 
# <font size = "5"> **Chapter 2: [Diffraction](../Diffraction/CH2_00-Diffraction.ipynb)** </font>
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # HW5:  Analyzing CBED Pattern
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Homework/Homework05-CBED.ipynb)
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
# Background and methods to analysis and quantification of data acquired with transmission electron microscopes
# 

# ## Load relevant python packages
# ### Check Installed Packages

# In[1]:


import sys
from pkg_resources import get_distribution, DistributionNotFound

def test_package(package_name):
    """Test if package exists and returns version or -1"""
    try:
        version = get_distribution(package_name).version
    except (DistributionNotFound, ImportError) as err:
        version = '-1'
    return version


# pyTEMlib setup ------------------
if test_package('pyTEMlib') < '0.2023.2.0':
    print('installing pyTEMlib')
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib -q')
# ------------------------------
print('done')


# ### Import numerical and plotting python packages
# Import the python packages that we will use:
# 
# Beside the basic numerical (numpy) and plotting (pylab of matplotlib) libraries,
# 
# and some libraries from the book
# * kinematic scattering library.
# * file_tools library

# In[1]:


# import matplotlib and numpy
#                       use "inline" instead of "notebook" for non-interactive plots
get_ipython().run_line_magic('pylab', '--no-import-all notebook')
get_ipython().run_line_magic('gui', 'qt')

# additional package 
import  itertools 
import scipy.constants as const

import ipywidgets as ipyw
import sys
sys.path.insert(0, '../../pyTEMlib')
# Import libraries from pyTEMlib
import pyTEMlib
import pyTEMlib.kinematic_scattering as ks         # kinematic scattering Library
                             # atomic form factors from Kirkland's book

### And we use the image tool library of Quantifit
import pyTEMlib.file_tools as ft
import pyTEMlib
print(pyTEMlib.__version__)


# ## Load CBED Pattern
# >
# >Please note, that this notebook will not work in Google colab becaus of the ``open file dialog``
# >
# First we select the diffraction pattern

# In[2]:


datasets  = ft.open_file()
sidpy_dataset = datasets['Channel_000']
view = sidpy_dataset.plot()


# ### Plotting on a logarithmic scale
# 
# The dynamic range in diffraction data is even larger than in images and so for a good presentation of the data it is advantagous to go to plot the intensities in a logarythmic scale.
# 
# To present data in logarythmic scale no nexgative values (noise) can be in these data and so all negative values in the dataset will be set to zero.
# 
# The factor 1 in front of the diffraction pattern in the log numpy function in the ``imshow`` is the gamma value.
# Changing that value will change the contrast.

# In[3]:


diff_pattern = np.array(sidpy_dataset).T
diff_pattern[diff_pattern<0] = 0.
extent = sidpy_dataset.get_extent([0,1])
print(extent)
fig = plt.figure() 
plt.imshow(np.log(1+diff_pattern),cmap="gray", vmin=np.max(np.log(1+diff_pattern))*0.5, extent=extent);



# ## Finding the center
# 
# ### Selection of  center disk
# 
# Select the center disk with the ring selector
# 

# In[4]:


from matplotlib.widgets import  EllipseSelector

center = np.array([1024,1024])

plt.figure(figsize=(8, 6))
plt.imshow(diff_pattern.T, origin = 'upper')
selector = EllipseSelector(plt.gca(), None,interactive=True)  # gca get current axis (plot)
radius = 559 
center = np.array(center)

selector.extents = (center[0]-radius,center[0]+radius,center[1]-radius,center[1]+radius)


# ### Read out center and radius

# In[5]:


xmin, xmax, ymin, ymax = selector.extents
x_center, y_center = selector.center
x_shift = x_center - diff_pattern.shape[0]/2
y_shift = y_center - diff_pattern.shape[1]/2
print(f'radius x-direction = {(xmax-xmin)/2:.0f} pixels')
print(f'radius y-direction = {(ymax-ymin)/2:.0f} pixels')

center = (x_center, y_center )
print(f'new center = {center} [pixels]')

out_tags ={}
out_tags['center'] = center


# ## Calculate Spot Pattern
# 
# see [Plotting of Diffraction Pattern](../Diffraction/CH2_08-Spot_Diffraction_Pattern.ipynb) for details
# 

# In[6]:


#Initialize the dictionary of the input
tags_experiment = {}
### Define Crystal
atoms  = ks.structure_by_name('silicon')

### Define experimental parameters:
tags_experiment['acceleration_voltage_V'] = 200.0 *1000.0 #V
tags_experiment['new_figure'] = False
tags_experiment['plot FOV'] = 30
tags_experiment['convergence_angle_mrad'] = 0
tags_experiment['zone_hkl'] = np.array([0,0,1])  # incident neares zone axis: defines Laue Zones!!!!
tags_experiment['mistilt']  = np.array([0,0,0])  # mistilt in degrees
tags_experiment['Sg_max'] = .4 # 1/nm  maximum allowed excitation error ; This parameter is related to the thickness
tags_experiment['hkl_max'] = 15   # Highest evaluated Miller indices

atoms.info['experimental'] = tags_experiment
######################################
# Diffraction Simulation of Crystal #
######################################

ks.kinematic_scattering(atoms, verbose=False)


# ## Plotting Experimental and Simulated Spot Diffraction Patterns
# 
# There is a problem with those diffraction pattern and they did not store the scale.
# So I am using an approximate one and you'll have to change that.

# In[7]:


# ----- Input ---------
rotation_angle = -7 # in degrees
convergence_angle = .11  # in 1/nm
pixel_size = 10 *1e-4
# -------------------
tags_simulation = atoms.info['diffraction']

#scale_300mm
#g = gx = gy = ft.get_slope(sidpy_dataset.x.values) # does not work here

g = gx = gy = pixel_size

print(f'Scale is {g:.4f} 1/nm per pixel')
extent= np.array([-center[0]*gx, (diff_pattern.shape[0]-center[0])*gx,(diff_pattern.shape[1]-center[1])*gy, -center[1]*gy])

# rotation matrix  around z axis to coincide with spots
angle = np.radians(rotation_angle)
c = np.cos(angle)
s = np.sin(angle)
r_mat = np.array([[c,-s,0],[s,c,0],[0,0,1]])
rotation_matrix = r_mat


spots_simulation =  np.dot(tags_simulation['allowed']['g'], rotation_matrix)
spots_ZOLZ = spots_simulation[tags_simulation['allowed']['ZOLZ']]
fig = plt.figure()
#fig.suptitle(' SAED in ' + str(tags_experiment['zone_hkl']), fontsize=20) 
plt.scatter(spots_ZOLZ[:,0], spots_ZOLZ[:,1], c='red',  alpha = 0.2,   label='spots')
plt.imshow(np.log2(1+diff_pattern).T,cmap="gray", extent=(extent), vmin=np.max(np.log2(1+diff_pattern))*0.5);

for i in range(len(tags_simulation['allowed']['g'])):
    if np.linalg.norm(tags_simulation['allowed']['g'][i]) <1:
        disk = plt.Circle((spots_simulation[i,0], spots_simulation[i,1] ), convergence_angle, alpha=0.3)
        plt.gca().add_artist(disk) 
        plt.text(spots_simulation[i,0], spots_simulation[i,1],str(tags_simulation['allowed']['hkl'][i]),
                fontsize = 8, horizontalalignment = 'center', verticalalignment ='bottom')
    


# ### What does the above figure convey?
# 
# 
# 

# ### What is the accuracy?
# 
# 
# 

# ## Conclusion
# 
# We need more information for the spot pattern than for the ring pattern.
# 
# The convergent beam pattern has the same kinematic diffraciton information but provides additional dynamical diffraction information.
# 
# A comparison between simulation and experiment can be very precise.
# 
# In principle, if you have the spots and the approximate center you can let an optimization routine do all the scaling for you (which we will do in the high resultion imaging section).
# 
# 

# In[ ]:




