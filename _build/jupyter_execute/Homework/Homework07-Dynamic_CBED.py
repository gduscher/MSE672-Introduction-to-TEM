#!/usr/bin/env python
# coding: utf-8

# 
# <font size = "5"> **Chapter 2: [Diffraction](../Diffraction/CH2_00-Diffraction.ipynb)** </font>
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # HW7:  Simulating CBED Pattern
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Homework/Homework07-Dynamic_CBED.ipynb)
#  
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
#     https://colab.research.google.com/github/gduscher/MSE672-Introduction-to-TEM/blob/main/Homework/Homework07-Dynamic_CBED.ipynb)
# 
# 
# 
# part of 
# 
# <font size = "5"> **[MSE672:  Introduction to Transmission Electron Microscopy](../_MSE672_Intro_TEM.ipynb)**</font>
# 
# by Gerd Duscher, Spring 2022
# 
# Microscopy Facilities<br>
# Institute of Advanced Materials & Manufacturing<br>
# Materials Science & Engineering<br>
# The University of Tennessee, Knoxville
# 
# Background and methods to analysis and quantification of data acquired with transmission electron microscopes
# 

# ## Homework Assignment
# 
# Simulate the silicon [001] CBED diffraction pattern from Lab 4 (use the one with the higher camera length.
# 
# That homework has to be done on Google colab.
# 
# Press the **Open In Colab** button above.
# 
# >
# >Please note that you will have to activate the ``GPU`` under the Menu ``Runtime`` select ``Change runtime type`` 
# >
# 
# Questions:
# 
# 1. Set the convergence angle to the value you derived at in Homework 5 (please note, here it has to be in mrad not 1/nm))
# 2. What thickness did you arrive at?
# 3. How can you change the intensity in the middle of the zero disk?
# 4. How is that related with the intensity in the center of the [220] disks?
# 

# ## Load relevant python packages
# ### Check Installed Packages

# In[5]:


import sys
from pkg_resources import get_distribution, DistributionNotFound

def test_package(package_name):
    """Test if package exists and returns version or -1"""
    try:
        version = get_distribution(package_name).version
    except (DistributionNotFound, ImportError) as err:
        version = '-1'
    return version

# Colab setup ------------------
if 'google.colab' in sys.modules:
    get_ipython().system('pip install ase')
    get_ipython().system('pip install abtem -q')
# pyTEMlib setup ------------------
else:
    if test_package('abtem') < '1.0.0b17':
        print('installing abtem')
        get_ipython().system('{sys.executable} -m pip install  --upgrade abtem -q')
# ------------------------------
print('done')


# ### Import numerical and plotting python packages
# This notebook: only requires:
# * matplotlib: for plotting
# * numpy for numerical arrays
# * ase: for the structure
# * abtem: for the multislice calculation

# In[7]:


import sys
if 'google.colab' in sys.modules:
    get_ipython().run_line_magic('pylab', '--no-import-all inline')
else:
    get_ipython().run_line_magic('pylab', '--no-import-all notebook')

# import atomic simulation environment
import ase
import ase.build
import ase.visualize

# import abitio-tem library
import abtem


# ## Let's Make a Structure
# >
# > Here we make Si in the [001] direction 
# >
# 
# Change the input here

# In[8]:


# ------------ Input ----------
thickness = 220. #  in nm
number_of_layers = 4  # per unit cell
#------------------------------

atoms = ase.build.bulk('Si', 'diamond', cubic=True)
lattice_parameter = atoms.cell[2,2]
layers = int(thickness / lattice_parameter *10)
atoms *= (32, 32, layers)

atoms.center()


# ### Make the potential
# 
# We are using the frozen phonon approximation to make a varied potential.
# 
# Sigma is the averge deviation of positions from the symmetric sites (in Angstrom). Please do not change that value.
# 
# Please note that I am using 1024 by 1024 pixels (set by the gpts variable)

# In[10]:


sigma = 0.1256
frozen_phonons = abtem.FrozenPhonons(atoms, 12, {'Si': sigma}, seed=13, directions='xyz')

einstein_potential = abtem.Potential(frozen_phonons, gpts=1024, 
                                     slice_thickness=lattice_parameter/number_of_layers, 
                                     projection='infinite', parametrization='kirkland', 
                                     precalculate=False)


# ##  Definition of electron probe
# 

# In[17]:


# -------- Input --------
convergence_angle = 5  # in mrad
device = 'gpu'
# -----------------------

probe = abtem.Probe(energy=200e3, semiangle_cutoff=convergence_angle,  device=device)
probe.grid.match(einstein_potential)


# ## Definition of detector
# 
# We use a CCD camera equivalent with lots of pixels

# In[18]:


detector = abtem.PixelatedDetector(max_angle='limit')


# ## Multislice calculation
# If you get an error, then you have to either set the ``device`` variable in the code cell before last to ``'cpu'`` or you set the runtime of google colab to 'gpu' (under Menu point: Runtime - Change runtime type) 

# In[ ]:


measurement = probe.build().multislice(einstein_potential, pbar=True, detector=detector)


# ## Output

# In[ ]:


measurement.show(power=0.4)


# ## Repeat 
# 
# Repeat with different thickness till you get the right pattern in the zero disk of the CBED pattern of homework 5 (return to the structure code cell)

# ## Save the calculation
# 
# We save the numpy array, you can download this file in google colab from the file menu on the left.

# In[19]:


with open('Si011-slice0_1.npy', 'wb') as f:
    np.save(f, measurement.array[0])


# In[ ]:




