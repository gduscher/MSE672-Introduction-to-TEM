#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# 
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# 
# 
# # The Electron 
# 
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Diffraction/CH2_01-Electron.ipynb)
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

# First we load the code to make figures from pyTEMlib
# ## Import packages for figures and 
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

if test_package('pyTEMlib') < '0.2023.1.0':
    print('installing pyTEMlib')
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib -q')

print('done')


# ### Load the plotting and figure packages

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pylab as plt
import numpy as np

import scipy.constants as const  #scientific constants


# ## Interaction of Common Particles with Matter
# 
# We generally use electron, photons, and neutrons for diffraction/scattering
# experiments.
# 
# These particles interact with differently with matter:
# 
#  <table style="width:80%">
#  
#   <tr>
#     <td>X-rays</td>
#     <td>$\leftrightarrow$</td>
#     <td>electron density</td>
#   </tr>
#   <tr>
#     <td>neutrons</td>
#     <td>$\leftrightarrow$</td>
#     <td>mass of nucleus</td>
#   </tr>
#   <tr>
#     <td>neutrons</td>
#     <td>$\leftrightarrow$</td>
#     <td>magnetic moment</td>
#   </tr>
#   <tr>
#     <td>electrons</td>
#     <td>$\leftrightarrow$</td>
#     <td>screened charge of nucleus</td>
#   </tr>
#  
# </table> 
# 
# We will deal with the nature of electrons more closely in the following

# ## Non-relativistic de Broglie Wavelength
# 
# 
# The electron is a elementary particle with spin $\frac{1}{2}$ (lepton).
# 
# 
# **Non--relativistic De Broglie wavelength** of electron: 
# 
# $\lambda = \frac{h}{p} = \frac{h}{\sqrt{2m_0E_{kin}}} \approx \frac{1.22}{\sqrt{E_{kin}}}$
# 
# 
# E is the kinetic energy of the electron: $E_{kin} = eU $ [eV].
# 
# The wave length in a TEM is usually
# a couple of picometers . This is a
# factor of 100 smaller than your
# XRD-source.
# 
# Obvioulsy, we are in the wave picture right now.

# In[3]:


## input 
acceleration_voltage_V = U = 300.0 * 1000.0 #V   

## energy
E_kin = eU = const.e * acceleration_voltage_V  # potential

wave_length_m = const.h/np.sqrt(2*const.m_e*E_kin) # non-relativistic wavelength in m


##please note that we will keep all length units in nm if possible.
##otherwise we useonly SI units!!!
wave_length_A = wave_length_m *1e10 # now in Angstrom

print(np.sqrt(2/const.m_e*E_kin)/const.c)

print(f'Classic wave length is {wave_length_A*100.:.2f} pm for acceleration voltage {acceleration_voltage_V/1000.:.1f} kV')
# Notice that we change units in the output to make them most readable.

print(f' which is a velocity of {np.sqrt(2/const.m_e*E_kin):.2f} m/s or {np.sqrt(2/const.m_e*E_kin)/const.c*100:.2f}% of the speed of light')


# ## Relativistic Correction
# In the table below we see that the speeds of the electron is rather close to the speed of light $c$
# 
# The formula for relativistic corrected wavelength is:
# $\lambda = \frac{h}{\sqrt{2m_e E_{kin} *(1+\frac{E_{kin}}{2 m_e c^2})}}$
# 
# **Please note:** All units are internally in SI units: kg, s, V, J, except the length wihich is in nm!
# 
# We multiply with the appropriate factors for the output

# In[8]:


# Input: Acceleration Voltage
E0 = acceleration_voltage = 200.0 *1000.0 #V

E_kin = eU = const.e * acceleration_voltage #potential

#relativisitic wavelength
wave_length = const.h/np.sqrt(2*const.m_e*E_kin*(1+E_kin/(2*const.m_e*const.c**2))) #in m

print(f'The relativistically corrected wave length is {wave_length*1e12:.2f} pm for acceleration voltage {acceleration_voltage/1000:.1f} kV')


# 100kV : $\lambda$ = 4 pm $<$ than diameter of an atom
# 
# The relativistic parameters are:
# 
# 
# |E (keV)|$\lambda$ (pm) | M/m$_0$ | v/c|
# --------|---------------|---------|----|
# |10 |  12.2 | 1.0796 | 0.1950 |
# |30 | 6.98 | 1.129 | 0.3284 |
# |100 | 3.70 | 1.1957 | 0.5482|
# |200 | 2.51 | 1.3914 |  0.6953|
# |400 | 1.64 | 1.7828 | 0.8275 |
# |1000 | 0.87 | 2.9569 | 0.9411|
# 
# The same functionality (and code) is used in the kinematic_scattering library and we can test the values of above table.
# 
# Please change the acceleration voltage (**acceleration_voltage**) above.
# 

# 
# ### Relativistic velocity
# 
# $$\frac{v^2}{c^2} = \frac{E_{kin}(E_{kin}+2m_e c^2)}{(E_{kin}+m_e c^2)^2}$$

# In[9]:


v = np.sqrt(E_kin*(E_kin+2*const.m_e*const.c**2)/(E_kin+const.m_e*const.c**2)**2)*const.c

print(f'The classic velocity of the electron  is {np.sqrt(2/const.m_e*E_kin):.2f} m/s or {np.sqrt(2/const.m_e*E_kin)/const.c*100:.2f}% of the speed of light')
print(f'The relativistic velocity of the electron  is {v:.2f} m/s or {v/const.c*100:.2f}% of the speed of light')


# ## That means that the resolution is not limited by the wavelength!

# In[11]:


# Import Kinematic Scattering Library
import pyTEMlib.kinematic_scattering as ks         # Kinematic sCattering Library

acceleration_voltage= 30*1e3
wave_length = ks.get_wavelength(acceleration_voltage)
print(f'The relativistically corrected wave length is {wave_length*1e2:.2f} pm for acceleration voltage {acceleration_voltage/1000:.1f} kV')

# Wavelength in Angstrom
def get_wavelength(acceleration_voltage):
    """
    Calculates the relativistic corrected de Broglie wave length of an electron

    Input:
    ------
        acceleration voltage in volt
    Output:
    -------
        wave length in Angstrom
    """

    eU = const.e * acceleration_voltage 
    return const.h/np.sqrt(2*const.m_e*eU*(1+eU/(2*const.m_e*const.c**2)))*10**10


# In[12]:


help(ks.get_wavelength)


# In[13]:


help(ks)


# ## Particle Flux and Current
# 
# It is important todetermine the order of magitude of how many electrons are hitting the sample.
# 
# The electron sources deliver in the order of $\mu$A current, but most of these electrons are not used. 
# 
# In a modern electron microscope, we talk about a range of 1pA to 1nA in the electron beam.
# 
# We start with the defition of an Ampere:
# $$A = \frac{C}{s}$$
# 
# That definition is enough to calculate the number ofelectron per time unit (flux).

# In[14]:


print(f" elementary charge: {const.physical_constants['elementary charge'][0]:.5g} {const.physical_constants['elementary charge'][1]}")
print(f'\n 1pA is {1e-12/const.e:.3} electrons/s')
print(f' 10pA is {10e-12/const.e *1e-3 :.0f} electrons/ms')
print(f'100pA is {100e-12/const.e*1 :.3} electrons/s')

print(f'\n at 10pA an electron will hit the sample every {const.e/10e-12 * 1e9:.2f} ns ')


# We see that we have much lower fluence in the TEM than in a laser (how could they do femtosecond pulses otherwise).
# 

# ## Navigation
# - <font size = "3">  **Back Chapter 1: [Introduction](../Introduction/CH1_00-Introduction.ipynb)** </font>
# - <font size = "3">  **Next: [Atomic Form Factor](CH2_02-Atomic_Form_Factor.ipynb)** </font>
# - <font size = "3">  **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# - <font size = "3">  **List of Content: [Front](../_MSE672_Intro_TEM.ipynb)** </font>

# In[ ]:




