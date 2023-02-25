#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # Structure Factors
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Diffraction/CH2_04-Structure_Factors.ipynb)
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

if test_package('pyTEMlib') < '0.2023.1.0':
    print('installing pyTEMlib')
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib -q')
print('done')


# ### Load the plotting and figure packages
# Import the python packages that we will use:
# 
# Beside the basic numerical (numpy) and plotting (pylab of matplotlib) libraries,
# * three dimensional plotting
# and some libraries from the book
# * kinematic scattering library.

# In[7]:


get_ipython().run_line_magic('matplotlib', ' notebook')
import matplotlib.pyplot as plt
import numpy as np

    
# 3D plotting package 
from mpl_toolkits.mplot3d import Axes3D # 3D plotting

# additional package 
import  itertools 
import scipy.constants as const

# Import libraries from the book
import pyTEMlib
import pyTEMlib.kinematic_scattering as ks         # Kinematic sCattering Library
                             # with Atomic form factors from Kirklands book
    
# it is a good idea to show the version numbers at this point for archiving reasons.
__notebook_version__ = '2022.01.20'
print('pyTEM version: ', pyTEMlib.__version__)
print('notebook version: ', __notebook_version__)


# 
# ## Define  Crystal
# 
# Here we define silicon but you can use any other structure you like.

# In[10]:


#Initialize the dictionary with all the input
atoms = ks.structure_by_name('Si')
print(atoms.symbols)
print(atoms.get_scaled_positions())

#Reciprocal Lattice 
# We use the linear algebra package of numpy to invert the unit_cell "matrix"
reciprocal_unit_cell = np.linalg.inv(atoms.cell).T # transposed of inverted unit_cell


# ### Reciprocal Lattice
# 
# Check out [Basic Crystallography](CH2_03-Basic_Crystallography.ipynb) notebook for more details on this.

# In[11]:


#Reciprocal Lattice 
# We use the linear algebra package of numpy to invert the unit_cell "matrix"
reciprocal_lattice = np.linalg.inv(atoms.cell).T # transposed of inverted unit_cell

print('reciprocal lattice\n',np.round(reciprocal_lattice,3))


# ### 2D Plot of Unit Cell in Reciprocal Space

# In[13]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(reciprocal_lattice[:,0], reciprocal_lattice[:,2], c='red', s=100)
plt.xlabel('h (1/A)')
plt.ylabel('l (1/A)')
ax.axis('equal');


# ### 3D Plot of Miller Indices

# In[17]:


hkl_max = 2
h  = np.linspace(-hkl_max,hkl_max,2*hkl_max+1)  # all evaluated single Miller Indices
hkl  = np.array(list(itertools.product(h,h,h) )) # all evaluated Miller indices

# Plot 2D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hkl[:,0], hkl[:,2], hkl[:,1], c='red', s=10)
plt.xlabel('h')
plt.ylabel('l')
fig.gca().set_zlabel('k')
#ax.set_aspect('equal')


# ## Reciprocal Space and Miller Indices
# 
# For a reciprocal cubic  unit cell with lattice parameter $b = \frac{1}{a}$:
# 
# $$	
# \vec{g}_{hkl} = \begin{pmatrix}h\\k\\l\end{pmatrix}  \cdot	\begin{pmatrix}b&0&0\\0&b&0\\0&0&b\end{pmatrix} 
# $$			
# 
# Or more general
# 			
# $$			
# \vec{g}_{hkl} = \begin{pmatrix}h\\k\\l\end{pmatrix} \cdot 	\begin{pmatrix}b_{1,1}&b_{1,2}&b_{1,3}\\b_{2,1}&b_{2,2}&b_{2,3}\\b_{3,1}&b_{3,2}&b_{3,3}\end{pmatrix} 
# $$
# 
# 
# The matrix is of course the reciprocal unit cell or the inverse of the structure matrix.
# 
# Therefore, we get any reciprocal lattice vector with the dot product of its Miller indices and the reciprocal lattice matrix.
# 
# 
# 
# 
# Spacing of planes with Miller Indices $hkl$
# $$			\begin{align*}
# 			|\vec{g}_{hkl}|& = \frac{1}{d}\\
# 			d &= \frac{1}{|\vec{g}_{hkl}|}
# 			\end{align*}$$
# 			
# The length of a vector is called its **norm**.
# 
# 
# Be careful there are two different notations for the reciprocal lattice vectors:
# - materials science 
# - physics
# 
# The notations are different in a factor $2\pi$.  The introduction of  $2\pi$ in physics allows to take care of the $n$ more naturally.
# 
# In the materials science notation the reciprocal lattice points are directly associated with the Bragg reflections in your diffraction pattern. <br>
# (OK,s we are too lacy to keep track of $2\pi$)

# ### All Possible Reflections
# 
# Are then given by the all permutations of the Miller indices and the reiprocal unit cell matrix.
# 
# All considered Miller indices are then produced with the itertool package of python.
# 

# In[19]:


hkl_max = 6#  maximum allowed Miller index

h  = np.linspace(-hkl_max,hkl_max,2*hkl_max+1)   # all evaluated single Miller Indices
hkl  = np.array(list(itertools.product(h,h,h) )) # all evaluated Miller indices
g_hkl = np.dot(hkl,reciprocal_unit_cell)         # all evaluated reciprocal lattice points

print(f'Evaluation of {g_hkl.shape} reflections of {hkl.shape} Miller indices')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(g_hkl[:,0], g_hkl[:,2], g_hkl[:,1], c='red', s=10)
plt.xlabel('u (1/A)')
plt.ylabel('v (1/A)')
fig.gca().set_zlabel('w (1/A)')


# ### Atomic form factor
# 
# If we look at the scattering power of a single atom that deflects an electron:
#     
# ![Single atom scattering](images/scattering_single_atom.jpg)
# 
# See [Atomic Form Factor](CH2_02-Atomic_Form_Factor.ipynb) for details

# ## Calculate Structure Factors 
# 
# To extend the single atom view of the atomic form factor $f(\theta$)  to a crystal, we change to the structure factor $F(\theta)$. The structure factor  $F(\theta)$ is a measure
# of the amplitude scattered by a unit cell of a crystal
# structure. 
# 
# Because $F(\theta)$ is an amplitude like  $f(\theta)$, it also
# has dimensions of length.We can define  $F(\theta)$ as: 
# $$ F_{hkl}(\theta) = \sum_{j=1}^{\inf} f_i(\theta) \mathrm{e}^{[-2 \pi i (h x_j+k y_j + l z_j)]} 
# $$
# 
# The sum is over all the $i$ atoms in the unit cell (with
# atomic coordinates $x_i, y_i, z_i$ 
# 
# The structure factors $f_i(\theta)$ are multiplied by a phase factor (the exponential function).
# The phase factor takes account of the difference in phase
# between waves scattered from atoms on different but
# parallel atomic planes with the same Miller indices$ (hkl)$.
# 
# The scattering angle $\theta$  is the angle between the angle between the incident
# and scattered electron beams.
# 
# 
# Please identify all the variables in line 10 below. Please note that we only do a finite number of hkl

# In[20]:


# Calculate Structure Factors

structure_factors = []

base = atoms.positions  # positions in Carthesian coordinates
for j  in range(len(g_hkl)):
    F = 0
    for b in range(len(base)):
        f = ks.feq(atoms[b].symbol,np.linalg.norm(np.dot(g_hkl[j], reciprocal_lattice))) # Atomic form factor for element and momentum change (g vector)
        F += f * np.exp(-2*np.pi*1j*(g_hkl[j]*base[b]).sum())        
    structure_factors.append(F)
F = structure_factors = np.array(structure_factors)


# ### All Allowed Reflections
# 
# The structure factor determines whether a reflection is allowed or not.
# 
# If the structure factor is zero, the reflection is called forbidden.

# In[21]:


# Allowed reflections have a non zero structure factor F (with a  bit of numerical error)
allowed = np.absolute(structure_factors) > 0.001

print(f' Of the evaluated {hkl.shape[0]} Miller indices {allowed.sum()} are allowed. ')

distances = np.linalg.norm(g_hkl, axis = 1)
# We select now all the 
zero = distances == 0.
allowed = np.logical_and(allowed,np.logical_not(zero))

F = F[allowed]
g_hkl = g_hkl[allowed]
hkl = hkl[allowed]
distances = distances[allowed]


# ### Families of reflections
# 
# reflections with the same length of reciprocal lattice vector are called families

# In[22]:


sorted_allowed = np.argsort(distances)

distances = distances[sorted_allowed]
hkl = hkl[sorted_allowed]
F = F[sorted_allowed]

# How many have unique distances and what is their muliplicity

unique, indices  = np.unique(distances, return_index=True)

print(f' Of the {allowed.sum()} allowed Bragg reflections there are {len(unique)} families of reflections.')


# ### Intensities and Multiplicities
# 

# In[23]:


multiplicitity = np.roll(indices,-1)-indices
intensity = np.absolute(F[indices]**2*multiplicitity)
print('\n index \t     hkl \t 1/d [1/Ang]     d [pm] \t  F \t multip. intensity' )
family = []
for j in range(len(unique)-1):
    i = indices[j]    
    i2 = indices[j+1]   
    family.append(hkl[i+np.argmax(hkl[i:i2].sum(axis=1))])
    print(f'{i:3g}\t {family[j]} \t  {distances[i]:.2f}  \t {1/distances[i]*100:.0f} \t {np.absolute(F[i]):.2f}, \t  {indices[j+1]-indices[j]:3g} \t {intensity[j]:.2f}') 
    


# ## Allowed reflections for Silicon:   
# $\ \ |F_{hkl}|^2 =  \begin{cases} (  h , k , l \ \ \mbox{ all odd} &\\
#                     (  h ,| k , l \ \  \mbox{all even}& \mbox{and}\ \ h+k+l = 4n\end{cases}$ 
# 
# Check above allowed reflections whether this condition is met for the zero order Laue zone.
# 
# 
# Please note that the forbidden and alowed reflections are directly a property of the structure factor.

# ## Diffraction with parallel Illumination 
# 
# Polycrystalline Sample  |Single Crystalline Sample
# :---------:|:-----------------:
# ring pattern |spot pattern
# depends on $F(\theta)$ | depends on $F(\theta)$ 
# 		| depends on excitation error $s$

# ## Ring Pattern
# <img src="images/CL375.jpg" alt="Bragg's Law" width="300" >
# <img src="images/ProfileOfCL375.jpg" alt="Bragg's Law" width="300" >
# 
# **Ring Pattern:**
# - The profile of a ring diffraction pattern (of a polycrystalline sample) is very close to what a you are used from X-ray diffraction.
# - The x-axis is directly the magnitude of the $|\vec{g}| = 1/d$ of a hkl plane set.
# 	
# - The intensity of a Bragg reflection is directly related to the square of the structure factor $I = F^2(\theta)$
# 	
# - The intensity of a ring is directly related to the multiplicity of the family of planes. 
# 
# 
# **Ring Pattern Problem:**
# -  Where is the center of the ring pattern
# - Integration over all angles (spherical coordinates)
# - Indexing of pattern is analog to x-ray diffraction. 
# 
# The Ring Diffraction Pattern are completely defined by the Structure Factor

# In[24]:


from matplotlib import patches
fig, ax = plt.subplots()
plt.scatter(0,0);
img = np.zeros((1024,1024))
extent = np.array([-1,1,-1,1])*np.max(unique)
plt.imshow(img, extent = extent)

for radius in unique:   
    circle = patches.Circle((0,0), radius*2, color='r', fill= False, alpha = 0.3)#, **kwargs)
    ax.add_artist(circle);
    
plt.xlabel('scattering angle (1/$\AA$)');


# ## Conclusion
# The scattering geometry provides all the tools to determine which reciprocal lattice points are possible and which of them are allowed.
# 
# Next we need to transfer out knowledge into a  diffraction pattern.

# ## Navigation
# 
# - <font size = "3">  **Back Chapter 1: [Basic Crystallography](CH2_03-Basic_Crystallography.ipynb)** </font>
# - <font size = "3">  **Next: [Analyzing Ring Diffraction Pattern](CH2_05-Diffraction_Rings.ipynb)** </font>
# - <font size = "3">  **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# - <font size = "3">  **List of Content: [Front](../_MSE672_Intro_TEM.ipynb)** </font>
# 
# 

# In[ ]:




