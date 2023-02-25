#!/usr/bin/env python
# coding: utf-8

# 
# <font size = "5"> **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # Basic Crystallography
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Diffraction/CH2_03-Basic_Crystallography.ipynb)
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
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib')
# ------------------------------
print('done')


# ### Load the plotting and figure packages
# Import the python packages that we will use:
# 
# Beside the basic numerical (numpy) and plotting (pylab of matplotlib) libraries,
# * three dimensional plotting
# and some libraries from the book
# * kinematic scattering library.

# In[1]:


get_ipython().run_line_magic('matplotlib', ' notebook')
import matplotlib.pyplot as plt
import numpy as np

    
# 3D plotting package 
from mpl_toolkits.mplot3d import Axes3D # 3D plotting

import ase

# Import libraries from the pyTEMlib
import pyTEMlib
import pyTEMlib.kinematic_scattering as ks         # kinematic scattering Library
                             # with atomic form factors from Kirkland's book
# it is a good idea to show the version numbers at this point for archiving reasons.
print('pyTEM version: ',pyTEMlib.__version__)


# 
# ## Define  Crystal
# 
# A crystal is well defined by its unit cell and the atom positions within, the so called base.
# The base consists of which element sits where within the unit cell
# 
# 
# The unit cell fills the volume completely when translated in all three directions. Placing the unit cell in a global carthesian coordination system, we need the length of the sides and their angles for a complete description. This is depicted in the graph below.
# ![unitcell_angles](images/unit_cell_angles.png)
# 
# Figure taken from the wikipedia page on lattice constants.
# 
# Mathematically it is more advantageous to describe the unit cell as matrix, the
# ### Structure Matrix
# 
# This matrix consists of rows of vectors that span the unit cell:
# $\begin{bmatrix}
#   a_1 & a_2 & a_3 \\
#   b_1 & b_2 & b_3 \\
#   c_1 & c_2 & c_3 \\
# \end{bmatrix} =\left[\vec{a},\vec{b},\vec{c}\right]$.
# 
# This structure matrix is also used to describe the super cells in materials simulations for example density functional theory.
# 
# The representation of unit cells as structure matrices allows also for easy conversions as we will see in the following.
# 

# In[2]:


# Create graphite unit cell (or structure matrix)
a = b = 2.46  # Angstrom
c = 6.71  # Angstrom
gamma = 120
alpha = beta = 90

## Create the structure matrix for a hexagonal system explicitly:
structure_matrix = np.array([[a,0.,0.],  ## also called the structure matrix
                    [np.cos(np.radians(gamma))*a,np.sin(np.radians(gamma))*a,0. ],
                     [0.,0.,c]
                    ])
print('structure matrix \n', np.round(structure_matrix,3))

elements = ['C']*4
base = [[0, 0, 0], [0, 0, 1/2], [1/3, 2/3, 0], [2/3, 1/3, 1/2]]
print('elements:', elements)
print('base \n',np.round(base,3))


# ### Store Information in ASE (atomic simulation environment) format

# In[3]:


atoms = ase.Atoms(elements, cell=structure_matrix, scaled_positions=base)
atoms


# We can retrieve the information stored

# In[4]:


print('structure matrix [nm]\n',np.round(atoms.cell.array,3))
print('elements \n',atoms.get_chemical_formula())
print('base \n',np.round(atoms.get_scaled_positions(), 3))


# A convenient function is provided by the kinematic_scttering library (loaded with name ks)

# In[5]:


atoms = ks.structure_by_name('Graphite')
atoms.positions


# ### Volume of Unit Cell
# We will need the volume of the unit cell  for unit conversions later.
# 
# Volume of the parallelepiped (https://en.wikipedia.org/wiki/Triple_product) : 
# $\vec{a} \cdot \vec{b} \times \vec{c} =  \det \begin{bmatrix}
#   a_1 & a_2 & a_3 \\
#   b_1 & b_2 & b_3 \\
#   c_1 & c_2 & c_3 \\
# \end{bmatrix} ={\rm det}\left(\vec{a},\vec{b},\vec{c}\right)$
# 
# We see that the structure matrix comes in handy for that calculation.

# In[6]:


volume = v = np.linalg.det(structure_matrix)
print(f"volume of unit cell: {volume:.4f} Ang^3")


# The same procedure is provided by ase

# In[10]:


print(f"volume of unit cell: {atoms.cell.volume:.4f} Ang^3")


# ### Vector Algebra in Unit Cell 
# We will use the linear algebra package of numpy (np.linalg) for our vector calculations.
# 
# The length of a vector is called its norm.
# 
# And the angle between two vectors is calculated by the dot product: $\vec{a} \cdot \vec{b} = \left\| \vec{a} \right\| \left\| \vec{b} \right\| \cos (\theta) $
# 
# > Note that python starts couting at 0 and so the second vector has index 1

# In[11]:


length_b = np.linalg.norm(structure_matrix[1])
print(f'length of second unit cell vector is {length_b:.3f} Ang' ) 

gamma = np.arccos(np.dot(structure_matrix[0]/length_b, structure_matrix[1]/length_b))
print(f'angle between a and b is {np.degrees(gamma):.1f} degree')


# ### Plot the unit cell
# 
# We use the visualization library of ase to plot structures.

# In[12]:


from ase.visualize.plot import plot_atoms

plot_atoms(atoms, radii=0.3, rotation=('0x,1y,0z'))


# In[13]:


from ase.visualize import view
view(atoms*(4,4,1))


# In[14]:


from ase.visualize import view
view(atoms*(3,3,1), viewer = 'x3d')


# ## Reciprocal Lattice 
# The unit cell in reciprocal space

# In[15]:


reciprocal_lattice = np.linalg.inv(atoms.cell.array).T # transposed of inverted unit_cell

print('reciprocal lattice [1/Ang.]:')
print(np.round(reciprocal_lattice,4))


# The same function is provided in ase package of Cell.

# In[16]:


print('reciprocal lattice [1/Ang.]:')
print(np.round(atoms.cell.reciprocal(),4))


# ### Reciprocal Lattice Vectors
# From your crystallography book and lecture you are probably used to the following expression for the reciprocal lattice vectors ($\vec{a}^*, \vec{b}^*, \vec{c}^*$)
# 
# $ \begin{align}
#   \vec{a}^* &=  \frac{\vec{b} \times \vec{c}}{\vec{a} \cdot \left(\vec{b} \times \vec{c}\right)} \\
#   \vec{b}^* &=  \frac{\vec{c} \times \vec{a}}{\vec{b} \cdot \left(\vec{c} \times \vec{a}\right)} \\
#   \vec{c}^* &=  \frac{\vec{a} \times \vec{b}}{\vec{c} \cdot \left(\vec{a} \times \vec{b}\right)}
# \end{align}$\
# 
# Where we see that the denominators of the above vector equations are the volume of the unit cell.
# 
# In physics book, you will see an additional factor of 2$\pi$, which is generally omitted in materials science and microscopy.

# In[17]:


## Now let's test whether this is really equivalent to the matrix expression above.
a,b,c = atoms.cell

a_recip = np.cross(b, c)/np.dot(a, np.cross(b, c))
print (np.round(a_recip, 3))
b_recip = np.cross(c, a)/np.dot(a, np.cross(b, c))
print (np.round(b_recip, 3))
c_recip = np.cross(a, b)/np.dot(a, np.cross(b, c))
print (np.round(c_recip, 3))

print('Compare to:')
print(np.round(reciprocal_lattice, 3))


# ## Conclusion
# 
# With these definitions we have everything to define a crystal and to analyse diffraction and imaging data of crystalline specimens.
# 
# Crystallography deals with the application of symmetry and group theory of symmetry to crystal structures.
# If you want to play around with symmetry and space groups, you can install the [spglib](http://atztogo.github.io/spglib/python-spglib.html#python-spglib). The spglib is especially helpfull for determination of reduced unit cells (the smallest possible ones, instead of the ones with the full symmetry).
# 
# A number of common crystal structures are defined in the kinematic_scattering libary of the pyTEMlib package under the function ''structure_by_name''. Try them out in this notebook.

# In[18]:


# As ususal the help function will show you the usage of a function:
help(ks.structure_by_name)


# Here are all the predifined crystal structures.
# 
# > Check out the [building tutorial of ase](https://wiki.fysik.dtu.dk/ase/ase/build/build.html) for more fun structures like nanotubes 

# In[19]:


print(ks.crystal_data_base.keys())


# Now use one name of above structures and redo this notebook

# ## Navigation
# 
# - <font size = "3">  **Back: [Basic Crystallography](CH2_03-Basic_Crystallography.ipynb)** </font>
# - <font size = "3">  **Next: [Structure Factors](CH2_04-Structure_Factors.ipynb)** </font>
# - <font size = "3">  **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# - <font size = "3">  **List of Content: [Front](../_MSE672_Intro_TEM.ipynb)** </font>

# ## Appendix: Read POSCAR
# 
# Load and draw a  crystal structure  from a POSCAR file
#  

# ### The function 

# In[17]:


from ase.io import read, write
import pyTEMlib.file_tools as ft
import os

def read_poscar(): # open file dialog to select poscar file
    file_name = ft.open_file_dialog_qt('POSCAR (POSCAR*.txt);;All files (*)')
    #use ase package to read file
    
    base = os.path.basename(file_name)
    base_name = os.path.splitext(base)[0]
    crystal = read(file_name, format='vasp', parallel=False)
    
    return crystal


# In[ ]:


atoms = read_poscar()
atoms


# In[ ]:




