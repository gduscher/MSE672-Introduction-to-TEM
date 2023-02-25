#!/usr/bin/env python
# coding: utf-8

# 
# 
# <font size = "5"> **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # CBED - Multislice Algorithm
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Diffraction/CCH2_D02-Multi_Slice.ipynb)
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

# ## Introduction
# In this notebook, we will make a dynamic simulation of the scattering process.
# 
# The core of this algorithm is used in many different libraries.
# 
# I follow in my description mostly the book of Kirkland, from where we also use the scattering parameters. The code has been completely rewritten so that the mechnaism of the algorithm can be understood on a basic level. 
# 
# All commercial and open-source codes use the same base algorithm but are highly tuned for usability, variabilty, features and speed. This was not the goal here.
# 
# Here are a few of the more common open-source programs
# - [abTEM](https://github.com/jacobjma/abTEM)
# - [clTEM](https://github.com/JJPPeters/clTEM)
# - [MuSTEM](https://github.com/ningustc/MuSTEM)
# - [Dr. Probe](https://er-c.org/barthel/drprobe/)
# 
# 
# ## Load relevant python packages
# ### Check Installed Packages
# There is a new pyTEMlib package, so make sure you run this cell

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
if test_package('pyTEMlib') < '0.2022.3.0':
    print('installing pyTEMlib')
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib -q')
# ------------------------------
print('done')


# ### Import numerical and plotting python packages
# Import the python packages that we will use:
# 
# Beside the basic numerical (numpy) and plotting (pylab of matplotlib) libraries, we need 3D plotting library and some scipy libraries
# 
# and a library from pyTEMlib:
# * kinematic scattering library.

# In[2]:


get_ipython().run_line_magic('pylab', '--no-import-all notebook')

# additional package 
from mpl_toolkits.mplot3d import Axes3D 
import itertools
import scipy.constants
import scipy.special 

import sys
sys.path.insert(0,'../../pyTEMlib')
# Import libraries from pyTEMlib
import pyTEMlib
import pyTEMlib.kinematic_scattering as ks         # kinematic scattering Library
                              # Atomic form factors from Kirkland's book
import pyTEMlib.dynamic_scattering 

# For archiving reasons it is a good idea to print the version numbers out at this point
print('pyTEM version: ',pyTEMlib.__version__)

__notebook__ = 'CH2_D02-Multislice_CBED'
__notebook_version__ = '2022_02_25'


# ## Overview of Multislice Algorithm
# 
# We build on top of the [Multislice notebook](CH2_D02-Multislice.ipynb) 
# 
# 1. Make projected atomic potentials of slices.
# 2. The transmission function will deal with the distorition of the wave by the atom potentials.
# 3. The Fresnel propagator takes care of the vacuum between the atomic layers.
# 4. We need to define the `` convergent`` incident wave. 
# 5. We let this wave travel through the different layers, iteratively.
# 

# ## Step1: Projected Potential 
# 
# Providing the potentials and placing the atoms is a surprisingly computer intensive task for initialization of the simulation. This notebook gives up flexibility for speed in placing the atoms (no sub-pixel movement).
# 
# 

# ### Slice Crystal.
# 
# The above crystal is an artificial construct.
# 
# Now we make a real crystal. 
# 
# >
# > For the multislice alogrythm we need to make the slices.
# >
# 
# We do this here on the based on the unit cell 
# 

# In[3]:


atoms = ks.structure_by_name('SrTiO3')
for i in range(len(atoms)):
    print(i, atoms[i].symbol, atoms.get_scaled_positions()[i])

super_cell = ks.ball_and_stick(atoms)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Corners and Outline of unit cell
h = (0, 1)
corner_vectors = np.dot(np.array(list(itertools.product(h, h, h))), atoms.cell)
trace = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 7], [6, 7], [6, 4], [1, 5], [2, 6], [3, 7]]
corners = []
for s, e in trace:
    corners.append([*zip(corner_vectors[s], corner_vectors[e])])

    for x, y, z in corners:
        ax.plot3D(x, y, z, color='blue')

for i, atom in enumerate(super_cell.positions):
    ax.scatter(atom[0], atom[1], atom[2],
               color=tuple(ks.jmol_colors[super_cell.get_atomic_numbers()[i]]),
               alpha=1.0, s=50)
    
xx, yy = np.meshgrid(range(-1,6), range(-1,6))
z = xx*0+atoms.cell[2][2]
ax.plot_surface(xx, yy, z, alpha=0.5)
ax.plot_surface(xx, yy, z-atoms.cell[2][2]/2, alpha=0.5)
ax.text(5, 0., atoms.cell[2][2]*.7, "$\Delta z$", color='black')
ax.plot([5, 5], [0,0],zs=[atoms.cell[2][2]/2, atoms.cell[2][2]], color='black', linewidth=3)
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_zlim(-1, 5);


# Here we have two equidistant layers that we can read off the z-component of the base 
# - one layer with z = 0.0 and 
# - one layer with z = 0.5

# ### Make Potentials for Slice

# In[23]:


# ------Input --------
size_in_pixel = 512
n_cell_x = 32
# --------------------

atoms = ks.structure_by_name('SrTiO3')
lattice_parameter = atoms.cell[0, 0]
pixel_size = lattice_parameter/(size_in_pixel/n_cell_x)


for i in range(len(atoms)):
    print(i, atoms[i].symbol, atoms.positions[i]/pixel_size)
layers = {}
layers[0] ={0:{'element': 'Sr', 'base': [atoms.positions[0, 0:2]]}, 
            1:{'element': 'O',  'base': [atoms.positions[3, 0:2]]}}
layers[1] ={0:{'element': 'Ti', 'base': [atoms.positions[1, 0:2]]}, 
            1:{'element': 'O',  'base': atoms.positions[[2,4], 0:2]}} 


a = lattice_parameter

pixel_size = a/(size_in_pixel/n_cell_x)

image_extent = [0, size_in_pixel*pixel_size, size_in_pixel*pixel_size,0]
slice_potentials = np.zeros([2,size_in_pixel,size_in_pixel])
for layer in layers:
    for atom in layers[layer]:
        elem = layers[layer][atom]['element']
        pos = layers[layer][atom]['base']
        slice_potentials[layer] += pyTEMlib.dynamic_scattering.potential_2dim(elem, size_in_pixel, size_in_pixel, n_cell_x, n_cell_x, a, pos)
plt.figure()
#plt.imshow(layer_potentials.sum(axis=0))
plt.imshow(slice_potentials[1], extent = image_extent)
plt.xlabel('distance ($\)')
plt.show()


# ## Step 3: Transmission Function for Each Slice 
# 
# The slice acts like a **Very Thin Specimen** in the ``weak phase approximation``.
# In that approximation, the sample causes only a phase change to the incident plane wave.
# 
# To retrieve the exit wave of that slice we just multiply the transmission function $t(\vec{x})$ with the plane wave $\exp (2\pi i k_z z)$
# 
# $$ \Psi_t(\vec{x}) = t(\vec{x}) \exp \left(2 \pi i k_z z \right) \approx t(\vec{x})  $$
# 
# The specimen transmission function depends on the projected potential $v_z(\vec{x})$ and the interaction parameter $\sigma$:
# $$t(\vec{x}) =  \exp \left( i \sigma v_z(\vec{x})\right)$$
# 
# with the interaction parameter $\sigma$:
# $$ 
# \sigma = \frac{2 \pi}{\lambda V} \left(  \frac{m_0 c^2 + eV}{2m_0c^2+eV} \right) = \frac{2 \pi m  e_0 \lambda}{h^2}
# $$
# with $ m = \gamma m_0$ and $eV$ the incident electron energy.

# In[24]:


acceleration_voltage = 60*1e3

transmission = pyTEMlib.dynamic_scattering.get_transmission(slice_potentials, acceleration_voltage)

plt.figure()
plt.imshow(transmission[0].imag, extent = image_extent)
plt.xlabel('distance ($\AA$)');


# ## Step 4: Propagator
# The Fresnel propagator $p$ propagates the wave through the vacuum of the layers between the (flat) atom potentials.
# $$
# p(x,y, \Delta z) = \mathcal{F} P(k_x, k_y, \Delta z)
# $$
# Mathematically, this propagator function has to be  convoluted with the wave, which is a multiplication in Fourier space $\mathcal{F}$.
# 
# $$
# P(k,\Delta z) = \exp(-i\pi \lambda k^2 \Delta z)
# $$
# 
# The Fourier space is limited in reciprocal vector to avoid aliasing. We realize that with  an aperture function.
# 
# Here we assume a cubic crystal and equidistant layers, but that of course is not always true.

# In[25]:


lattice_parameter = atoms.cell[0,0]
field_of_view = n_cell_x*lattice_parameter
number_layers = 2
delta_z = [atoms.cell[2,2]/number_layers, atoms.cell[2,2]/number_layers]
wavelength = ks.get_wavelength(acceleration_voltage)

bandwidth_factor = 2/3   # Antialiasing bandwidth limit factor

propagator = pyTEMlib.dynamic_scattering.get_propagator(size_in_pixel, delta_z, number_layers, wavelength, field_of_view, 
                             bandwidth_factor, verbose=True)

recip_FOV = size_in_pixel/field_of_view/2.
reciprocal_extent = [-recip_FOV,recip_FOV,recip_FOV,-recip_FOV]
layer = 0
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
fig.suptitle(f"propagator of delta z = {delta_z[layer]:.3f} $\AA$")
ax[0].set_title(f"real part")
ax[0].imshow(propagator[0].real,extent=reciprocal_extent)
ax[0].set_xlabel('frequency (1/$\AA$)')
ax[1].set_title(f"imaginary part")
ax[1].set_xlabel('frequency (1/$\AA$)')
ax[1].imshow(propagator[0].imag,extent=reciprocal_extent)


# ## Step 5: Incident Wave
# 
# The definition of the incident wave is the only fundamental change compared to the [SAED multislice notebook](CH2_D02-Multislice.ipynb)
# 
# We take advantage from the fact that aperfect convergent probe in reciprocal space is the Fourier transform of an aperture function (probe forming aperture) which is perfectly coherently illuminated (all amplitudes and phases the same). 
# 
# I'll use the fft_shift function to put the probe in the middle, which is on top of a Sr column.
# 

# In[26]:


# --- Input ------------
convergence_angle = 0.12 # in 1/Ang
# ----------------------
print(f'Convergence angle is {wavelength*convergence_angle*1000:.2f} mrad')
# aperture function
dk = 1 / field_of_view
k = np.array(dk * (-size_in_pixel / 2. + np.arange(size_in_pixel)))
t_xv, t_yv = np.meshgrid(k, k)
theta = np.sqrt(t_xv ** 2 + t_yv ** 2)

aperture_function = np.zeros([size_in_pixel,size_in_pixel], dtype=complex)
aperture_function[theta<convergence_angle] = 1.

incident_wave = np.fft.fftshift(np.fft.ifft2(aperture_function))

plt.figure()
plt.imshow(np.abs(incident_wave), extent = image_extent)
plt.xlabel('distance ($\AA$)')
plt.show()


# ## Step 6: Multislice Loop
# 
# Combining the transmission function $t$ and the Frensel propagator $p$ we get
# for each slice:
# $$
# \Psi(x,y,z+\Delta z) = p(x,y,\Delta z) \otimes \left[t(x,y,z)\Psi(x,y,z) \right] + \mathcal{O}(\Delta z^2)
# $$
# 
# or an expression that bettere relfects the iterative nature of this equation for starting layer n :
# 
# $$
# \Psi_{n+1}(x,y,) = p_n(x,y,\Delta z) \otimes \left[t_n(x,y,z)\Psi_n(x,y,z) \right] + \mathcal{O}(\Delta z^2)
# $$
# 
# Again the convolution $\otimes$ will be done as a multiplication in Fourier space.

# In[27]:


# ------Input------------- #
number_of_unit_cell_z = 40 # this will give us the thickness
# ------------------------ #    

number_of_layers = 2
exit_wave = pyTEMlib.dynamic_scattering.multi_slice(incident_wave, number_of_unit_cell_z, number_of_layers, transmission, propagator)
    
print(f"simulated {atoms.info['title']} for thickness {number_of_unit_cell_z*atoms.cell[0, 0]/10:.3f} nm")

wave = np.fft.fft2(exit_wave)
intensity = np.abs(np.fft.fftshift(np.fft.ifft2(wave*np.conjugate(wave))))

plt.figure()
plt.title('intensity of exit wave')
plt.imshow(intensity, extent = image_extent)
plt.xlabel('distance ($\AA$)');


# ## Diffraction Pattern
# 
# according to [J. M. Cowley and A. F. Moodie](https://doi.org/10.1107/S0365110X57002194), the diffraction pattern $U(u,v)$ is the Fourier transform of the exit wave function in the far-field (diffraction plane): 
# 
# $$
# U(u,v) = \mathcal{F} \left( \exp(i*\varphi(x,y)) \right)
# $$
# 
# where $\varphi$ is the exit wave of the crystal.

# In[28]:


# -----Input-----------
additional_layers = 100
# ---------------------
diffraction_pattern =  np.fft.fft2(np.exp(1j*exit_wave))
diffraction_pattern[0,0] = 0

# adding a multislice calculation
exit_wave2 = pyTEMlib.dynamic_scattering.multi_slice(exit_wave, additional_layers, number_of_layers, transmission, propagator)
diffraction_pattern2 = np.fft.fft2(np.exp(1j*exit_wave2))
diffraction_pattern2[0,0] = 0

fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
fig.suptitle(f"simulated diffraction patterns of {atoms.info['title']}")
ax[0].set_title(f"thickness {number_of_unit_cell_z*atoms.cell[0, 0]:.3f} $\AA$")
ax[0].imshow(np.power(np.abs(diffraction_pattern), 1), extent=reciprocal_extent,vmax= 0.2)
ax[0].set_xlabel('spatial frequency (1/$\AA$)')
ax[1].set_title(f"thickness {(number_of_unit_cell_z+additional_layers)*atoms.cell[0,0]:.3f} $\AA$")
ax[1].imshow(np.power(np.abs(diffraction_pattern2),1), extent=reciprocal_extent)
ax[1].set_xlabel('spatial frequency (1/$\AA$)')

ax[1].set_aspect('equal')


# ## Sampling in Real- and Reciprocal space
# 
# The real-space sampling is extremely important because it controls the accuracy of the simulation at high scattering angles. Especially in CBED calculations, we ususally want to see the zero disk with lots of pixels and include the HOLZ rings, so we need to think about sampling in real space affects the reciprocal space.
# 
# The sampling defines the maximum spatial frequency $k_{max}$
# 
# via the formula:
# 
# $$k_{max}=\frac{2}{p},$$
# 
# where $p$ is the real-space sampling distance otherwise called pixel size. 
# 
# Because, we have the same number of pixels in real and reciprocal space the sampling of one space influences the other according to above equation.
# 
# To counteract aliasing artifacts due to the periodicity assumption of a discrete 
# Fourier transform, abTEM supresses spatial frequencies above $\frac{2}{3}$ of the 
# maximum scattering angle, further reducing the maximum effective scattering angle by a 
# factor of $\frac{2}{3}$. 
# 
# Hence the maximum scattering angle $\alpha_{max}$
# is given by:
# $$\alpha_{max} =  \frac{2}{3} \frac{\lambda}{2p}$$
# 
# where $\lambda$ is the relativistic electron wavelength.
# 
# >
# >As an example, consider a case where we want to simulate 80 keV electron scattering up 
# to angles of 200 mrads. Plugging these values into the above equation gives a 
# sampling of ∼0.0052 nm.
# >
# >Therefore, we require at least 5pm pixel size in order to reach a maximum scattering angle of 200 mrads. 
# 
# **In practice, you should ensure that the simulation is converged with respect to pixel size.**
# 
# >
# > So repeat above notebook with a large number of uni cells to see the effect on the diffraction pattern
# >
# 

# ## Summary
# 
# The multislice algrithm allows to calculate all the dynamic diffraction effects we need. 
# 
# Changing the incident wave from a palne to a convergent one is all one needs to do to simulate CBED patterns.
# 
# More sophisticated program allowmore flexibility and include frozen phonon calculatins and differ
# 
# 

# ## Navigation
# 
# 
# - <font size = "3">  **Back: [Mutli-Slice Theory](CH2_D02-Multislice.ipynb)** </font>
# - <font size = "3">  **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# - <font size = "3">  **List of Content: [Front](../_MSE672_Intro_TEM.ipynb)** </font>

# In[ ]:




