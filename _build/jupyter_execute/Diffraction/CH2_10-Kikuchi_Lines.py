#!/usr/bin/env python
# coding: utf-8

# 
# <font size = "5"> **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# # Kikuchi Lines
# 
# [Download](https://raw.githubusercontent.com/gduscher/MSE672-Introduction-to-TEM//main/Diffraction/CH2_10-Kikuchi_Lines.ipynb)
# 
# 
# part of 
# 
# <font size = "5"> **[MSE672:  Introduction to Transmission Electron Microscopy](../_MSE672_Intro_TEM.ipynb)**</font>
# 
# by Gerd Duscher, Spring 2021
# 
# Microscopy Facilities<br>
# Joint Institute of Advanced Materials<br>
# Materials Science & Engineering<br>
# The University of Tennessee, Knoxville
# 
# Background and methods to analysis and quantification of data acquired with transmission electron microscopes
# 

# ## Load relevant python packages
# ### Check Installed Packages

# In[ ]:


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
if test_package('pyTEMlib') < '0.2022.2.5':
    print('installing pyTEMlib')
    get_ipython().system('{sys.executable} -m pip install  --upgrade pyTEMlib -q')
# ------------------------------
print('done')


# ### Load the plotting and figure packages
# Import the python packages that we will use:
# 
# Beside the basic numerical (numpy) and plotting (pylab of matplotlib) libraries,
# * itertools (iterations of variables here to generate all possible Miller indices)
# * scipy.constants (all physical constants from scientific library scipy)
# some libraries from the book
# * kinematic scattering library.
# * diffraction_plot (which could be accessed through the scatteering library)

# In[2]:


# import matplotlib and numpy
#                       use "inline" instead of "notebook" for non-interactive plots
get_ipython().run_line_magic('pylab', ' notebook')
    
# 3D plotting package used
from mpl_toolkits.mplot3d import Axes3D # 3D plotting 

# additional package 
import itertools 
import scipy.constants as const

# Import libraries from the pyTEMlib
import pyTEMlib
import pyTEMlib.kinematic_scattering as ks         # Kinematic sCattering Library
                             # Atomic form factors from Kirklands book
from pyTEMlib import diffraction_plot as diff_plot


__notebook_version__ = '2022.02.21'

print('pyTEM version: ', pyTEMlib.__version__)
print('notebook version: ', __notebook_version__)


# ## Kikuchi Pattern
# 
# An electron can be scattered elastically into the Bragg angle, or it can be scattered inelastically
# 
# The inelastically scattered electrons are emitted in all directions.
# 
# These inelastically scattered electrons are then diffracted at the crystal planes according to Bragg's law. 
# 
# 
# ![Kossel cone](images/Kikuchi-small.png)
# 
# This results in a cone, where the Bragg diffraction is allowed; this cone is called Kossel cone. Because the electrons are predominantly scattered in the direction of the incident beam, this Bragg diffraction will take some intensity out of the original beam (dark line: deficit line in  the figure )and transfer it to a different angle (bright band: excess line in the figure. Inside this bands is some intensity which depends on the dynamic scattering.

# The most important feature of the Kikuchi band is that the Kossel cones are fixed to the crystal and allow easy navigation in reciprocal space, because nothing changes with the angle (the excitation error and thickness changes for SAD patterns). 
# A common example if you want to tilt out of a zone axis, but you also want to keep an interface edge on; you just follow the Kikuchi band of the interface plane.
# 
# Note: 
# >the Kikuchi bands are bent, they just appear straight, because the Ewald sphere is so large. 
# 
# Note: 
# >According to the explanation above, no Kikuchi lines should be visible if we are in a low order zone axis. This is not true because of the channeling effect and dynamic scattering, but these effects do not change anything about the location, where we expect the Kikuchi lines.

# 
# ## Constructing Kikuchi Maps
# ### Define  crystal
# 

# In[3]:


### Please choose another crystal like: Silicon, Aluminium, GaAs , ZnO
atoms = ks.structure_by_name('silicon')
atoms


# ### Plot the unit cell
# Just to be sure the crystal structure is right

# In[4]:


from ase.visualize.plot import plot_atoms

plot_atoms(atoms, radii=0.3, rotation=('0x,4y,0z'))


# ### Parameters for Diffraction Calculation
# 
# Please note that we are using a rather small number of reflections: the maximum number of Miller indices 
# > maximum hkl is 1

# In[5]:


tags = {}
atoms.info['experimental'] = {'acceleration_voltage_V': 20.0 *1000.0, #V
                              'convergence_angle_mrad': 0,
                              'zone_hkl': np.array([0,1,1]),  # incident neares zone axis: defines Laue Zones!!!!
                              'Sg_max': 0.05, # 1/Ang  maximum allowed excitation error ; This parameter is related to the thickness
                              'hkl_max': 1 }  # Highest evaluated Miller indices


# ###  Calculation

# In[6]:


ks.kinematic_scattering(atoms, False)


# The results are in the info attribute as a dictionary in *atoms.info['diffraction']*
# 
# ### Plot Selected Area Electron Diffraction Pattern

# In[7]:


# ########################
# Plot ZOLZ SAED Pattern #
# ########################

# Get information as dictionary
diffraction = atoms.info['diffraction']
#We plot only the allowed diffraction spots
points = diffraction['allowed']['g']
# we sort them by order of Laue zone
ZOLZ = diffraction['allowed']['ZOLZ']

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
# We plot the x,y axis only; the z -direction is set to zero - this is our projection
ax.scatter(points[ZOLZ,0], points[ZOLZ,1], c='red', s=40)

# zero spot plotting
ax.scatter(0,0, c='red', s=100)
ax.scatter(0,0, c='white', s=40)

ax.axis('equal')
FOV = .3
plt.ylim(-FOV,FOV); plt.xlim(-FOV,FOV);


# ## Kikuchi Line Construction
# The Kikuchi lines are the Bisections of lines from the center spot to the Bragg spots.
# 
# The line equation for a bisection of a line between two points $(A(x_A,y_A), B(x_B,y_B))$ is  given by the formula:
# 
# $y=-\frac{x_A-x_B}{y_A-y_B}x+\frac{x_A^2-x_B^2+y_A^2-y_B^2}{2 \cdot(y_A-y_B)}$
# 
# If $y_A = y_B$, then x is constant at $x= \frac{1}{2} (x_A+x_B)$
# 
# In our case  point $B$ is $(0,0)$ and so above equation is:
# 
# $y=-\frac{x_A}{y_A}x+\frac{x_A^2+y_A^2}{2 y_A}$ 
# 
# If $y_A$ is zero, the line is horizontal and $ x$ is constant at $x= \frac{1}{2} x_A$.

# In[8]:


pointsZ = points[ZOLZ]

g = pointsZ[1,0:2]

x_A, y_A = g


slope = -x_A/y_A
y_0 = (x_A**2+ y_A**2)/(2*y_A)

# Starting point of Kikuchi Line
x1 = -FOV
y1 = y_0+slope*x1
# End point of Kikuchi Line
x2= FOV
y2 = y_0+slope*x2


print(([x1,x2],[y1,y2]))
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
# We plot the x,y axis only; the z -direction is set to zero - this is our projection
ax.scatter(points[ZOLZ,0], points[ZOLZ,1], c='red', s=20 , alpha = .3)
ax.scatter(g[0], g[1], c='red', s=40)
# Draw kikuchi
ax.plot([x1,x2],[y1,y2],c='r')
ax.text(g[0]/2+0.04,g[1]/2+0.05, 'Kikuchi',color ='r', rotation=37)
ax.plot([0,g[0]],[0,g[1]],c='black')

# zero spot plotting
ax.scatter(0,0, c='red', s=100)
ax.scatter(0,0, c='white', s=40)

ax.axis('equal')
FOV = .3
plt.ylim(-FOV,FOV); plt.xlim(-FOV,FOV); plt.show()
plt.ylabel('angle (1/$\AA$)' )


# Assume we have an SAD pattern, from experiment or simulation. Draw a line from the center to each ZOLZ diffraction spot and draw a line perpendicular to the first line at half the distance. We name the line with the same Miller indices as the spot. The distance between two pair of lines is then again $|\vec{g}|$ of the original spots. 
# 
# If we know in which direction the next low order zone axis is and how far we have to tilt to get to it, we can draw that pole too.

# In[9]:


pointsZ = points[ZOLZ]

g = pointsZ[0,0:2]

x_A, y_A = g
print(x_A, y_A)
slope = -x_A/y_A
y_0 = (x_A**2+ y_A**2)/(2*y_A)

x1 = -FOV
y1 = y_0+slope*x1

x2= FOV
y2 = y_0+slope*x2


print(([x1,x2],[y1,y2]))
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
# We plot the x,y axis only; the z -direction is set to zero - this is our projection
ax.scatter(points[ZOLZ,0], points[ZOLZ,1], c='red', s=20 , alpha = .3)
ax.scatter(g[0], g[1], c='red', s=40)
ax.plot([x1,x2],[y1,y2],c='r')
ax.text(x1/2,g[1]/2-.5, 'Kikuchi',color ='r', rotation=52)
ax.plot([0,g[0]],[0,g[1]],c='black')

# Make right angle symbol
ax.scatter(g[0]/2, g[1]/2)
#ax.plot([g[0]/2*.09, g[0]/2*.9+.01], [g[1]/2*.9,g[1]/2*.9+.01],c='black') 
#ax.plot([g[0]/2*.9, g[0]/2*.9+.01],[g[1]/2*1.1,g[1]/2*1.1-.01],c='black') 

# zero spot plotting
ax.scatter(0,0, c='red', s=100)
ax.scatter(0,0, c='white', s=40)

ax.axis('equal')
FOV = .4
plt.ylim(-FOV,FOV); plt.xlim(-FOV,FOV); plt.show()


# ### Plotting of Kikuchi Pattern

# In[10]:


pointsZ = points[ZOLZ]

g = pointsZ[:,0:2]

FOV = .4
slope = -g[:,0]/g[:,1]
y_0 = (g[:,0]**2+ g[:,1]**2)/(2*g[:,1])

x1 = -FOV
y1 = y_0+slope*x1

x2= FOV
y2 = y_0+slope*x2


# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
# We plot the x,y axis only; the z -direction is set to zero - this is our projection
ax.scatter(points[ZOLZ,0], points[ZOLZ,1], c='red', s=20 )
ax.plot([x1,x2],[y1,y2],c='r', alpha = 0.5)




# zero spot plotting
ax.scatter(0,0, c='red', s=100)
ax.scatter(0,0, c='white', s=40)

ax.set_aspect('equal')
FOV = .6
plt.ylim(-FOV,FOV); plt.xlim(-FOV,FOV); plt.show()


# ## Plotting of Whole Kikuchi Pattern
# with a few more Bragg peaks, so please increase **hkl_max** and see what happens!
# 

# In[11]:


# ------- Input -----------
hkl_max = 3
# -------------------------

tags = atoms.info['experimental']
tags['hkl_max'] = hkl_max
tags['crystal_name'] = 'silicon'
ks.kinematic_scattering(atoms, False)
tagsD = atoms.info['diffraction']

#We plot only the allowed diffraction spots
points = tagsD['allowed']['g']
# we sort them by order of Laue zone
ZOLZ = tagsD['allowed']['ZOLZ']

pointsZ = points[ZOLZ]
g = pointsZ[:,0:2]

FOV = .6

g[g[:,1] ==0. , 1] = 1e-12  # to avoid division by zero
slope = -g[:,0]/g[:,1]
y_0 = (g[:,0]**2+ g[:,1]**2)/(2*g[:,1])

x1 = -FOV
y1 = y_0+slope*x1
x2= FOV
y2 = y_0+slope*x2

# Plot
fig = plt.figure()
plt.title(f" {tags['crystal_name']} in {tags['zone_hkl']}")
ax = fig.add_subplot(111)
# We plot the x,y axis only; the z -direction is set to zero - this is our projection
ax.scatter(points[ZOLZ,0], points[ZOLZ,1], c='red', s=20 )
ax.plot([x1,x2],[y1,y2],c='r', alpha = 0.5)


# zero spot plotting
ax.scatter(0,0, c='red', s=100)
ax.scatter(0,0, c='white', s=40)

ax.set_aspect('equal')
FOV = .6
plt.ylim(-FOV,FOV); plt.xlim(-FOV,FOV); plt.show()


# ## Plotting of Whole Kikuchi Pattern with kineamtic_scattering Library

# In[12]:


atoms = ks.structure_by_name('silicon')
atoms.info['experimental'] = {'crystal_name': 'silicon',
                              'acceleration_voltage_V': 200.0 *1000.0, #V
                              'convergence_angle_mrad': 0,
                              'Sg_max': .03,   # 1/Ang  maximum allowed excitation error ; This parameter is related to the thickness
                              'hkl_max': 9,   # Highest evaluated Miller indices
                              'zone_hkl': np.array([1, 1, 1]),  
                              'mistilt alpha degree': .0,  # -45#-35-0.28-1+2.42
                              'mistilt beta degree': 0.,
                              'plot_FOV': 2}

ks.kinematic_scattering(atoms)
        
atoms.info['output'].update(diff_plot.plotSAED_parameter())

atoms.info['output'].update({'plot Kikuchi': True,
                             'color Kikuchi': 'navy',
                             'background': 'gray',
                             'label size': 12,
                             'plot shift x': 0,
                             'plot shift y': 0,
                             'image gamma': 2,
                             'label color': 'skyblue',
                             'color zero': 'white',
                             'plot_labels': False})

############
## OUTPUT
############

diff_plot.plot_diffraction_pattern(atoms)
plt.gca().set_xlim(-2,2)
plt.gca().set_ylim(-2,2)


# ## Mistilt  in Kikuchi Pattern
# 
# Becasue Kikuchi pattern are directly fixed to the crystal any mistiolt can immediately be detected
# 

# In[16]:


# -------Input ------ # 
mistilt_alpha = 2  # in degree
mistilt_beta = -2.0
# ------------------- #


# Crystal unit cell defintion 
crystal_name = 'Aluminum'
atoms2 = ks.structure_by_name(crystal_name)

atoms2.info['experimental'] = {'crystal_name': crystal_name,
                              'acceleration_voltage_V': 200.0 *1000.0, #V
                              'convergence_angle_mrad': 0.,
                              'Sg_max': .03,   # 1/Ang  maximum allowed excitation error ; This parameter is related to the thickness
                              'thickness': 20,  # in Ang
                              'hkl_max': 11,   # Highest evaluated Miller indices
                              'zone_hkl': np.array([0, 0, 1]),  
                              'mistilt_alpha degree': mistilt_alpha,  
                              'mistilt_beta degree': mistilt_beta,
                              'plot_FOV': 2}

ks.kinematic_scattering(atoms2)
        
atoms2.info['output']=diff_plot.plotSAED_parameter()

atoms2.info['output'].update({'plot Kikuchi': False,
                             'color Kikuchi': 'navy',
                             'background': 'gray',
                             'label size': 12,
                             'plot shift x': 0,
                             'plot shift y': 0,
                             'image gamma': 2,
                             'label color': 'skyblue',
                             'color zero': 'white',
                             'plot_labels': False})


############
## OUTPUT
############

diff_plot.plot_diffraction_pattern(atoms2, grey=False)
plt.gca().set_xlim(-2, 2)
plt.gca().set_ylim(-2, 2)


# ## Kikuchi Maps
# 
# Extremly helpfull for tilting a sample to a different zone axis are maps of Kikuchi pattern as the one below.
# 
# ![Diamond Kikuchi Map](images/Diamondkikuchi.png)
# 

# If you are in a specific zone axis, which we identify by its symmetry (like the SAED 

# ## Kikuchi Lines and Excitation Error
# 
# Since the Kikuchi lines are rigidly attached to the crystal, we can use them to measure the excitation error. The diffraction geometry is shown in the figure above.  If $\vec{s}_g = 0$ the deficient line of a Kikuchi band (a pole) lays in the origin. If we tilt the sample so that the whole band shifts 
# to the bright line, then the excitation error is positive. The angle $\eta$ (which is equal to $\epsilon$) is related to the excitation error by simple trigonometry:
# \begin{eqnarray}
# \eta &=& \frac{x}{L}= \frac{x \lambda}{R d}\\
# \epsilon &=& \frac{s}{g}\\
# s &=& \epsilon = \frac{x}{L}g = \frac{x}{Ld}\\
# \frac{R}{L}&=& 2\theta_B=\frac{\lambda}{d}\\
# s&=& \frac{xL}{d}=\frac{x}{d}\cdot\frac{\lambda}{Rd}\\
# s&=& \frac{x\lambda}{R d^2}=\frac{x}{R}\lambda g^2
# \end{eqnarray}

# In[22]:


# --- Input -----
s_g= -.03 # in 1/Ang 
# ---------------

plt.figure()
from pyTEMlib import animation
animation.deficient_kikuchi_line(s_g= s_g)


# Principle of excitation error determination. You can use this to obtain exact Bragg (s_g=0) conditions  in your diffraction pattern.

# ## Conclusion
# The Kikuchi lines are directly related to the Bragg reflections and therefore show the same symmetry as the diffraction pattern.
# 

# ## Navigation
# 
# - <font size = "3">  **Back: [Unit Cel Determination](CH2_09-Unit_Cell.ipynb)** </font>
# - <font size = "3">  **Next: [HOLZ Lines](CH2_11-HOLZ_Lines.ipynb)** </font>
# - <font size = "3">  **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# - <font size = "3">  **List of Content: [Front](../_MSE672_Intro_TEM.ipynb)** </font>

# In[ ]:





# ## Appendix
# ### An Excursion to Hough space
# 
# The line can be defined by 
# - its end points
# - intersection with y and slope
# or 
# - closest distance to origin and angle with x-axis.
# 
# The latter is called **Hough Space** and when we look at the Kikuchi construction we see that this is easily accomplished in that case:
# - The closest distance to origin is half the distance to the Bragg reflection.
# - The angle between the x-axis and the angle to the Bragg reflection plus 90$^o$ is the Hough line angle
# 
# > That means that polar coordinates and Hough space are closely related

# In[63]:


points_polar = np.array([np.linalg.norm(points[:,:2], axis=1), np.arctan2(points[:,1],points[:,0])], dtype=float).T
points_polar


# In[15]:


def hough2cartesian(points_polar, maxlength = .4):
    """ converts polar coordinates to lines and cartesian points """
    dist = points_polar[:, 0]
    angle = points_polar[:, 1]

    (x0, y0) = dist/2 * np.array([np.cos(angle), np.sin(angle)])
    
    h_xp = x0 + maxlength * np.cos(angle- np.pi/2)
    h_yp = y0 + maxlength * np.sin( angle-np.pi/2)
    h_xm = x0 - maxlength * np.cos( angle-np.pi/2)
    h_ym = y0 - maxlength * np.sin( angle-np.pi/2)
    
    return h_xp, h_yp, h_xm, h_ym, x0*2, y0*2

def cartesian2hough(points, maxlength = .4):
    """ converts cartesian coordinates to lines and polar points """
    
    dist = np.linalg.norm(points[:,:2], axis=1)
    angle = np.arctan2(points[:,1],points[:,0])
    
    h_xp = points[:,0]/2 + maxlength * np.cos(angle-np.pi/2)
    h_yp = points[:,1]/2 + maxlength * np.sin(angle-np.pi/2)
    h_xm = points[:,0]/2 - maxlength * np.cos(angle-np.pi/2)
    h_ym = points[:,1]/2 - maxlength * np.sin(angle-np.pi/2)
    
    return h_xp, h_yp, h_xm, h_ym, dist, angle

points_polar = np.array([np.linalg.norm(points[:,:2], axis=1), np.arctan2(points[:,1],points[:,0])], dtype=float).T

hxp,hyp,hxm,hym, x0, y0 = hough2cartesian(points_polar)

plt.figure()
plt.scatter(x0, y0, c='red', s=20 , alpha = .3),
plt.scatter(0, 0, c='red', s=20 , alpha = .8),
for i in range(len(hxm)):
    plt.plot([hxm[i], hxp[i]], [hym[i], hyp[i]])
plt.gca().set_aspect('equal')


# Polar coordinates allow us to rotate the diffraction pattern quite easily, just add a fixed value to the angle part of the coordinates

# In[16]:


# ------Input -----
rotation_angle = 10 # in degree
# -----------------
points_polar = np.array([np.linalg.norm(points[:,:2], axis=1), np.arctan2(points[:,1],points[:,0])], dtype=float).T
points_polar[:,1] -= np.deg2rad(rotation_angle)

hxp,hyp,hxm,hym, x0, y0 = hough2cartesian(points_polar)

plt.figure()
plt.scatter(x0, y0, c='red', s=20 , alpha = .3),
plt.scatter(0, 0, c='red', s=20 , alpha = .8),
for i in range(len(hxm)):
    plt.plot([hxm[i], hxp[i]], [hym[i], hyp[i]])
plt.gca().set_aspect('equal')


# ### Hough-Transform
# From Wikipedia
# 
# ![Hough Transform](images/Hough-example-result-en.png)
# 

# In[ ]:




