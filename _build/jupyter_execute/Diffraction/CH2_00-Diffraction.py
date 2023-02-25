#!/usr/bin/env python
# coding: utf-8

# <font size = "5"> **Chapter 2: [Diffraction](CH2_00-Diffraction.ipynb)** </font>
# 
# 
# <hr style="height:1px;border-top:4px solid #FF8200" />
# 
# #  Chapter 2:  Diffraction
# 
# part of 
# 
# <font size = "5"> **[MSE672:  Introduction to Transmission Electron Microscopy](../_MSE672_Intro_TEM.ipynb)**</font>
# 
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
# ## Content
# 
# The Diffraction chapter has the following sections:
# 
# ### Kinematic Scattering
# * [The Electron](CH2_01-Electron.ipynb)
# * [Atomic Form Factor](CH2_02-Atomic_Form_Factor.ipynb)
# * [Basic Crystallography](CH2_03-Basic_Crystallography.ipynb)
# * [Structure Factors](CH2_04-Structure_Factors.ipynb)
# * [Analyzing Ring Diffraction Pattern](CH2_05-Diffraction_Rings.ipynb)
# * [Kinematic Scattering Geometry](CH2_06-Kinematic_Scattering_Geometry.ipynb)
#     * [Relrod - Sample Geometry and Excitation Error](CH2_06b-Relrod.ipynb)
# * [Plotting of Diffraction Pattern](CH2_07-Plotting_Diffraction_Pattern.ipynb)
# * [Analyzing Spot Diffraction Pattern](CH2_08-Spot_Diffraction_Pattern.ipynb)
# * [Unic Cell Determination and Stereographic Projection](CH2_09-Unit_Cell.ipynb)
# * [Kikuchi Lines](CH2_10-Kikuchi_Lines.ipynb)
# * [HOLZ Lines](CH2_11-HOLZ_Lines.ipynb)
# * [Lattice Determination with HOLZ](CH2_12-HOLZ_Example.ipynb) 
# 
# ### Dynamic Scattering
# 
# * [Bloch Waves](CH2_D01-Bloch.ipynb)
# * [Multislice Algorithm](CH2_D02-Multislice.ipynb)
# * [CBED with Multisice](CH2_D03-Multislice_CBED.ipynb)
# 
# 
# # Introduction
# 
# ## Basics
# 
# - Diffraction is the direct result of the interaction (with or without
# energy transfer) of electrons and matter.
# - Kinematic diffraction theory describes only the Bragg angles (the position of the Bragg reflections) but not the intensity in a real
# diffraction pattern.
# - Dynamic theory is responsible for the intensity variation of the Bragg reflections
# 
# ## Diffraction and Imaging
# 
# - To achieve an image from a diffraction pattern only a **Fourier Transformation** of parts of the diffraction pattern is needed.
# - Any image in a TEM can be described as **Fourier Filtering**, because we select the beams which form the images. The knowledge of which and how many diffracted beams contribute to the image formation is crucial.
# - Because the intensity of selected diffracted beams is necessary to calculate image intensities, dynamic diffraction theory is necessary.
# - Understanding difraction theory of electrons is at the core of the analysis of TEM data.
# 
# 
# >According to *Fourier-Optical* view of the TEM, we first form a diffraction pattern (in Fourier space) and after another (inverse) Fourier
# Transformation of parts of this diffraction pattern we are back in real space and can observe an image.
# 
# ## Dynamic and Kinematic Theory
# 
# 
# 
# 
# - Kinematic Theory is based on a single scattering event per electron
# - Kinematic theory is used in neutron and X-ray diffraction almost exclusively.
# 
# - Dynamic theory incorporates multi scattering events.
# - Dynamic theory results in *Rocking Curves* (oscillations) of intensities of diffracted beams with sample thickness.
# - Dynamic theory can analytically be solved only for the two beam case, strangely the basis for conventional TEM.
# 

# ## Diffraction and Scattering
# 
# Electrons can be viewed as particles and/or as waves.
# 
# ### Particle-wave dualism:
# 
#  <table style="width:80%">
#   <tr>
#     <th>Scattering</th>
#     <th>$\leftrightarrow$</th>
#     <th>Diffraction</th>
#   </tr>
#   <tr>
#     <td>Particle picture</td>
#     <td>$\leftrightarrow$</td>
#     <td>Wave picture</td>
#   </tr>
#  
# </table> 
# 
# We will switch back and forth between these two pictures, depending on which is mathematically easier to express.
# 
# 
# ## Kinematic Diffraction Theory Buzz Words
# 
# The following terms will be important in Kinematic Theory:
# 
#  <table style="width:80%">
#  
#   <tr>
#     <td>f</td>
#     <td> atomic scattering factor</td>
#     <td>scattering strength of an atom </td>
#         
#   </tr>
#   <tr>
#     <td>F</td>
#     <td>form factor</td>
#       <td>combination of symmetrry and atomic scattering factor</td>
#   </tr>
#     <tr>
#     <td></td>
#     <td>forbidden reflection</td>
#       <td>a direct result of the form factor</td>
#   </tr>
#   <tr>
#     <td>$\sigma$</td>
#     <td>cross-section</td>
#       <td>scattering probability expressed as an effective area </td>
#   </tr>
#   <tr>
#     <td>$\frac{\partial \sigma}{\partial \Omega}$</td>
#     <td>cross-section</td>
#     <td> scattering probability in a solid angle </td>
#   </tr>
#   <tr>
#     <td>$\lambda$</td>
#     <td>mean free path</td>
#     <td> scattering probability expressed as a path-length between two scattering events</td>
#   </tr> 
# </table> 
# 
# We will start discussiong this terms in the [Atomic Form Factor](CH2_02-Atomic_Form_Factor.ipynb) and following notebooks

# ## Navigation
# - <font size = "3">  **Back Chapter 1: [Introduction](../Introduction/CH1_00-Introduction.ipynb)** </font>
# - <font size = "3">  **Next: [The Electron](CH2_01-Electron.ipynb)** </font>
# - <font size = "3">  **List of Content: [Front](../_MSE672_Intro_TEM.ipynb)** </font>
# 
# 

# In[ ]:




