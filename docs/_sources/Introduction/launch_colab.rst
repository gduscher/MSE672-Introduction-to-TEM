
Launch Google Colab
===========================

When you launch a Google Colab notebook, much of the software we will use in class is already installed. It is not always the latest version of the software, however. In fact, as of early January 2021, Colab is running Python 3.6, whereas you will run Python 3.8 on your machine through your Anaconda installation. Nonetheless, all of the analyses we do for this class will work just fine in Colab.

Because the notebooks in Colab have software preinstalled, and no more, you will often need to install software before you can run the rest of the code in a notebook. To enable this, when necessary, in the first code cell of each notebook in this class, we will have the following code (or a variant thereof depending on what is needed or if the default installations of Colab change). Running this code will not affect running your notebook on your local machine; the same notebook will work on your local machine or on Colab.
::
			  
	# Colab setup ------------------
	%%capture
	import os, sys
	if "google.colab" in sys.modules:
		!pip install pyTEMlib
	# ------------------------------
			  
		  

In addition to installing the necessary software on a Colab instance, we also have to have access to the data we want to analyse. We expect these data to be on your google drive. You will have to follow the directions to give permission to the notebook to access these data.
::
	
	# Google Drive setup -----------
	if "google.colab" in sys.modules:
		from google.colab import drive
		drive.mount("/content/drive")
	# ------------------------------
			  
		 

When running on your local machine, the path to the data can be chosen without restrictions.

Collaborating with Colab
------------------------
If you want to collaborate with another student or with the course staff on a notebook, you can click “Share” on the top right corner of the Colab window and choose with whom and how (the defaults are fine) you want to share. 

