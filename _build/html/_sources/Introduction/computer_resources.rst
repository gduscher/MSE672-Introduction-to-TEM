Setting up computing resources
==============================


This lecture is based on jupyter notebooks. These jupyter notebooks are class notes, grahic program,
simulation and analysis tools in one.

In this lecture you have the choice to run these notebooks on your computer
or in the cloud with Google colab. In any case no programming is necessary but encouraged.
At the minimum you will have to change some input parameters.

I expect you to install anaconda and let the notebooks run on your computer.

However, we will need a linux environment for dynamic simulations and we will use Google colab then.  

Google Colab
-------------

In order to use Google Colab, you must have a Google account.
UTK students and employees have an account and can log in with their vol id.
Many of you may have a personal Google account, usually set up for things like GMail, YouTube, etc.
For your work in this class, it advantageous to use your UTK account.
This will facilitate collaboration with your
teammates in the course, as well as with course staff.

Google Colab are most tested for Chrome, Firefox, and Safari

You can launch a Colab notebook by
simply navigating to https://colab.research.google.com/.
Alternatively, you can click a "notebook link” on this webpage
and you will launch that notebook in Colab.

Watchouts when using Colab
--------------------------

If you do run a notebook in Colab, you are doing your computing on one
of Google’s computers via a virtual machine. You get two CPU cores and
12 GB of RAM. You can also get GPUs and TPUs (Google’s tensor processing
units).
The computing resources should be enough for all of our calculations
in this course.
However, there are some limitations you should be aware of.

    The interactivity of graphs is very limited. For example, no selection
    of datacan be done interactively. With the ``bokeh`` plotting packages
    you can at least zoom in and out of images and line plots.

    If your notebook is idle for too long, you will get disconnected
    from your notebook. “Idle” means that cells are not being edited
    or executed. The idle timeout varies depending on the load on
    Google’s computers.

    Your virtual machine will disconnect if it is being used for too long.
    It typically will only available for 12 hours before disconnecting,
    though times can vary, again based on load.

These limitations are in place so that Google can offer Colab for free.
If you want more cores, longer timeouts, etc., you might want to check
out Colab Pro. However, the free tier should work well for you in the
course. You of course can always run on your own machine, and in fact
are encouraged to do so except where collaboration is necessary.

There are additional software-specific watchouts when using Colab.

    Colab does not allow for full functionality Bokeh apps and some
    Panel functionality will therefore not be used in this course.

    Colab instances have specific software installed, so you will need
    to install anything else you need in your notebook.
    This is not a major burden, and is discussed in the next section.

I recommend reading the Colab FAQs for more information about Colab.
Software in Colab

When you launch a Google Colab notebook, much of the software we will
use in class is already installed. It is not always the latest version
of the software, however. In fact, as of early January 2021, Colab
is running Python 3.6, whereas you will run Python 3.8 on your machine
through your Anaconda installation. Nonetheless, all of
the analyses we do for this class will work just fine in Colab.

Colab setup
-----------

Because the notebooks in Colab have software preinstalled,
and no more, you will often need to install software before you can
run the rest of the code in a notebook. To enable this, when necessary,
in the first code cell of each notebook in this class, we will have the
following code (or a variant thereof depending on what is needed or if
the default installations of Colab change). Running this code will not
affect running your notebook on your local machine;
the same notebook will work on your local machine or on Colab.

# Colab setup ------------------
%%capture
import os, sys
if "google.colab" in sys.modules:
!pip install pyTEMlib
# ------------------------------

Google Drive Setup
------------------
In addition to installing the necessary software on a Colab instance,
we also have to have access to the data we want to analyse. We expect these
data to be on your google drive. You will have to follow the directions to
give permission to the notebook to access these data.

# Google Drive setup -----------
if "google.colab" in sys.modules:
from google.colab import drive
drive.mount("/content/drive")
# ------------------------------

When running on your local machine, the path to the data can be chosen without restrictions.

Collaborating with Colab
-------------------------
If you want to collaborate with another student or with the course staff
on a notebook, you can click “Share” on the top right corner of the
Colab window and choose with whom and how (the defaults are fine)
you want to share.


