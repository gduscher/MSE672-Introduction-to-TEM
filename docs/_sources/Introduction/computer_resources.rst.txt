Setting up computing resources

This lecture is based on jupyter notebooks. These jupyter notebooks are class notes, grahic program,
simulation and analysis tools in one.

In this lecture you have the choice to run these notebooks on your computer
or in the cloud with Google colab. In any case no programming is necessary but encouraged.
At the minimum you will have to change some input parameters.

I recommend to install anaconda and let the notebooks run on your computer.

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

If you want to collaborate with another student or with the course staff
on a notebook, you can click “Share” on the top right corner of the
Colab window and choose with whom and how (the defaults are fine)
you want to share.


Installation on your own machine

We now proceed to discuss installation of the necessary software on your own machine.

Uninstalling Anaconda

Unless you have experience with Anaconda and know how to set up
environments, if you have previously installed Anaconda with a version
of Python other than 3.8, you need to uninstall it, removing it
completely from your computer.
You can find instructions on how to do that from the official

uninstallation documentation.
Downloading and installing Anaconda
Downloading and installing Anaconda is simple.

1. Go to the Anaconda distribution homepage and download the graphical installer.
2. Be sure to download Anaconda for Python 3.8 for the appropriate operating system.
3. Follow the on-screen instructions for installation.
4. You may be prompted for optional installations, like PyCharm. You will not need these for the course.

That’s it! After you do that, you will have a functioning Python distribution.

Launching Jupyter notebook

After installing the Anaconda distribution, you should be able to launch
the Anaconda Navigator.
If you are using macOS, this is available in your Applications menu.
If you are using Windows, you can do this from the Start menu.
Launch Anaconda Navigator.

We will be using jupyter notebooks throughout the course.
You should see an option to launch Jupyter Notebook.
When you do that, a new browser window or tab will open with Jupyter
notebook running.
For the updating and installation of necessary packages, click on
Terminal to launch a terminal. You will get a terminal window
(probably black) with a bash prompt. We refer to this text interface
in the terminal as the command line.

The conda package manager

conda is a package manager for keeping all of your packages up-to-date.
It has plenty of functionality beyond our basic usage in class,
which you can learn more about by reading the docs. We will primarily
be using conda to install and update packages.

conda works from the command line. Now that you know how to get a
command line prompt, you can start using conda. The first thing we’ll
do is update conda itself. Enter the following on the command line

conda update conda

You will be prompted to continue this operation, so press y to continue.
Next, we’ll update the packages that came with the Anaconda distribution.
To do this, enter the following on the command line:

conda update --all

If anything is out of date, you will be prompted to perform the updates,
and press y to continue. (If everything is up to date, you will just see
a list of all the installed packages.) They may even be some downgrades.
This happens when there are package conflicts where one package requires
an earlier version of another. conda is very smart and figures all of
this out for you, so you can almost always say “yes” (or “y”) to conda
when it prompts you.

