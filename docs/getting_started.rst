.. _getting_started:

********************************************************************************
Getting Started
********************************************************************************

.. Write installation instructions here

**Prerequisites**

0. Operating System:
    **Windows 10** and **Mac(!)**
1. `Rhinoceros 3D 6.0 <https://www.rhino3d.com/>`_
    We will use Rhino / Grasshopper as a frontend for inputting
    geometric and numeric paramters, and use various python packages as the
    computational backends. The tight integration between Grasshopper and python
    environments is made possible by `COMPAS <https://compas-dev.github.io/>`_
    and `COMPAS_FAB <https://gramaziokohler.github.io/compas_fab/latest/>`_.
2. `Git <https://git-scm.com/>`_
    We need ``Git`` for fetching required packages from github.
3. `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
    We will install all the required python packages using
    `Miniconda` (a light version of Anaconda). Miniconda uses
    **environments** to create isolated spaces for projects'
    depedencies.
4. `Microsoft Visual Studio Build tools <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_
    Our simulation environment ``pybullet`` has a C++ backend, which needs
    ``Microsoft Visual C++ 14.0`` to compile and build the python bindings. Note that this is needed only for Windows OS.

**Working in a conda environment**

It is recommended to set up a conda environment to create a clean, isolated space for
installing all the required python packages. We've provided a conda environment file
to make this process easy - please do the following steps:

Type in the following commands in your Anaconda terminal
(search for ``Anaconda Prompt`` in the Windows search bar):

::

    git clone https://github.com/createchaos/coop_assembly.git
    cd coop_assembly
    conda env create -f coop_assembly_ws.yml

Wait for the building process to finish, the command above will
fetch and build all the required packages, which will take some time
(5~10 mins).

If you see an error message like ``Error: Microsoft Visual C++ 14.0 is required``,
please see `troubleshooting <./docs/troubleshooting.rst>`_ for instructions to install the Microsoft Visual Studio Build tools.

Then, activate the newly created conda environment (with all the needed packages installed):

::

    conda activate coop_assembly_ws

Great - we are almost there! Now type `python` in your Anaconda Prompt, and test if the installation went well:

::

    >>> import compas
    >>> import compas_fab
    >>> import pybullet
    >>> import coop_assembly
    >>> import ikfast_ur5

If that doesn't fail, you're good to go! Exit the python interpreter (either typing `exit()` or pressing `CTRL+Z` followed by `Enter`).

Now let's make all the installed packages available inside Rhino. Still from the Anaconda Prompt, type the following:

In order to make ``coop_assembly`` accessible in Rhino/Grasshopper,
we need the run the following commands in the Anaconda prompt first
and then **restart Rhino**:

::

    python -m compas_rhino.install
    python -m compas_rhino.install -p coop_assembly ur_online_control compas_fab roslibpy

And you should be able to see outputs like:

::

   Installing COMPAS packages to Rhino 6.0 IronPython lib:
   IronPython location: C:\Users\<User Name>\AppData\Roaming\McNeel\Rhinoceros\6.0\Plug-ins\IronPython (814d908a-e25c-493d-97e9-ee3861957f49)\settings\lib

   compas               OK
   compas_rhino         OK
   compas_ghpython      OK
   compas_bootstrapper  OK
   coop_assembly        OK

   Completed.

Congrats! 🎉 You are all set!

Grasshopper examples can be found in ``examples\workshop_dms``.
