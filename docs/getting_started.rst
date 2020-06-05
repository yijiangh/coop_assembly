.. _getting_started:

********************************************************************************
Getting Started
********************************************************************************

.. Write installation instructions here

**Prerequisites**

0. Operating System:
    **Windows 10** and **Mac(!)** both works!
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
installing all the required python packages.

Type in the following commands in your Anaconda terminal to create a conda environment
(search for ``Anaconda Prompt`` in the Windows search bar):

::

    git clone --recursive https://github.com/yijiangh/coop_assembly.git
    cd coop_assembly
    conda create -n cp_ws python=3.7
    conda activate cp_ws

Notice that we have cloned this repository using the `--recursive` flag
(if not then type `git submodule update --init --recursive`) to make sure we have
all the git submodules in place.

The last two lines above will create a conda environment called `cp_ws`.
Now, we install all the depedency packages:

::

    pip install ./external/compas_fab
    pip install ./external/pybullet_planning
    pip install -e .

This will install the two main depedencies `compas_fab` and `pybullet_planning`. Notice
that we are using a customized version of `compas_fab` here, which might be in conflict
with the `compas_fab` version used in your other projects. So the conda environment helps
you isolate them here ✨ The last line install `coop_assembly` in a `debug` mode,
which means your changes will be directly reflected in your execution (from python
or from GHPython invokes).

If you see an error message like ``Error: Microsoft Visual C++ 14.0 is required``,
please see `troubleshooting <./docs/troubleshooting.rst>`_ for instructions to install
the Microsoft Visual Studio Build tools.

Great - we are almost there! Now type `python` in your Anaconda Prompt, and test if the installation went well:

::

    >>> import compas
    >>> import compas_fab
    >>> import pybullet
    >>> import coop_assembly

If that doesn't fail, you're good to go! Exit the python interpreter (either typing `exit()` or pressing `CTRL+Z` followed by `Enter`).

Now let's make all the installed packages available inside Rhino. Still from the Anaconda Prompt, type the following:

In order to make ``coop_assembly`` accessible in Rhino/Grasshopper,
we need the run the following commands in the Anaconda prompt first
and then **restart Rhino**:

::

    python -m compas_rhino.install
    python -m compas_rhino.install -p coop_assembly

And you should be able to see outputs like:

::

   Installing COMPAS packages to Rhino 6.0 IronPython lib:
   IronPython location: C:\Users\<User Name>\AppData\Roaming\McNeel\Rhinoceros\6.0\Plug-ins\IronPython (814d908a-e25c-493d-97e9-ee3861957f49)\settings\lib

   compas               OK
   compas_rhino         OK
   compas_ghpython      OK
   coop_assembly        OK

   Completed.

Congrats! 🎉 You are all set!

Grasshopper examples can be found in the `examples` folder. For Stefana and her students,
please see `examples/shape_gen_GH <../examples/shape_gen_GH>`_ for the latest examples on design generation.
