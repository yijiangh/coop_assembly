Troubleshooting
================

Sometimes things don't go as expected. Here are some of answers to the most common issues you might bump into:

------------

..

    Q: Error: Microsoft Visual C++ 14.0 is required

.. _vc14_instruction:

1. Follow the `link <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_
   to download the visual studio build tools.
2. Click the ``vs_buildtools__xxx.exe`` file you just downloaded.
3. Follow the instruction of the Visual Studio Installer, until it
   finishes its downloading and installation.
4. Select ``C++ Build Tools`` and click ``Install``.

.. image:: images/visual_studio_installer_snapshot.png
   :scale: 50 %
   :alt: visual studio installer snapshot
   :align: center

------------

..

    Q: I've installed ``git`` but ``git`` commands don't work in the commandline prompt.

See `this post <https://stackoverflow.com/a/53706956>`_ for instructions on
how to add ``git`` to the environment PATH in Windows.

------------

..

    Q: `conda` commands don't work.

Try running them from the *Conda Prompt*. Depending on how you installed Anaconda, it might not be available by default on the normal Windows command prompt.

------------

..

    Q: When trying to install the framework in Rhino, it fails indicating the lib folder of IronPython does not exist.

Make sure you have opened Rhino 6 and Grasshopper at least once, so that it finishes setting up all its internal folder structure.

------------

..

    Q: In Xfunc call, error message "Cannot find DLL specified. (_Arpack ...)"

This happens because some previous calls blocked the ``scipy`` complied libraries.
For a temporal fix, in your conda environment, uninstall ``pip install scipy`` and
then ``pip install scipy=1.3.1`` works.

Updating packages
-----------------

Updating individual packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Updating only some of the dependencies? Here are some convenient links
(remember to activate your conda environment by ``conda activate <env name>``
before you do these!):

Update ``coop_assembly``:

::

    pip install --upgrade git+https://github.com/yijiangh/coop_assembly.git@dev#egg=coop_assembly
    python -m compas_rhino.install -p coop_assembly
