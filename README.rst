=============
coop_assembly
=============

.. start-badges

.. image:: https://img.shields.io/badge/compas-0.18.1-blue
    :target: https://github.com/compas-dev/compas
    :alt: compas version

.. image:: https://img.shields.io/badge/compas-0.14.0-pink
    :target: https://github.com/compas-dev/compas_fab
    :alt: compas_fab version

-----

.. image:: https://img.shields.io/badge/License-MIT-blue
    :target: https://github.com/stefanaparascho/coop_assembly/blob/dev/LICENSE
    :alt: License MIT

.. image:: https://travis-ci.com/yijiangh/coop_assembly.svg?branch=dev
    :target: https://travis-ci.com/yijiangh/coop_assembly

.. image:: https://readthedocs.org/projects/coop-assembly/badge/?version=latest
    :target: https://coop-assembly.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/yijiangh/coop_assembly/badge.svg?branch=dev
    :target: https://coveralls.io/github/yijiangh/coop_assembly?branch=dev

.. end-badges

.. Write project description

**coop_assembly**: Geometry generation of robotically assembled spatial structures.

**News:**

    9-21-2019: This package contains materials for the `FABRICATION-INFORMED DESIGN OF
    ROBOTICALLY ASSEMBLED STRUCTURES <https://design-modelling-symposium.de/workshops/fabrication-informed-design-of-robotically-assembled-structures/>`_
    for the Design Modeling Symposium Berlin 2019.

Getting Started
----------------

See `Getting Started <./docs/getting_started.rst>`_ for installation instructions.

Planning examples
-----------------

Regression search examples
``````````````````````````

First, make sure that you first follow the ``Installating sequence and motion planning stack`` section
of the `Getting Started <./docs/getting_started.rst>`_ document.

Then, Run the regression search algorithm for find a construction plan for spatial bars.

::

    python -m coop_assembly.planning.run -v -w

Issue ``python -m coop_assembly.planning.run -h`` to print out the helps.

And you should be able to see the following (press Enter when asked in the commandline) animation (click image for the youtube video):

.. image:: http://img.youtube.com/vi/KGrHz5gNqqc/0.jpg
    :target: http://www.youtube.com/watch?feature=player_embedded&v=KGrHz5gNqqc
    :alt: bar assembly simulated demo

PDDLStream examples (WIP)
`````````````````````````

First, follow the ``Installating PDDLStream (WIP)`` section
of the `Getting Started <./docs/getting_started.rst>`_ document to set up `PDDLStream` and `pyplanners`.

2D version
::::::::::

This is a 2D additive construction domain, developed for testing our algorithms.

::

    python -m examples.assembly2D.run -v -w -p 2D_tower_skeleton.json -a incremental_sa

The `incremental_sa` is the `incremental algorithm <https://arxiv.org/pdf/1802.08705.pdf>`_ in PDDLStream, using
the `pyplanners <https://github.com/caelan/pyplanners>`_ for supporting
`semantic attachments <http://www2.informatik.uni-freiburg.de/~ki/papers/dornhege-etal-icaps09.pdf>`_.

Issue ``python -m examples.assembly2D.run -h`` to print out the helps.

If run correctly, you should see an animation (click image for the youtube video) like the following
(subject to changes) after a plan is found:

.. image:: http://img.youtube.com/vi/xAPpfH2SzDo/0.jpg
    :target: http://www.youtube.com/watch?feature=player_embedded&v=xAPpfH2SzDo
    :alt: bar assembly simulated demo

3D version
::::::::::

WIP.

Troubleshooting
---------------

Sometimes things don't go as expected. Checkout the `troubleshooting <./docs/troubleshooting.rst>`_ documentation for answers to the most common issues you might bump into.

Credits
-------

This package was initiated by Stefana Parascho <parascho@princeton.edu> `@stefanaparascho <https://github.com/stefanaparascho>`_
at the CREATE lab, Princeton University.
The sequence and motion planning scripts are developed by Yijiang Huang <yijiangh@mit.edu> `@yijiangh <https://github.com/yijiangh>`_
 with `collaborators <./AUTHORS.rst>`_.
