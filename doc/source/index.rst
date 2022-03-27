.. pygasflow documentation master file, created by
   sphinx-quickstart on Mon Jan 31 13:59:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pygasflow's documentation!
=====================================

pygasflow provides a few handful functions to quickly perform:

* Quasi-1D ideal gasdynamic (perfect gas) computations with Python. The
  following solvers are implemented:

  * Isentropic flow: ``isentropic_solver`` (or ``ise``).
  * Fanno flow: ``fanno_solver`` (or ``fan``).
  * Rayleigh flow: ``rayleigh_solver`` (or ``ray``).
  * Normal/Oblique shock waves: ``shockwave_solver`` (or ``ss``).
  * Conical shock waves: ``conical_shockwave_solver`` (or ``css``).

  If a solver doesn't suit your needs, try and search into the submodules
  for a suitable function.

* Aerothermodynamic Computations with Python (``pygasflow.atd`` module):

  * Correlations to estimate boundary layer thickness, heat flux and wall
    shear stress over a flat plate or a stagnation region.
  * Newtonian Flow Theory to estimate the pressure distribution around
    objects and their aerodynamic characteristics. 

The following charts has been generated with the functions included in this package:

.. image:: _static/isentropic.png
   :width: 200
   :alt: Isentropic Flow

.. image:: _static/fanno.png
   :width: 200
   :alt: Fanno Flow

.. image:: _static/rayleigh.png
   :width: 200
   :alt: Rayleigh Flow

.. image:: _static/conical-flow.png
   :width: 200
   :alt: Conical Flow

.. image:: _static/mach-beta-theta.png
   :width: 200
   :alt: Shockwave relations

.. image:: _static/shock-reflection.png
   :width: 200
   :alt: Shock Reflection



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install.rst
   examples/index.rst
   modules/index.rst
   changelog.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
