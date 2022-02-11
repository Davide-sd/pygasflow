.. pygasflow documentation master file, created by
   sphinx-quickstart on Mon Jan 31 13:59:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pygasflow's documentation!
=====================================

pygasflow provides a few handful functions to quickly perform quasi-1D ideal
gasdynamic (perfect gas) computations with Python.

The following flow relations are implemented:

* Isentropic flow
* Fanno flow
* Rayleigh flow
* Shock wave relations (normal shock, oblique shock, conical shock)

Depending on the problem at hand, you can:

* use the appropriate :doc:`solver </modules/solvers/index>` which, given a
  parameter (mach number or ratio), computes all other ratios and mach
  numbers. This is the preferred option.
* search the appropriate function in the following submodules:
  ``pygasflow.isentropic``, ``pygasflow.fanno``,
  ``pygasflow.rayleigh``, ``pygasflow.shockwave``.

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
   modules/index.rst
   changelog.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
