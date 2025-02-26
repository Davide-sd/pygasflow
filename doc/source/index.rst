.. pygasflow documentation master file, created by
   sphinx-quickstart on Mon Jan 31 13:59:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pygasflow's documentation!
=====================================

pygasflow is a Python package that provides a few handful functions to quickly perform:

* Compressible flow computation in the quasi-1D ideal gasdynamic (perfect gas) regime. The
  following solvers are implemented:

  * Isentropic flow: :func:`~pygasflow.solvers.isentropic.isentropic_solver`.
  * Fanno flow: :func:`~pygasflow.solvers.fanno.fanno_solver`.
  * Rayleigh flow: :func:`~pygasflow.solvers.rayleigh.rayleigh_solver`.
  * Normal shock waves: :func:`~pygasflow.solvers.shockwave.normal_shockwave_solver`.
  * Oblique shock waves: :func:`~pygasflow.solvers.shockwave.shockwave_solver`.
  * Conical shock waves: :func:`~pygasflow.solvers.shockwave.conical_shockwave_solver`.
  * Pressure-deflection diagrams: :class:`~pygasflow.shockwave.PressureDeflectionLocus` and
    :class:`~pygasflow.interactive.diagrams.PressureDeflectionDiagram`.

  If a solver doesn't suit your needs, try and search into the submodules
  for a suitable function.

* Aerothermodynamic computations (``pygasflow.atd`` module):

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

.. image:: _static/conical-shock.png
   :width: 200
   :alt: Conical Flow

.. image:: _static/oblique-shock.png
   :width: 200
   :alt: Shockwave relations

.. image:: _static/shock-reflection.png
   :width: 200
   :alt: Shock Reflection

The following screenshots highlights the interactive capabilities implemented
in this module:

.. image:: _static/interactive-rayleigh.png
   :width: 200
   :alt: Shock Reflection

.. image:: _static/interactive-oblique-shock.png
   :width: 200
   :alt: Shock Reflection

.. image:: _static/interactive-nozzles.png
   :width: 200
   :alt: Shock Reflection


Development and Support
=======================

If you feel like a feature could be implemented, open an
`issue or create a PR <https://github.com/Davide-sd/pygasflow/issues>`_.

If you really want a new feature but you don't have the capabilities or the
time to make it work, I'm willing to help; but first, open an issue or send
me an email so that we can discuss a sponsorship strategy.

Developing this module and its documentation was no easy job. Implementing
new features and fixing bugs requires time and energy too. If you found this
module useful and would like to show your appreciation, please consider
sponsoring this project with either one of these options:

.. button-link:: https://www.buymeacoffee.com/davide_sd
    :color: primary

    :fas:`mug-hot;fa-xl` Buy me a Coffee

.. button-link:: https://github.com/sponsors/Davide-sd
    :color: primary

    :fab:`github;fa-xl` Github Sponsor :fas:`heart;fa-xl`


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install.rst
   examples/index.rst
   modules/index.rst
   deprecations.rst
   changelog.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
