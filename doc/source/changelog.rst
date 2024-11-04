Changelog
---------

v1.2.0
======

* Added ``oblique_mach_downstream`` to ``pygasflow.shockwave``.
  Thank you `Dr Chad File <https://github.com/archeryguru2000>`_ for this
  contribution.

* Added support for Numpy >= 2.0.0.
  Thank you `David Chartrand <https://github.com/DavidChartrand>`_ for this
  contribution.

* Fixed conda packaging.


v1.1.1
======

* Included build for Python 3.11.


v1.1.0
======

* Added aliases to solvers:

  * ``ise`` for ``isentropic_solver``.
  * ``fan`` for ``fanno_solver``.
  * ``ray`` for ``rayleigh_solver``.
  * ``ss`` for ``shockwave_solver``.
  * ``css`` for ``conical_shockwave_solver``.

* Added Aerothermodynamic module (``pygasflow.atd``):

  * correlations to compute boundary layer thickness, heat flux, wall
    shear stress.
  * functions to compute the pressure distribution and aerodynamic
    characteristics with the Newtonian (and modified Newtonian)
    flow theory.


v1.0.6
======

* added `to_dict` keyword argument to solvers.
* Improved doctests
* Added latex equations to ReadTheDocs documentation
* Added examples to ReadTheDocs documentation
* Added linkcode resolve to documentation


v1.0.5
======

* Updated README
* Released conda and pypi packages


v1.0.2
======

* Added Sphinx Documentation and doctests.
* Added ``plot`` method to nozzles.
* Improved Tests.
