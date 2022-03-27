Changelog
---------

v1.10.0
=======

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
