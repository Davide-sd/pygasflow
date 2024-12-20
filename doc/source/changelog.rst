Changelog
---------

future
======

* DEPRECATIONS: here is a list of newly introduced deprecations.
  See :ref:`deprecations-page` to understand the motivations.

  * The following is a list of parameter names used by
    :func:`~pygasflow.solvers.shockwave.shockwave_solver` and
    :func:`~pygasflow.solvers.shockwave.conical_shockwave_solver`
    which are now deprecated:

    * ``"m1"`` indicated the upstream Mach number. ``"mu"`` should be used
      instead.
    * ``"mn1"`` indicated the upstream normal Mach number. ``"mnu"`` should be
      used  instead.
    * ``"m2"`` indicated the downstream Mach number. ``"md"`` should be used
      instead.
    * ``"mn2"`` indicated the downstream normal Mach number. ``"mnd"`` should be
      used instead.
    * ``"m"`` indicated the upstream Mach number of a conical shock wave.
      ``"mu"`` should be used instead.

    These changes also applies to the dictionary of results, if the solver was
    executed with the keyword argument ``to_dict=True``.

  * ``beta_theta_max_for_unit_mach_downstream`` has been deprecated in favor
    of :func:`~pygasflow.shockwave.sonic_point_oblique_shock`.
  * ``beta_from_mach_max_theta`` has been deprecated in favor
    of :func:`~pygasflow.shockwave.detachment_point_oblique_shock`.
  * ``beta_theta_c_for_unit_mach_downstream`` has been deprecated in favor
    of :func:`~pygasflow.shockwave.sonic_point_conical_shock`.

* BREAKING:

  * Removed classes ``Ideal_Gas`` and ``Flow_State`` from
    ``pygasflow.utils.common`` as no longer useful to the module.
  * Refactoring of ``pygasflow.nozzles`` and ``pygasflow.solver.de_laval``
    in order to use the `holoviz param <https://param.holoviz.org/>`_ module,
    which allows for a more robust code base while also providing the
    foundation for interactive applications. In particular,
    :class:`~pygasflow.solvers.de_laval.De_Laval_Solver`,
    :class:`~pygasflow.nozzles.cd_conical.CD_Conical_Nozzle`,
    :class:`~pygasflow.nozzles.cd_top.CD_TOP_Nozzle`,
    :class:`~pygasflow.nozzles.moc.CD_Min_Length_Nozzle` are no longer
    compatible with previous versions of the module.

* Added global option ``pygasflow.defaults.solver_to_dict`` in order to
  make all solvers return a dictionary of results if it's set to True.
  Look at :ref:`solver-page` to see an example of its use.

* Added new functions :func:`~pygasflow.solvers.gas.gas_solver`,
  :func:`~pygasflow.solvers.gas.ideal_gas_solver` and
  :func:`~pygasflow.solvers.gas.sonic_condition`.

* Added a new sub-module, ``pygasflow.interactive``, which provides
  a web-based GUI (graphical user interface) to many of the functionalities
  of the module, implemented with `holoviz panel <https://panel.holoviz.org/>`_.
  The GUI allows:

  * for an easier and non-programmatic way of getting quick results.
  * to easily explore different configurations.
  * reliability: over the years there have been many web-based compressible
    flow GUIs on the internet. However, they are not guaranteed to exists
    forever. On the other hand, this sub-module is part of pygasflow, and it
    will always be readily available should the user needs it.

* Added a new sub-module, ``pygasflow.interactive.diagram``, which provides
  functionalities to quickly creates diagram related to compressible flows,
  like isentropic diagram, Fanno diagram, oblique shock diagram, shock polar
  diagram, etc.

* Added :class:`~pygasflow.shockwave.PressureDeflectionLocus` and
  :class:`~pygasflow.interactive.diagram.pressure_deflection.PressureDeflectionDiagram`
  to easily create pressure-deflection diagrams and compute related
  quantities.

* Improved reliability of
  :func:`~pygasflow.shockwave.sonic_point_oblique_shock` and
  :func:`~pygasflow.shockwave.sonic_point_conical_shock`.

* Fixed bug with :func:`~pygasflow.solvers.shockwave.shockwave_solver` and
  propagation of the specific heats ratio when the flow deflection angle and
  the shock wave angle were provided as input arguments.

* Fixed bug with :func:`~pygasflow.rayleigh.m_from_critical_temperature_ratio`
  and propagation of the specific heats ratio.

* Fixed bug with :func:`~pygasflow.shockwave.mach_from_theta_beta`,
  which computed a wrong Mach number for the special case beta=90, theta=0.

* Fixed bug with some functions that computed wrong results when
  integer numbers were provided as arguments.

* Fixed bug with :func:`~pygasflow.shockwave.shock_polar` and propagation of
  a parameter to other functions.

* Fixed functions that raised *RuntimeWarning: divide by zero encountered
  in divide*.


v1.2.1
======

* Fix import for aerothermodynamics sub-module.
* Updated doctest outputs to the format used by NumPy >= 2.0.0.


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
