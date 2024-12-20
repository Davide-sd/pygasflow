.. _solver-page:

Solvers
-------

All the following solvers accepts the ``to_dict`` keyword argument:

* ``to_dict=False``: returns a list/tuple of results (default behavior).
* ``to_dict=True``: returns a dictionary of results.

If we are executing multiple solver calls and we are interested in
dictionaries of results, the quickest way is by setting a global
default option of the module, thus skipping the ``to_dict`` keyword argument:

>>> import pygasflow
>>> from pygasflow.solvers import isentropic_solver
>>> res = isentropic_solver("m", 3)
[np.float64(3.0), np.float64(0.02722368370386282), np.float64(0.0762263143708159), np.float64(0.35714285714285715), np.float64(0.051532504691298484), np.float64(0.12024251094636311), np.float64(0.4285714285714286), np.float64(2.8281038574584607), np.float64(4.234567901234571), np.float64(19.47122063449069), np.float64(49.75734674434607)]
>>> type(res)
list
>>> # set the global default option
>>> pygasflow.defaults.solver_to_dict = True
>>> isentropic_solver("m", 3)
{'m': np.float64(3.0), 'pr': np.float64(0.02722368370386282), 'dr': np.float64(0.0762263143708159), 'tr': np.float64(0.35714285714285715), 'prs': np.float64(0.051532504691298484), 'drs': np.float64(0.12024251094636311), 'trs': np.float64(0.4285714285714286), 'urs': np.float64(2.8281038574584607), 'ars': np.float64(4.234567901234571), 'ma': np.float64(19.47122063449069), 'pm': np.float64(49.75734674434607)}
>>> type(res)
dict


Isentropic Solver
=================

.. module:: pygasflow.solvers.isentropic

.. autofunction:: isentropic_solver


Fanno Solver
============

.. module:: pygasflow.solvers.fanno

.. autofunction:: fanno_solver


Rayleigh Solver
===============

.. module:: pygasflow.solvers.rayleigh

.. autofunction:: rayleigh_solver


Shockwave Solvers
=================

.. module:: pygasflow.solvers.shockwave

.. autofunction:: normal_shockwave_solver

.. autofunction:: oblique_shockwave_solver

.. autofunction:: conical_shockwave_solver


De Laval Solver
===============

.. module:: pygasflow.solvers.de_laval

.. autofunction:: find_shockwave_area_ratio

.. autoclass:: De_Laval_Solver
   :members:


Gas Related Solvers
===================

.. module:: pygasflow.solvers.gas
.. autofunction:: gas_solver
.. autofunction:: ideal_gas_solver
.. autofunction:: sonic_condition
