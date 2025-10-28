.. _solver-page:

Solvers
-------

All the following solvers accepts the ``to_dict`` keyword argument:

* ``to_dict=False``: returns a list of results.
* ``to_dict=True``: returns a dictionary of results. This option has a few
  advantages:

  * it is easier to retrieve a particular quantity from a dictionary of
    results, because user only needs to remember the key name, which are
    often self explanatory, for example ``"pr"`` stands for pressure ratio.
    On the other hand, with a list of results user needs to remember the index
    of a particular quantity.
  * it is easier to maintain back-compatibility. Suppose users are returning
    list of results from a solver, and accessing a particular quantity with
    some index. Time passes and the module gets implemented further, maybe new
    quantities are added to the list of results, maybe some quantity is removed.
    At this point, user's code may be broken. This is unlikely to happen with a
    dictionary of results, where deprecation warning can easily be inserted
    when a particular key is accessed, informing users on the best course of
    action.

The default behavior is ``to_dict=False`` (for back-compatibility reasons).
If we are executing multiple solver calls and we are interested in
dictionaries of results, the quickest way is by setting a global
default option of the module, thus skipping the ``to_dict`` keyword argument:

Regardless of the ``to_dict`` option, results returned from the solvers are
enhanced lists/dicts, which exposes the ``show()`` method in order to
print a nice tables with numerical results and explanatory labels.

Here is an example. As we load the module, the solver returns a
list of results:

>>> import pygasflow
>>> from pygasflow.solvers import (
...     isentropic_solver,
...     oblique_shockwave_solver,
...     fanno_solver,
... )
>>> res_ise = isentropic_solver("m", 3)
>>> isinstance(res_ise, list)
True
>>> res_ise
[np.float64(3.0), np.float64(0.02722368370386282), np.float64(0.0762263143708159), np.float64(0.35714285714285715), np.float64(0.051532504691298484), np.float64(0.12024251094636311), np.float64(0.4285714285714286), np.float64(1.9639610121239315), np.float64(4.234567901234571), np.float64(19.47122063449069), np.float64(49.75734674434607)]
>>> res_ise.show()    # doctest: +NORMALIZE_WHITESPACE
idx   quantity
--------------------------
0     M                        3.00000000
1     P / P0                   0.02722368
2     rho / rho0               0.07622631
3     T / T0                   0.35714286
4     P / P*                   0.05153250
5     rho / rho*               0.12024251
6     T / T*                   0.42857143
7     U / U*                   1.96396101
8     A / A*                   4.23456790
9     Mach Angle              19.47122063
10    Prandtl-Meyer           49.75734674

We can change the global option, after which all solvers will return
dictionaries of results:

>>> pygasflow.defaults.solver_to_dict = True
>>> res_ise = isentropic_solver("m", 3)
>>> isinstance(res_ise, dict)
True
>>> res_ise
{'m': np.float64(3.0), 'pr': np.float64(0.02722368370386282), 'dr': np.float64(0.0762263143708159), 'tr': np.float64(0.35714285714285715), 'prs': np.float64(0.051532504691298484), 'drs': np.float64(0.12024251094636311), 'trs': np.float64(0.4285714285714286), 'urs': np.float64(1.9639610121239315), 'ars': np.float64(4.234567901234571), 'ma': np.float64(19.47122063449069), 'pm': np.float64(49.75734674434607)}
>>> res_ise.show()    # doctest: +NORMALIZE_WHITESPACE
key     quantity
----------------------------
m       M                        3.00000000
pr      P / P0                   0.02722368
dr      rho / rho0               0.07622631
tr      T / T0                   0.35714286
prs     P / P*                   0.05153250
drs     rho / rho*               0.12024251
trs     T / T*                   0.42857143
urs     U / U*                   1.96396101
ars     A / A*                   4.23456790
ma      Mach Angle              19.47122063
pm      Prandtl-Meyer           49.75734674
>>> res_shock = oblique_shockwave_solver("mu", 4, "theta", 15, flag="weak")
>>> isinstance(res_shock, dict)
True
>>> res_shock
{'mu': np.float64(4.0), 'mnu': np.float64(1.819872100323947), 'md': np.float64(2.929007710626549), 'mnd': np.float64(0.6121186581027711), 'beta': np.float64(27.062876925385396), 'theta': np.float64(15.0), 'pr': np.float64(3.6972568717937433), 'dr': np.float64(2.3907318881276685), 'tr': np.float64(1.546495820026601), 'tpr': np.float64(0.803820352213437)}
>>> res_shock.show()    # doctest: +NORMALIZE_WHITESPACE
key     quantity
---------------------
mu      Mu                4.00000000
mnu     Mnu               1.81987210
md      Md                2.92900771
mnd     Mnd               0.61211866
beta    beta             27.06287693
theta   theta            15.00000000
pr      pd/pu             3.69725687
dr      rhod/rhou         2.39073189
tr      Td/Tu             1.54649582
tpr     p0d/p0u           0.80382035

It must be noted that we can still force a solver to return a list of results
by setting the ``to_dict = False`` keyword argument. In other words, this
keyword argument has a stronger priority than the global option:

>>> res_ise = isentropic_solver("m", 3, to_dict=False)
>>> isinstance(res_ise, list)
True

Now, to go back to the default behavior, which returns list of results:

>>> pygasflow.defaults.solver_to_dict = False
>>> res_fanno = fanno_solver("m", 4)
>>> isinstance(res_fanno, list)
True
>>> res_fanno
[np.float64(4.0), np.float64(0.1336306209562122), np.float64(0.46770717334674267), np.float64(0.28571428571428575), np.float64(10.718750000000002), np.float64(2.138089935299395), np.float64(0.6330649317809258), np.float64(2.3719945443662134)]


Isentropic Solver
=================

.. module:: pygasflow.solvers.isentropic

.. autofunction:: isentropic_solver
.. autofunction:: print_isentropic_results
.. autofunction:: isentropic_compression


Fanno Solver
============

.. module:: pygasflow.solvers.fanno

.. autofunction:: fanno_solver
.. autofunction:: print_fanno_results


Rayleigh Solver
===============

.. module:: pygasflow.solvers.rayleigh

.. autofunction:: rayleigh_solver
.. autofunction:: print_rayleigh_results


Shockwave Solvers
=================

.. module:: pygasflow.solvers.shockwave

.. autofunction:: normal_shockwave_solver
.. autofunction:: print_normal_shockwave_results

.. autofunction:: oblique_shockwave_solver
.. autofunction:: print_oblique_shockwave_results

.. autofunction:: conical_shockwave_solver
.. autofunction:: print_conical_shockwave_results

.. autofunction:: shock_compression


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
.. autofunction:: print_gas_results
.. autofunction:: ideal_gas_solver
.. autofunction:: print_ideal_gas_results
.. autofunction:: sonic_condition
.. autofunction:: print_sonic_condition_results
