.. _solver-page:

Solvers
-------

All the following solvers accepts the ``to_dict`` keyword argument:

* ``to_dict=False``: returns a list/tuple of results.
* ``to_dict=True``: returns a dictionary of results. This option has a few
  advantages:

  * it is easier to retrieve a particular quantity from a dictionary of
    results, because user only needs to remember the key name, which are
    often self explanator, for example ``"pr"`` stands for pressure ratio.
    On the other hand, with a list of results user needs to remember the index
    of a particular quantity.
  * it is easier to maintain back-compatibility. Suppose users are returning
    list of results from a solver, and accessing a particular quantity with
    some index. Time passes and the module gets implemented further, maybe new
    quantities are added to the list of results, maybe some quantity is removed.
    At this point, user's code maybe broken. This is unlikely to happen with a
    dictionary of results, where deprecation warning can easily be inserted
    when a particular key is accessed, informing users on the best course of
    action.

The default behavior is ``to_dict=False`` (for back-compatibility reasons).
If we are executing multiple solver calls and we are interested in
dictionaries of results, the quickest way is by setting a global
default option of the module, thus skipping the ``to_dict`` keyword argument:

Here is an example. As we load the module, the solver returns a
list of results:

>>> import pygasflow
>>> from pygasflow.solvers import (
...     isentropic_solver,
...     oblique_shockwave_solver,
...     fanno_solver,
... )
>>> res_ise = isentropic_solver("m", 3)
>>> type(res_ise)
<class 'list'>
>>> res_ise
[np.float64(3.0), np.float64(0.02722368370386282), np.float64(0.0762263143708159), np.float64(0.35714285714285715), np.float64(0.051532504691298484), np.float64(0.12024251094636311), np.float64(0.4285714285714286), np.float64(2.8281038574584607), np.float64(4.234567901234571), np.float64(19.47122063449069), np.float64(49.75734674434607)]

We can change the global option, after which all solvers will return
dictionaries of results:

>>> pygasflow.defaults.solver_to_dict = True
>>> res_ise = isentropic_solver("m", 3)
>>> type(res_ise)
<class 'dict'>
>>> res_ise
{'m': np.float64(3.0), 'pr': np.float64(0.02722368370386282), 'dr': np.float64(0.0762263143708159), 'tr': np.float64(0.35714285714285715), 'prs': np.float64(0.051532504691298484), 'drs': np.float64(0.12024251094636311), 'trs': np.float64(0.4285714285714286), 'urs': np.float64(2.8281038574584607), 'ars': np.float64(4.234567901234571), 'ma': np.float64(19.47122063449069), 'pm': np.float64(49.75734674434607)}
>>> res_shock = oblique_shockwave_solver("mu", 4, "theta", 15, flag="weak")
>>> type(res_shock)
<class 'pygasflow.utils.common.ShockResults'>
>>> res_shock
{'mu': np.float64(4.0), 'mnu': np.float64(1.819872100323947), 'md': np.float64(2.929007710626549), 'mnd': np.float64(0.6121186581027711), 'beta': np.float64(27.062876925385396), 'theta': np.float64(15.0), 'pr': np.float64(3.6972568717937433), 'dr': np.float64(2.3907318881276685), 'tr': np.float64(1.546495820026601), 'tpr': np.float64(0.803820352213437)}

It must be noted that we can still force a solver to return a list of results
by setting the ``to_dict = False`` keyword argument. In other words, this
keyword argument has a stronger priority than the global option:

>>> res_ise = isentropic_solver("m", 3, to_dict=False)
>>> type(res_ise)
<class 'list'>

Now, to go back to the default behavior, which returns list of results:

>>> pygasflow.defaults.solver_to_dict = False
>>> res_fanno = fanno_solver("m", 4)
>>> type(res_fanno)
<class 'list'>
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
