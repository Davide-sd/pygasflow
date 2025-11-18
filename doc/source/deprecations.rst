.. _deprecations-page:

Deprecations
------------

This page explains the motivations behind deprecations introduced by a
particular version of the module.

v1.4.0
======

* Deprecated ``pygasflow.generic.sound_speed`` in favor of
  :func:`~pygasflow.common.sound_speed` in order to remove duplicates.


v1.3.0
=======

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

  Similarly, if users request a dictionary of results (by setting
  ``to_dict=True``), the aforementioned replacements are used as keys.

  The motivation behind this design choice is to improve code clarity and
  quality. Suppose for example you are working on a flow through
  multiple shock system, and you assign a name to the states upstream
  and downstream of each shock. For example, 1 and 2 represents the
  upstream and downstream conditions of the first shock wave, A, respectively.
  2 and 3 represents the upstream and downstream conditions of the second
  shock wave, B, and so on. The following situation was very common with the
  old naming convention:

  .. code-block::

    M1 = 3
    shockA = shockwave_solver("m1", M1, "theta", 15)
    M2 = shockA["m2"]
    shockB = shockwave_solver("m1", M2, "theta", 20)
    M3 = shockB["m2"]

  The last two commands are confusing. In particular:

  * ``shockwave_solver("m1", M2...``: here we were telling the solver that
    the upstream Mach number to shockB is M2.
  * ``M3 = shockB["m2"]``: here, with ``"m2"`` we are retrieving the downstream Mach number
    of shockB and assigning it to M3.

  This quickly become even more confusing as we add more shockwaves,
  both for people writing the code, as well as to people reading it.

  With the new naming convention, the above example becomes:

  .. code-block::

    M1 = 3
    shockA = shockwave_solver("mu", M1, "theta", 15)
    M2 = shockA["md"]
    shockB = shockwave_solver("mu", M2, "theta", 20)
    M3 = shockB["md"]

  Here, it is clear that we are specifying M2 to be the upstream Mach number
  to shockB and that M3 is the downstream Mach number of shockB.

* ``beta_theta_max_for_unit_mach_downstream`` has been deprecated because
  it generated confusion: it returned the the shock wave angle, beta, where
  the downstream Mach number was sonic, and the maximum deflection angle,
  theta, associated to the user-provided upstream Mach number.

  Instead, :func:`~pygasflow.shockwave.sonic_point_oblique_shock` should
  be used, which returns the shock wave angle, beta, and the deflection angle,
  theta, where  the downstream Mach number is sonic.

* ``beta_from_mach_max_theta`` has been deprecated because, while it also
  computed the maximum deflection angle, theta_max, it only returned
  the former. However, it is often useful to know both results.

  Hence, :func:`~pygasflow.shockwave.detachment_point_oblique_shock`
  should be used, which returns the shock wave angle, beta, and the maximum
  deflection angle, theta_max, associated to the detachment point.

* ``beta_theta_c_for_unit_mach_downstream`` has been deprecated in order to
  have a consistent function name. Hence,
  :func:`~pygasflow.shockwave.sonic_point_conical_shock` should be used,
  which returns the shock wave angle, beta, and the half-cone angle, theta_c,
  where the downstream Mach number is sonic.

