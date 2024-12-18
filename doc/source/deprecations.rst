.. _deprecations-page:

Deprecations
------------

This page explains the motivations behind deprecations.
They are listed according to the module version when they were
first introduced.

vFuture
=======

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

