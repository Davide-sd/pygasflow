
Nozzles
-------

This module contains classes to build/visualize nozzle geometries, which can
then be used with ``De_Laval_Solver``.


Converget-Divergent Conical
===========================

.. module:: pygasflow.nozzles.cd_conical

.. autoclass:: CD_Conical_Nozzle

.. autofunction:: pygasflow.nozzles.cd_conical.CD_Conical_Nozzle.__init__

.. autofunction:: pygasflow.nozzles.cd_conical.CD_Conical_Nozzle.build_geometry


Converget-Divergent TOP
=======================

.. module:: pygasflow.nozzles.cd_top

.. autoclass:: CD_TOP_Nozzle

.. autofunction:: pygasflow.nozzles.cd_top.CD_TOP_Nozzle.__init__

.. autofunction:: pygasflow.nozzles.cd_top.CD_TOP_Nozzle.build_geometry


Minimum Length Nozzle with MoC
==============================

.. module:: pygasflow.nozzles.moc

.. autofunction:: min_length_supersonic_nozzle_moc

.. autoclass:: CD_Min_Length_Nozzle

.. autofunction:: pygasflow.nozzles.moc.CD_Min_Length_Nozzle.__init__

.. autofunction:: pygasflow.nozzles.moc.CD_Min_Length_Nozzle.build_geometry


Rao Parabola Angles
===================

.. module:: pygasflow.nozzles.rao_parabola_angles

.. autoclass:: Rao_Parabola_Angles

.. autofunction:: pygasflow.nozzles.rao_parabola_angles.Rao_Parabola_Angles.angles_from_Lf_Ar

.. autofunction:: pygasflow.nozzles.rao_parabola_angles.Rao_Parabola_Angles.area_ratio_from_Lf_angle

.. autofunction:: pygasflow.nozzles.rao_parabola_angles.Rao_Parabola_Angles.plot
