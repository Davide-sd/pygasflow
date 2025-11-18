Install
-------

``pygasflow`` can be installed with pip or conda::

    pip install pygasflow

Or::

    conda install conda-forge::pygasflow


Basic Usage
===========

If we are interested in working with pint units:

.. code-block:: python

    import pint
    import pygasflow
    from pygasflow import isentropic_solver

    # create a UnitRegistry
    ureg = pint.UnitRegistry()
    # use short units notation
    ureg.formatter.default_format = "~"
    # let pygasflow knows which UnitRegistry to use
    pygasflow.defaults.pint_ureg = ureg

    T1 = 290 * ureg.K
    p1 = 1 * ureg.atm

    # compute ratios and quantities for an isentropic flow at Mach 2
    res = isentropic_solver("m", 2, to_dict=True)
    res.show()
    # key     quantity
    # ----------------------------
    # m       M                        2.00000000
    # pr      P / P0                   0.12780453
    # dr      rho / rho0               0.23004815
    # tr      T / T0                   0.55555556
    # prs     P / P*                   0.24192491
    # drs     rho / rho*               0.36288737
    # trs     T / T*                   0.66666667
    # urs     U / U*                   1.63299316
    # ars     A / A*                   1.68750000
    # ma      Mach Angle              30.00000000
    # pm      Prandtl-Meyer           26.37976081

    # compute total quantities
    T0 = (1 / res["tr"]) * T1
    P0 = (1 / res["pr"]) * p1
    print("---------")
    print(f"P0 = {P0}")
    print(f"T0 = {T0}")
    # ---------
    # P0 = 7.824449066867263 atm
    # T0 = 522.0 K

If we are not interested about working with units:

.. code-block:: python

    from pygasflow import isentropic_solver

    T1 = 290  # K
    p1 = 1    # atm

    # compute ratios and quantities for an isentropic flow at Mach 2
    res = isentropic_solver("m", 2, to_dict=True)
    res.show()
    # key     quantity
    # ----------------------------
    # m       M                        2.00000000
    # pr      P / P0                   0.12780453
    # dr      rho / rho0               0.23004815
    # tr      T / T0                   0.55555556
    # prs     P / P*                   0.24192491
    # drs     rho / rho*               0.36288737
    # trs     T / T*                   0.66666667
    # urs     U / U*                   1.63299316
    # ars     A / A*                   1.68750000
    # ma      Mach Angle              30.00000000
    # pm      Prandtl-Meyer           26.37976081

    # compute total quantities
    T0 = (1 / res["tr"]) * T1
    P0 = (1 / res["pr"]) * p1
    print("---------")
    print(f"P0 = {P0} atm")
    print(f"T0 = {T0} K")
    # ---------
    # P0 = 7.824449066867263 atm
    # T0 = 522.0 K
