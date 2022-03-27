# pygasflow

[![PyPI version](https://badge.fury.io/py/pygasflow.svg)](https://badge.fury.io/py/pygasflow)
[![Install with conda](https://anaconda.org/davide_sd/pygasflow/badges/installer/conda.svg)](https://anaconda.org/Davide_sd/pygasflow)
[![Documentation Status](https://readthedocs.org/projects/pygasflow/badge/?version=latest)](https://pygasflow.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Davide-sd/pygasflow/HEAD)

**pygasflow** is a Python package that provides a few handful functions to quickly perform:

* Quasi-1D ideal gasdynamic (perfect gas). The following solvers are implemented:
  * ``isentropic_solver`` (or ``ise``).
  * ``fanno_solver`` (or ``fan``).
  * ``rayleigh_solver`` (or ``ray``).
  * ``shockwave_solver`` (or ``ss``) for normal and oblique shock waves.
  * ``conical_shockwave_solver`` (or ``css``).
  * ``De_Laval_solver`` and the ``nozzles`` sub-module, containing functions and classes to understand convergent-divergent nozzles, Rao's TOP nozzles (Thrust Optmizie Parabolic), Minimum Length nozzle with Method of Characteristics. Nozzles can be used to quickly visualize their geometric differences or to solve the isentropic expansion with the `De_Laval_Solver` class. 

* Aerothermodynamic computations (``pygasflow.atd`` module):
  * Correlations to estimate boundary layer thickness, heat flux and wall shear stress over a flat plate or a stagnation region.
  * Newtonian Flow Theory to estimate the pressure distribution around objects and their aerodynamic characteristics. 

The following charts has been generated with the functions included in this package:
<div>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/isentropic.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/fanno.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/rayleigh.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/mach-beta-theta.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/conical-flow.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/shock-reflection.png" width=250/>
</div>


## Installation

The repository is avaliable on PyPi:

```
pip install pygasflow
```

And also on Conda:

```
conda install -c davide_sd pygasflow 
```


## Usage

The easiest way is to call a solver. Let's say you need to solve an isentropic flow: 

```python
from pygasflow import isentropic_solver
help(isentropic_solver)
isentropic_solver("m", 2, to_dict=True)
# {'m': 2.0,
#  'pr': 0.12780452546295096,
#  'dr': 0.2300481458333117,
#  'tr': 0.5555555555555556,
#  'prs': 0.24192491286747442,
#  'drs': 0.36288736930121157,
#  'trs': 0.6666666666666667,
#  'urs': 2.3515101530718505,
#  'ars': 1.6875000000000002,
#  'ma': 30.000000000000004,
#  'pm': 26.379760813416457}
```

Should a solver not be sufficient for your use case, feel free to explore the code implemented inside each flow's type, maybe you'll find a function that suits your needs.

Please:

* take a look at the notebooks contained in the [examples](examples/) folder. You can also try this package online with Binder. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Davide-sd/pygasflow/HEAD)
* visit the [documentation page](https://pygasflow.readthedocs.io/en/latest/).
* If you find any errors, open an issue or submit a pull request!
