# pygasflow

[![PyPI version](https://badge.fury.io/py/pygasflow.svg)](https://badge.fury.io/py/pygasflow)
[![Install with conda](https://anaconda.org/davide_sd/pygasflow/badges/installer/conda.svg)](https://anaconda.org/Davide_sd/pygasflow)
[![Documentation Status](https://readthedocs.org/projects/pygasflow/badge/?version=latest)](https://pygasflow.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Davide-sd/pygasflow/HEAD)

**pygasflow** provides a few handful functions to quickly perform quasi-1D ideal gasdynamic (perfect gas) computations with Python (see requirements below).

The following charts has been generated with the functions included in this package:
<div>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/isentropic.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/fanno.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/rayleigh.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/mach-beta-theta.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/conical-flow.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/pygasflow/master/imgs/shock-reflection.png" width=250/>
</div>

The package contains the following sub-modules:

* `isentropic.py`: functions to solve isentropic flows;
* `fanno.py`: functions to solve Fanno flows;
* `rayleigh.py`: functions to solve Rayleigh flows;
* `showckwaves.py`: functions to solve normal / oblique / conical shock waves;
* `solvers`: the previous modules contains dozens of functions. For convenience, a few solvers have been implemented that, by providing a few parameters, solves the flows by computing the most important ratios (pressure ratio, ..., critical temperature ratio, ...). **These are most likely the functions you will want to use**.
* `nozzles`: functions and classes to understand convergent-divergent nozzles, Rao's TOP nozzles (Thrust Optmizie Parabolic), Minimum Length nozzle with Method of Characteristics. Nozzles can be used to quickly visualize their geometric differences or to solve the isentropic expansion with the `De_Laval_Solver` class. 


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

The easiest way to use this code is to call a solver. The following solvers are implemented:

* `isentropic_solver`
* `fanno_solver`
* `rayleigh_solver`
* `shockwave_solver`: normal shockwave and obliques shock wave.
* `conical_shockwave_solver`
* `De_Laval_Solver`: isentropic expansion through the Convergent-Divergent nozzle.

Let's say you need to solve an isentropic flow: 

```
from pygasflow import isentropic_solver
help(isentropic_solver)
```

Should a solver not be sufficient for your use case, feel free to explore the code implemented inside each flow's type, maybe you'll find a function that suits your needs.

Please:

* take a look at the notebooks contained in the [examples](examples/) folder. You can also try this package online with Binder. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Davide-sd/pygasflow/HEAD)
* visit the [documentation page](https://pygasflow.readthedocs.io/en/latest/).
* If you find any errors, submit an issue or a pull request!
