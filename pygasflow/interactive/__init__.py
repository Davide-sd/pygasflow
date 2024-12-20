"""
Objectives of the interactive modules:

1. to have a web-based GUI (Graphical User Interface) always available to
   the user.
2. to simplify even further the use of ``pygasflow``.
3. to make parametric visualizations easier than ever, thus improving
   the learning experience about quasi-1D ideal gasdynamic.

The interactive module is built with `Holoviz Param <https://param.holoviz.org/>`_
and `Panel <https://panel.holoviz.org/>`_. It is composed of several components:

* Diagrams, used to visualize the general results of a particular solver.
* Tabulators, used to visualize dataframes containing the numerical results
  of a particular solver.
* Sections, composed of diagrams and/or tabulators.
* Pages (or tabs), composed of different sections.
* Overall application, composed of different pages (or tabs).

The following are the most important functions available to the users.

"""

from pygasflow.interactive.compressible import compressible_app
from pygasflow.interactive.diagrams import PressureDeflectionDiagram

__all__ = [
   "compressible_app",
   "PressureDeflectionDiagram"
]
