# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
import inspect
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../'))

import pygasflow
from datetime import datetime


# -- Project information -----------------------------------------------------

project = 'pygasflow'
copyright = '%s, Davide Sandonà' % datetime.now().year
author = 'Davide Sandonà'

here = os.path.dirname(__file__)
repo = os.path.join(here, '..', '..')
_version_py = os.path.join(repo, 'pygasflow', '_version.py')
version_ns = {}
with open(_version_py) as f:
    exec (f.read(), version_ns)

v = version_ns["__version__"]
# The short X.Y version
version = ".".join(v.split(".")[:-1])
# The full version, including alpha/beta/rc tags
release = v


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.linkcode',
    'sphinx_math_dollar', 'sphinx.ext.mathjax',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_rtd_theme',
    'nbsphinx',
    'nbsphinx_link', # to link to ipynb files outside of the source folder
]

# hide the table inside classes autodoc
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

blobpath = "https://github.com/Davide-sd/pygasflow/blob/master/pygasflow/"

def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != 'py':
        return

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(pygasflow.__file__))
    return blobpath + fn + linespec

# Replace literal math expression with latex expressions.
# longer expressions first!
replacements = {
    "Cp = Cpt2 * cos(eta)**2": r"$C_{p} = C_{p, t2} \cos^{2}{\eta}$",
    "cos(eta) = cos(alpha) * cos(beta) * sin(theta) - cos(theta) * sin(phi) * sin(beta) - cos(phi) * cos(theta) * sin(alpha) * cos(beta)": r"$\cos{\eta} = \cos{\alpha} \cos{\beta} \sin{\theta} - \cos{\theta} \sin{\phi} \sin{\beta} - \cos{\phi} \cos{\theta} \sin{\alpha} \cos{\beta}$",
    "cos(eta) = 0": r"$\cos{\eta} = 0$",
    "cos(eta)": r"$\cos{\eta}$",
    "Sc -> 0": r"$Sc \rightarrow 0$",
    "Sc -> oo": r"$Sc \rightarrow \infty$",
    "Sc = O(1)": r"$Sc = O(1)$",
    "Pe -> 0": r"$Pe \rightarrow 0$",
    "Pe -> oo": r"$Pe \rightarrow \infty$",
    "Pe = O(1)": r"$Pe = O(1)$",
    "Re -> 0": r"$Re \rightarrow 0$",
    "Re -> oo": r"$Re \rightarrow \infty$",
    "Re = O(1)": r"$Re = O(1)$",
    "Sr = 0": r"$Sr = 0$",
    "Sr -> 0": r"$Sr \rightarrow 0$",
    "Sr = O(1)": r"$Sr = O(1)$",
    "Kn <= 0.01": r"$Kn \lessapprox 0.01$",
    "0.01 <= Kn <= 0.1": r"$0.01 \lessapprox Kn \lessapprox 0.1$",
    "0.1 <= Kn <= 10": r"$0.1 \lessapprox Kn \lessapprox 10$",
    "Kn >= 10": r"$Kn \gtrapprox 10$",
    "Pr -> 0": r"$Pr \rightarrow 0$",
    "Pr -> oo": r"$Pr \rightarrow \infty$",
    "Pr = O(1)": r"$Pr = O(1)$",
    "cos(alpha) * sin(theta) + sin(alpha) * cos(theta) * cos(beta) = 0": r"$\cos{\alpha} \sin{\theta} + \sin{\alpha} \cos{\theta} \cos{\beta}$",
    "beta in [0, 2*pi]": r"$\beta \in [0, 2 \pi]$",
    "x = a * y^2 + b * y + c": r"$x = a y^{2} + b y + c$",
    "xN = a * yN^2 + b * yN + c": r"$x_{N} = a y_{N}^{2} + b y_{N} + c$",
    "xE = a * yE^2 + b * yE + c": r"$x_{E} = a y_{E}^{2} + b y_{E} + c$",
    "dxN / dyN = 2 * a * yN + b = 1 / tan(theta_N)": r"$\frac{d x_{N}}{d y_{N}} = 2 a y_{N} + b = \frac{1}{\tan{\theta_{N}}}$",
    "dxE / dyE = 2 * a * yE + b = 1 / tan(theta_E)": r"$\frac{d x_{E}}{d y_{E}} = 2 a y_{E} + b = \frac{1}{\tan{\theta_{E}}}$",
    "xE - xN": r"$x_{E} - x_{N}$",
    "(A * x + C * y)^2 + D * x + E * y + F = 0": r"$\left(A x + C y\right)^{2} + D x + E y + F = 0$",
    "0 < T/T* < Critical_Temperature_Ratio(0, gamma)": r"$0 < \frac{T}{T^{*}} < \left.\frac{T}{T^{*}}\right|_{M=0, \, \gamma}$",
    "rho/rho* > np.sqrt((gamma - 1) / (gamma + 1))": r"$\frac{\rho}{\rho^{*}} > \sqrt{\frac{\gamma - 1}{\gamma + 1}}$",
    "0 <= U/U* < (1 / np.sqrt((gamma - 1) / (gamma + 1)))": r"$0 \le \frac{U}{U^{*}} < \sqrt{\frac{\gamma + 1}{\gamma - 1}}$",
    "0 <= fp <= ((gamma + 1) * np.log((gamma + 1) / (gamma - 1)) - 2) / (2 * gamma)": r"$0 \le \frac{4 f L^{*}}{D} \le \frac{\left(\gamma + 1\right) \log{\frac{\gamma + 1}{\gamma - 1}} - 2}{2 \gamma}$",
    "(s*-s)/R >= 0": r"$\frac{s^{*} - s}{R} \ge 0$",
    "0 <= rho/rho0 <= 1": r"$0 \le \frac{\rho}{\rho_{0}} \le 1$",
    "A/A* >= 1": r"$\frac{A}{A^{*}} \ge 1$",
    "0 <= Mach Angle <= 90": r"$0 \le \mu \le 90$",
    "0 <= P/P0 <= 1" : r"$0 \le \frac{P}{P_{0}} \le 1$",
    "0 <= T/T0 <= 1" : r"$0 \le \frac{T}{T_{0}} \le 1$",
    "0 <= T0/T0* < 1" : r"$0 \le \frac{T_{0}}{T_{0}^{*}} < 1$",
    "0 < T/T* < T/T*(M(d(CTR)/dM = 0))": r"$0 < \frac{T}{T^{*}} < \frac{T}{T^{*}}\left(\left.M\right|_{\frac{d \left(T / T^{*}\right)}{d M} = 0}, \gamma\right)$",
    "0 < P/P* < P/P*(M=0)": r"$0 < \frac{P}{P^{*}} < \left.\frac{P}{P^{*}}\right|_{M=0, \gamma}$",
    "1 <= P0/P0* < P0/P0*(M=0)": r"$1 \le \frac{P_{0}}{P_{0}^{*}} < \left.\frac{P_{0}}{P_{0}^{*}}\right|_{M=0, \gamma}$",
    "P0/P0* >= 1": r"$\frac{P_{0}}{P_{0}^{*}} \ge 1$",
    "rho/rho* > gamma / (gamma + 1)": r"$\frac{\rho}{\rho^{*}} > \frac{\gamma}{\gamma + 1}$",
    "0 < U/U* < (1 + gamma) / gamma": r"$0 < \frac{U}{U^{*}} < \frac{\gamma + 1}{\gamma}$",
    "1 <= rho2/rho1 < (gamma + 1) / (gamma - 1)": r"$1 \le \frac{\rho_{2}}{\rho_{1}} < \frac{\gamma + 1}{\gamma - 1}$",
    "0 <= P02/P01 <= 1": r"$0 \le \frac{P_{2}^{0}}{P_{1}^{0}} \le 1$",
    "((gamma - 1) / 2 / gamma) < M_2 < 1": r"$\frac{\gamma - 1}{2 \gamma} < M_{2} < 1$",
    "0 <= theta <= 90": r"$0 \le \theta \le 90$",
    "0 <= beta <= 90": r"$0 \le \beta \le 90$",
    "M2 = 1": r"$M_{2} = 1$",
    "M1 >= 1": r"$M_{1} \ge 1$",
    "V_r": r"$V_{r}$",
    "V_theta": r"$V_{\theta}$",
    "mach_angle <= beta <= 90": r"$\mu \, \text(Mach Angle) \le beta \le 90$",
    "0 < theta_c < 90": r"$0 < \theta_{c} < 90$",
    "fp >= 0": r"$\frac{4 f L^{*}}{D} \ge 0$",
    "P/P*": r"$\frac{P}{P^{*}}$",
    "P/P0": r"$\frac{P}{P_{0}}$",
    "T/T*": r"$\frac{T}{T^{*}}$",
    "T/T0": r"$\frac{T}{T_{0}}$",
    "A/A*": r"$\frac{A}{A^{*}}$",
    "rho/rho*": r"$\frac{\rho}{\rho^{*}}$",
    "rho/rho0": r"$\frac{\rho}{\rho_{0}}$",
    "P0/P0*": r"$\frac{P_{0}}{P_{0}^{*}}$",
    "T0/T0*": r"$\frac{T_{0}}{T_{0}^{*}}$",
    "(s*-s)/R": r"$\frac{s^{*} - s}{R}$",
    "4fL*/D": r"$\frac{4 f L^{*}}{D}$",
    "U/U*": r"$\frac{U}{U^{*}}$",
    "gamma > 1": r"$\gamma > 1$",
    "M > 0": r"$M > 0$",
    "M >= 1": r"$M \ge 1$",
    "P2/P1": r"$\frac{P_{2}}{P_{1}}$",
    "T2/T1": r"$\frac{T_{2}}{T_{1}}$",
    "rho2/rho1": r"$\frac{\rho_{2}}{\rho_{1}}$",
    "P02/P01": r"$\frac{P_{2}^{0}}{P_{1}^{0}}$",
    "T02/T01": r"$\frac{T_{2}^{0}}{T_{1}^{0}}$",
    "(s2 - s1) / C_p": r"$\frac{s_{2} - s_{1}}{C_{p}}$",
    "Pt2 / P1": r"$\frac{P_{t2}}{P_{1}}$",
    "Ps / Pt2": r"$\frac{P_{s}}{P_{t2}}$",
}
def replace(app, what, name, obj, options, lines):
    for i in range(len(lines)):
        for k, v in replacements.items():
            if k in lines[i]:
                lines[i] = lines[i].replace(k, v)

def setup(app):
    app.connect('autodoc-process-docstring', replace);
