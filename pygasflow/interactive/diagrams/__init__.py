import panel as pn
from pygasflow.interactive.diagrams.isentropic import IsentropicDiagram
from pygasflow.interactive.diagrams.fanno import FannoDiagram
from pygasflow.interactive.diagrams.rayleigh import RayleighDiagram
from pygasflow.interactive.diagrams.normal_shock import NormalShockDiagram
from pygasflow.interactive.diagrams.oblique_shock import ObliqueShockDiagram
from pygasflow.interactive.diagrams.conical_shock import ConicalShockDiagram
from pygasflow.interactive.diagrams.gas import GasDiagram, SonicDiagram
from pygasflow.interactive.diagrams.nozzle import NozzleDiagram
from pygasflow.interactive.diagrams.de_laval import DeLavalDiagram
from pygasflow.interactive.diagrams.pressure_deflection import PressureDeflectionDiagram
from pygasflow.interactive.diagrams.shock_polar import ShockPolarDiagram


def diagram(select="isentropic", interactive=False, show=True, **params):
    """Create the selected diagram.

    Parameters
    ----------
    select : str
        Available options are: isentropic, fanno, rayleigh, normal_shock,
        oblique_shock, conical_shock, gas, sonic, shock_polar
    interactive : bool
        If True, an Holoviz's panel object will be returned. If False,
        something else will be returned: see ``show`` for more info.
    show : bool
        Controls what is going to be returned by the function.
    **params :
        Keyword arguments passed to the diagram component for further
        customization, like ``title``, ``x_label``, ``y_label``,
        ``x_range``, ``y_range``, ``size``.

    Returns
    -------
    diagram : :class:`~panel.layout.Column` or :class:`~bokeh.plotting.figure`
        If ``interactive=True``, a panel's :class:`~panel.layout.Column` object
        will be returned, containing the widgets and the figure. It will be
        automatically rendered on Jupyter Notebook/Lab. If the interpreter
        is unable to render it, execute the ``.show()`` method on the returned
        object.
        If ``interactive=False, show=True``, a panel's
        :class:`~panel.pane.Bokeh` object will be returned, containing
        the figure only. It will be automatically rendered on Jupyter
        Notebook/Lab. If the interpreter is unable to render it, execute the
        ``.show()`` method on the returned object.
        If ``interactive=False, show=False`` it returns the Diagram
        object, which can be used for further customizations.

    Examples
    --------

    Visualize a diagram about the oblique shock properties:

    .. panel-screenshot::
        :large-size: 700,400

        from pygasflow.interactive import diagram
        diagram("oblique_shock", size=(700, 400), interactive=False)

    Visualize an interactive application about the isentropic relations:

    .. panel-screenshot::

        from pygasflow.interactive import diagram
        diagram("isentropic", interactive=True)

    """
    mapping = {
        "isentropic": IsentropicDiagram,
        "fanno": FannoDiagram,
        "rayleigh": RayleighDiagram,
        "normal_shock": NormalShockDiagram,
        "oblique_shock": ObliqueShockDiagram,
        "conical_shock": ConicalShockDiagram,
        "gas": GasDiagram,
        "sonic": SonicDiagram,
        "shock_polar": ShockPolarDiagram,
    }
    if not select in mapping:
        raise ValueError(
            "``select`` must be one of the following"
            f" options: {list(mapping.keys())}."
            f" Instead, '{select}' was received."
        )
    diagram = mapping[select](**params)
    if interactive:
        return diagram.servable()
    if show:
        return pn.pane.Bokeh(diagram.figure).servable()
    return diagram


__all__ = [
    "IsentropicDiagram",
    "FannoDiagram",
    "RayleighDiagram",
    "NormalShockDiagram",
    "ObliqueShockDiagram",
    "ConicalShockDiagram",
    "GasDiagram",
    "SonicDiagram",
    "NozzleDiagram",
    "DeLavalDiagram",
    "PressureDeflectionDiagram",
    "ShockPolarDiagram",
    "diagram",
]
