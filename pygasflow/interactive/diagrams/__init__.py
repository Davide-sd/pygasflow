from bokeh.plotting import show as bokeh_show
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


def diagram(select="isentropic", interactive=False, show=True, **params):
    """Create the selected diagram.

    Parameters
    ----------
    select : str
        Available options are: isentropic, fanno, rayleigh, normal_shock,
        oblique_shock, conical_shock, gas, sonic
    interactive : bool
        If True, an Holoviz's panel object will be returned. If False,
        a Bokeh figure will be returned.
    show : bool
        If ``interactive=False``, controls wheter the Bokeh figure should be
        shown on the screen.
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
        If ``interactive=False``, it returns a Bokeh
        :class:`~bokeh.plotting.figure`, which will be also visualized on the
        screen (either a Jupyter Notebook/Lab's cell or in a new
        browser window).

    Examples
    --------

    Visualize a diagram about the oblique shock properties:

    .. bokeh-plot::
        :source-position: above

        from pygasflow.interactive import diagram
        from bokeh.plotting import show
        show(diagram("oblique_shock", size=(700, 400), show=False))

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
        bokeh_show(diagram.figure)
    return diagram.figure


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
    "diagram",
]
