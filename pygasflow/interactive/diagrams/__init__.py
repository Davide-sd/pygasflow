from bokeh.plotting import show
from pygasflow.interactive.diagrams.isentropic import IsentropicDiagram
from pygasflow.interactive.diagrams.fanno import FannoDiagram
from pygasflow.interactive.diagrams.rayleigh import RayleighDiagram
from pygasflow.interactive.diagrams.normal_shock import NormalShockDiagram
from pygasflow.interactive.diagrams.oblique_shock import ObliqueShockDiagram
from pygasflow.interactive.diagrams.conical_shock import ConicalShockDiagram

def diagram(select="isentropic", interactive=False, **params):
    """Create the selected diagram.

    Parameters
    ----------
    select : str
        Available options are: isentropic, fanno, rayleigh, normal_shock,
        oblique_shock, conical_shock
    interactive : bool
        If True, an Holoviz's panel object will be returned.
    **params :
        Keyword arguments passed to the diagram component for further
        customization.

    Returns
    -------
    diagram : Column or None
        If ``interactive=True``, a panel's ``Column`` object will be
        returned, containing the widgets and the figure. It will be
        automatically rendered on Jupyter Notebook/Lab. If the interpreter
        is unable to render it, execute the ``.show()`` method on the returned
        object.
        If ``interactive=False``, it returns None. The figure will be
        visualized on the screen (either a Jupyter Notebook/Lab's cell or in a
        new browser window).
    """
    mapping = {
        "isentropic": IsentropicDiagram,
        "fanno": FannoDiagram,
        "rayleigh": RayleighDiagram,
        "normal_shock": NormalShockDiagram,
        "oblique_shock": ObliqueShockDiagram,
        "conical_shock": ConicalShockDiagram,
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
    show(diagram.figure)


__all__ = [
    "IsentropicDiagram",
    "FannoDiagram",
    "RayleighDiagram",
    "NormalShockDiagram",
    "ObliqueShockDiagram",
    "ConicalShockDiagram",
    "diagram"
]
