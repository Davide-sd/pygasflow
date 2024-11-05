import panel as pn
from pygasflow.interactive.pages import (
    IsentropicPage,
    FannoDiagram,
    RayleighDiagram,
    NormalShockDiagram,
    ObliqueShockDiagram,
    ConicalShockDiagram
)


class CompressibleFlow(pn.viewable.Viewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components = {
            T.page_title: T() for T in [
                IsentropicPage,
                FannoDiagram,
                RayleighDiagram,
                NormalShockDiagram,
                ObliqueShockDiagram,
                ConicalShockDiagram
            ]
        }
        self.tabs = pn.Tabs(
            *list(self.components.items()),
            stylesheets=[stylesheet]
        )

    def __panel__(self):
        return self.tabs
