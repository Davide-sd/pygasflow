import panel as pn
from pygasflow.interactive.pages import (
    IsentropicPage,
    FannoPage,
    RayleighPage,
    NormalShockPage,
    ObliqueShockPage,
    ConicalShockPage
)
from pygasflow.interactive.pages.base import stylesheet


class CompressibleFlow(pn.viewable.Viewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pages = [T() for T in [
                IsentropicPage,
                FannoPage,
                RayleighPage,
                NormalShockPage,
                ObliqueShockPage,
                ConicalShockPage
            ]
        ]
        self.components = {p.page_title: p for p in pages}
        self.tabs = pn.Tabs(
            *list(self.components.items()),
            stylesheets=[stylesheet]
        )

    def __panel__(self):
        return self.tabs


def compressible_app():
    i = CompressibleFlow()

    def update_sidebar(tab_idx):
        return list(i.components.values())[tab_idx].controls

    template = pn.template.MaterialTemplate(
        title="Compressible Flow Calculator",
        site="pygasflow",
        site_url="https://pygasflow.readthedocs.io/",
        main_max_width="100%",
        main=i,
        sidebar=pn.bind(update_sidebar, i.tabs.param.active),
        theme="dark"
    )
    template.config.raw_css = [
        ".title {font-size: 1em; font-weight: bold;}",
        ".mdc-top-app-bar__row {min-height: 48px;height: 48px;}"
    ]
    return template.servable()

