import panel as pn
import param
from pygasflow.interactive.pages import (
    IsentropicPage,
    FannoPage,
    RayleighPage,
    NormalShockPage,
    ObliqueShockPage,
    ConicalShockPage,
    GasPage,
    NozzlesPage
)
from pygasflow.interactive.pages.base import stylesheet


class CompressibleFlow(pn.viewable.Viewer):
    theme = param.String("default", doc="""
        Theme used by this page. Useful to apply custom stylesheet
        to sub-components.""")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pages = [T(theme=self.theme) for T in [
                GasPage,
                IsentropicPage,
                FannoPage,
                RayleighPage,
                NormalShockPage,
                ObliqueShockPage,
                ConicalShockPage,
                NozzlesPage
            ]
        ]
        self.components = {p.page_title: p for p in pages}
        self.tabs = pn.Tabs(
            *list(self.components.items()),
            stylesheets=[stylesheet]
        )

    def __panel__(self):
        return self.tabs


def compressible_app(theme="default"):
    """Create the 'Compressible Flow Calculator' web application.

    Note that while the application works on a browser, all the computation
    is done with Python and the pygasflow module.

    Parameters
    ----------
    theme : str
        Can be ``"default"`` (light theme) or ``"dark"`` for using
        a dark theme.

    Examples
    --------

    Launching a server from a command line. This is the content of
    a user-created file, ``pygasflow_gui.py``:

    .. code-block::

        from pygasflow.interactive import compressible_app
        compressible_app().servable()

    Then, from the command line:

    .. code-block::

        panel serve pygasflow_gui.py

    Launching a server from Jupyter Notebook:

    .. panel-screenshot::

        from pygasflow.interactive import compressible_app
        app = compressible_app()
        app.show()

    """
    if not isinstance(theme, str):
        raise TypeError("`theme` must be a string.")
    theme = theme.lower()
    allowed_themes = ["default", "dark"]
    if theme not in allowed_themes:
        raise ValueError(f"`theme` must be one of {allowed_themes}.")

    i = CompressibleFlow(theme=theme)

    def update_sidebar(tab_idx):
        return list(i.components.values())[tab_idx].controls

    template = pn.template.MaterialTemplate(
        title="Compressible Flow Calculator",
        site="pygasflow",
        site_url="https://pygasflow.readthedocs.io/",
        main_max_width="100%",
        main=i,
        sidebar=pn.bind(update_sidebar, i.tabs.param.active),
        theme=theme
    )
    template.config.raw_css = [
        ".title {font-size: 1em; font-weight: bold;}",
        ".mdc-top-app-bar__row {min-height: 48px;height: 48px;}"
    ]
    return template

