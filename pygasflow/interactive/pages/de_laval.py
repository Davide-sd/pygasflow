import param
import pandas as pd
import panel as pn
from io import StringIO
from bokeh.models.widgets.tables import NumberFormatter
from pygasflow.solvers import De_Laval_Solver
from pygasflow.nozzles import (
    CD_Conical_Nozzle,
    CD_TOP_Nozzle,
    CD_Min_Length_Nozzle
)
from pygasflow.nozzles.nozzle_geometry import Nozzle_Geometry
from pygasflow.interactive.pages.base import (
    BasePage,
    BaseSection,
    _get_tab_header_hover_bg
)
from pygasflow.interactive.diagrams.de_laval import (
    NozzleDiagram,
    DeLavalDiagram
)


class DeLavalSection(BaseSection):
    solver = param.ClassSelector(
        class_=De_Laval_Solver, doc="""
        De Laval instance used to compute numerical results.""")

    _pointer_to_diagram = param.Parameter(doc="""
        I need this to update the error_log of DeLavalSection when
        an error occurs in the diagram.""")

    def __init__(self, **params):
        solver = params.pop("solver", De_Laval_Solver(
            P0=8*101325, T0=303.15, R=287.05, gamma=1.4,
            geometry=CD_Min_Length_Nozzle(R0=0, is_interactive_app=True)
        ))
        diagram = DeLavalDiagram(solver=solver, _show_all_ui_controls=True)
        params.setdefault("solver", solver)
        params.setdefault("diagrams", [lambda: diagram])
        params.setdefault("diagram_collapsed", False)
        params.setdefault("tabulators", [
            dict(
                filename="nozzle_flow_conditions",
                save_index=True,
                columns_map={
                    "Chocked": "Chocked",
                    "Shock at Exit": "Shock at Exit",
                    "Supercritical": "Supercritical",
                },
            ),
            dict(
                filename="nozzle_states",
                save_index=True,
                columns_map={
                    "Throat": "Throat",
                    "Upstream SW": "Upstream SW",
                    "Downstream SW": "Downstream SW",
                    "Exit": "Exit",
                },
            )
        ])
        params.setdefault("title", "De Laval Section")
        params.setdefault("wrap_in_card", False)
        super().__init__(**params)

    @param.depends("solver.flow_states", watch=True, on_init=True)
    def update_dataframe(self):
        self.tabulators[0].results = self.solver.flow_conditions
        self.tabulators[1].results = self.solver.flow_states

    def _create_elements_for_ui(self):
        elements = []
        for diagram in self.diagrams:
            elements.append(
                pn.Card(
                    diagram,
                    title="Diagram",
                    sizing_mode='stretch_width',
                    collapsed=self.diagram_collapsed
                )
            )
        elements.append(pn.pane.Str(self.param.error_log))
        elements.append(pn.FlexBox(*self.tabulators))
        return elements

    @param.depends(
        "solver.error_log",
        "solver.geometry.error_log",
        watch=True
    )
    def _update_error_log(self):
        msg = self.solver.error_log + "\n" + self.solver.geometry.error_log
        self.error_log = msg


class NozzlesPage(BasePage, pn.viewable.Viewer):

    nozzle_select = param.Selector(
        objects={"Conical": 0, "TOP": 1, "Minimum Length": 2},
        default=2,
        label="Select nozzle:")

    nozzle_list = param.List(item_type=Nozzle_Geometry)

    def __init__(self, **params):
        # create the two diagrams
        params["nozzle_list"] = [
            CD_Conical_Nozzle(R0=0, Rj=0, is_interactive_app=True),
            CD_TOP_Nozzle(R0=0, is_interactive_app=True),
            CD_Min_Length_Nozzle(R0=0, is_interactive_app=True),
        ]
        params.setdefault("page_title", "Nozzles")
        params.setdefault("page_description",
            "Flow in a convergent-divergent nozzle.")
        params.setdefault("sections", [
            DeLavalSection(theme=self.theme)
        ])
        super().__init__(**params)

    @param.depends("nozzle_select", watch=True)
    def _update_geometry(self):
        self.sections[0].solver.geometry = self.nozzle_list[self.nozzle_select]

    def _create_sidebar_controls(self):
        self.controls = pn.Column(
            self.param.nozzle_select,
            "Tabulator:",
            self.param.num_decimal_places
        )

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown(
                self.param.page_description,
                styles={
                    "font-size": "1.2em"
                }
            ),
            *self.sections
        )
