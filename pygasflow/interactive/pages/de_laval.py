import param
import panel as pn
from pygasflow.solvers import De_Laval_Solver
from pygasflow.nozzles import (
    CD_Conical_Nozzle,
    CD_TOP_Nozzle,
    CD_Min_Length_Nozzle
)
from pygasflow.nozzles.nozzle_geometry import Nozzle_Geometry
from pygasflow.interactive.pages.base import (
    BasePage,
    BaseSection
)
from pygasflow.interactive.diagrams.de_laval import DeLavalDiagram


class DeLavalSection(BaseSection):
    solver = param.ClassSelector(
        class_=De_Laval_Solver, doc="""
        De Laval instance used to compute numerical results.""")

    _pointer_to_diagram = param.Parameter(doc="""
        I need this to update the error_log of DeLavalSection when
        an error occurs in the diagram.""")

    _pointer_to_mass_flow_rate = param.String(doc="""
        This parameter will be used to show the mass flow rate inside
        the current section.""")

    def __init__(self, **params):
        solver = params.pop("solver", De_Laval_Solver(
            P0=8*101325, T0=303.15, R=287.05, gamma=1.4,
            nozzle=CD_Min_Length_Nozzle(R0=0, is_interactive_app=True)
        ))
        diagram = DeLavalDiagram(solver=solver, _show_all_ui_controls=True)
        params.setdefault("solver", solver)
        params.setdefault("diagrams", [lambda: diagram])
        params.setdefault("diagram_collapsed", False)
        params.setdefault("tabulators", [
            dict(
                filename="nozzle_flow_summary",
                save_index=True,
                columns_map={
                    "Flow Type": "Flow Type",
                    "Condition": "Condition",
                },
                float_formatters_exclusion=["Condition"]
            ),
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

    @param.depends(
        "solver.flow_states", "solver.flow_condition_summary",
        watch=True, on_init=True)
    def update_dataframe(self):
        self.tabulators[0].results = self.solver.flow_condition_summary
        self.tabulators[1].results = self.solver.flow_conditions
        self.tabulators[2].results = self.solver.flow_states

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
        elements.append(pn.pane.Markdown(
            self.param._pointer_to_mass_flow_rate,
            styles={"font-size": "16px"}
        )),
        elements.append(pn.pane.Str(self.param.error_log))
        elements.append(pn.FlexBox(*self.tabulators))
        return elements

    @param.depends(
        "solver.error_log",
        "solver.nozzle.error_log",
        watch=True
    )
    def _update_error_log(self):
        msg = self.solver.error_log + "\n" + self.solver.nozzle.error_log
        self.error_log = msg

    @param.depends(
        "solver.mass_flow_rate",
        "num_decimal_places",
        watch=True, on_init=True
    )
    def _update_mass_flow_rate(self):
        self._pointer_to_mass_flow_rate = "Mass flow rate [kg/s]: " + str(round(
            self.solver.mass_flow_rate, self.num_decimal_places))
        self.solver._num_decimal_places = self.num_decimal_places


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
            DeLavalSection(theme=params.get("theme", "default"))
        ])
        super().__init__(**params)

    @param.depends("nozzle_select", watch=True)
    def _update_nozzle(self):
        self.sections[0].solver.nozzle = self.nozzle_list[self.nozzle_select]

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
