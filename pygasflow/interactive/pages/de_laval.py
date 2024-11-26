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
from pygasflow.interactive.pages.base import Common, TabulatorSection, _get_tab_header_hover_bg
from pygasflow.interactive.diagrams.de_laval import (
    NozzleDiagram,
    DeLavalDiagram
)


class Tabulator(pn.viewable.Viewer):
    """Display a pn.widgets.Tabulator associated to some dataframe.
    """

    title = param.String(doc="Title for this tabulator.")

    num_decimal_places = param.Integer(
        4, bounds=(2, None), label="Number of decimal places:",
        doc="Controls the number of decimal places shown on the tabulator."
    )

    columns_map = param.Dict({}, doc="""
        Each solver returns a dictionary, mapping string names to numbers.
        Sometime, these names can't be directly used in the columns of a
        dataframe, because it would be meaningless/confusing. This dictionary
        maps the names returned by the results of the computation, to column
        names to be used in the dataframe.""")

    float_formatters_exclusion = param.List(
        doc="List of column names to be excluded by float formatter."
    )

    results = param.DataFrame(doc="""
        Stores the results of the computation which will be shown
        on the tabulator.""")

    filename = param.String("",
        doc="File name for the CSV-file download.")

    theme = param.String("default", doc="""
        Theme used by this page. Useful to choose which stylesheet
        to apply to sub-components.""")

    save_index = param.Boolean(False, doc="""
        Include dataframe index in the CSV file.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._create_tabulator()

    def _df_to_csv_callback(self):
        sio = StringIO()
        self.results.copy().to_csv(sio, index=self.save_index)
        sio.seek(0)
        return sio

    def _create_tabulator(self):
        # NOTE: tabulator needs to be created at __init__ and stored in an
        # attribute in order to be able to modify the formatters when
        # num_decimal_places is changed
        self._tabulator = pn.widgets.Tabulator(
            self.param.results,
            text_align="center",
            header_align="center",
            disabled=True,
            stylesheets=[
                ":host .tabulator {font-size: 12px;}",
                ":host .tabulator-row .tabulator-cell {padding:8px 4px;}"
                ":host .tabulator-row {min-height: 4px;}",
                # the following 3d lines represents a single rule
                ":host .tabulator .tabulator-header"
                " .tabulator-col.tabulator-sortable.tabulator-col-sorter-element:hover"
                " {background-color: %s; !important}" % _get_tab_header_hover_bg(self.theme)
            ],
            formatters={
                name: NumberFormatter(
                    format="0." + "".join(["0"] * self.num_decimal_places))
                for name in self.columns_map.values()
                if name not in self.float_formatters_exclusion
            }
        )

    @param.depends("num_decimal_places", watch=True)
    def _update_formatters(self):
        self._tabulator.formatters = {
            name: NumberFormatter(format="0." + "".join(
                ["0"] * self.num_decimal_places))
            for name in self.columns_map.values()
            if name not in self.float_formatters_exclusion
        }

    def __panel__(self):
        return pn.Column(
            pn.widgets.FileDownload(
                callback=self._df_to_csv_callback,
                filename=self.filename + ".csv"
            ),
            self._tabulator
        )


class BaseSection(pn.viewable.Viewer):
    """Base class for sections to be shown inside a page.
    """

    title = param.String(doc="Title for this section.")

    error_log = param.String(default="", doc="""
        Visualize errors occuring during computation.""")

    num_decimal_places = param.Integer(
        4, bounds=(2, None), label="Number of decimal places:",
        doc="Controls the number of decimal places shown on the tabulator."
    )

    columns_map = param.Dict({}, doc="""
        Each solver returns a dictionary, mapping string names to numbers.
        Sometime, these names can't be directly used in the columns of a
        dataframe, because it would be meaningless/confusing. This dictionary
        maps the names returned by the results of the computation, to column
        names to be used in the dataframe.""")

    wrap_in_card = param.Boolean(True,
        doc="Wrap the tabulator into a collapsible card.")

    solver = param.Callable(doc="""
        Solver to be used to compute numerical results.""")

    results = param.DataFrame(doc="""
        Stores the results of the computation which will be shown
        on the tabulator.""")

    diagram_collapsed = param.Boolean(True,
        doc="Wheter the diagram is hidden (True) or shown right away (False).")

    theme = param.String("default", doc="""
        Theme used by this page. Useful to choose which stylesheet
        to apply to sub-components.""")

    def __init__(self, **params):
        self._tabulators = []
        self._diagrams = []
        for t in params.pop("tabulators", []):
            self._tabulators.append(Tabulator(**t))
        self._diagrams = params.pop("diagrams", [])
        super().__init__(**params)

    @param.depends("num_decimal_places", watch=True)
    def _update_formatters(self):
        for tab in self._tabulators:
            tab.num_decimal_places = self.num_decimal_places

    def _create_elements_for_ui(self):
        elements = []
        for diagram in self._diagrams:
            elements.append(
                pn.Card(
                    diagram,
                    title="Diagram",
                    sizing_mode='stretch_width',
                    collapsed=self.diagram_collapsed
                )
            )
        elements.append(pn.pane.Str(self.param.error_log))
        elements.append(pn.FlexBox(*self._tabulators))
        return elements

    def __panel__(self):
        elements = self._create_elements_for_ui()
        col = pn.Column(*elements)
        if self.wrap_in_card:
            return pn.Card(
                col,
                title=self.title,
                sizing_mode='stretch_width',
                collapsed=False
            )
        return col


class DeLavalSection(BaseSection):
    solver = param.ClassSelector(
        class_=De_Laval_Solver, doc="""
        De Laval instance used to compute numerical results.""")

    def __init__(self, **params):
        solver = De_Laval_Solver(
            P0=8*101325, T0=303.15, R=287.05, gamma=1.4,
            geometry=CD_Min_Length_Nozzle(R0=0)
        )
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
        self._tabulators[0].results = self.solver.flow_conditions
        self._tabulators[1].results = self.solver.flow_states

    def _create_elements_for_ui(self):
        elements = []
        for diagram in self._diagrams:
            elements.append(
                pn.Card(
                    diagram,
                    title="Diagram",
                    sizing_mode='stretch_width',
                    collapsed=self.diagram_collapsed
                )
            )
        elements.append(pn.pane.Str(self.param.error_log))
        elements.append(pn.FlexBox(*self._tabulators))
        return elements


class NozzlesPage(Common, pn.viewable.Viewer):

    nozzle_select = param.Selector(
        objects={"Conical": 0, "TOP": 1, "Minimum Length": 2},
        default=2,
        label="Select nozzle:")

    nozzle_list = param.List(item_type=Nozzle_Geometry)

    def __init__(self, **params):
        # create the two diagrams
        params["nozzle_list"] = [
            CD_Conical_Nozzle(R0=0, Rj=0),
            CD_TOP_Nozzle(R0=0),
            CD_Min_Length_Nozzle(R0=0),
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
