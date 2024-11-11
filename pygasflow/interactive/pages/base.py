import numpy as np
import param
import pandas as pd
import panel as pn
from io import StringIO
from itertools import product
from bokeh.models.widgets.tables import NumberFormatter
from pygasflow.interactive.diagrams.flow_base import BasePlot
from pygasflow.shockwave import max_theta_from_mach, beta_from_mach_max_theta


pn.extension()


def _parse_input_string(s):
    """Convert a string to a numpy array."""
    return np.fromstring(s, sep=",")


def _combine(list_of_results):
    """Combine a list of dictionaries sharing the same keys into a
    new dictionary sharing the same keys but aggregating all values.
    """
    types = [type(t) for t in list_of_results]
    if not all(isinstance(t, dict) for t in list_of_results):
        raise TypeError(
            "All `results` must be instances of `dict`. Instead, this"
            " this `list_of_results` was received: %s" % types
        )

    new_results = {}
    for k in list_of_results[0]:
        arr = np.concatenate([
            d[k] if hasattr(d[k], "__iter__") else np.atleast_1d(d[k])
            for d in list_of_results
        ])
        new_results[k] = arr
    return new_results


def _get_tab_header_hover_bg(theme):
    return "#404040" if theme == "dark" else "#e6e6e6"


stylesheet = """
.bk-clearfix hr {
    border:none;
    border-top: 1px solid var(--design-secondary-color, var(--panel-secondary-color));
}
:host(.bk-above) .bk-header .bk-tab {font-size: 1.25em;}
.bk-active {
    color: white !important;
    background-color: var(--pn-tab-active-color) !important;
    font-weight: bold;
}
"""


class TabulatorSection(pn.viewable.Viewer):
    """Base class for sections to be shown inside a page.
    """

    title = param.String(doc="Title for this section.")

    computation_info = param.String(default="", doc="""
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

    internal_map = param.Dict({}, doc="""
        Dropdown selection are used to chose the parameter to be sent
        to the solver. Often, the same quantity may have multiple meanings.
        For example, the value of the critical area ratio in isentropic flow
        can be used to compute solutions both for subsonic and supersonic case.
        This dictionary maps the value used in the dropdown selection
        to a key name used in the `results` dictionary returned by
        the solver.""")

    float_formatters_exclusion = param.List(
        doc="List of column names to be excluded by float formatter."
    )

    wrap_in_card = param.Boolean(True,
        doc="Wrap the tabulator into a collapsible card.")

    solver = param.Callable(doc="""
        Solver to be used to compute numerical results.""")

    results = param.DataFrame(doc="""
        Stores the results of the computation which will be shown
        on the tabulator.""")

    filename = param.String("",
        doc="File name for the CSV-file download.")

    diagram = param.Parameter(
        doc="The class responsible to create a particular diagram.")

    theme = param.String("default", doc="""
        Theme used by this page. Useful to choose which stylesheet
        to apply to sub-components.""")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_tabulator()

    def _df_to_csv_callback(self):
        sio = StringIO()
        self.results.copy().to_csv(sio, index=False)
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
        elements = []
        if self.diagram is not None:
            elements.append(
                pn.Card(
                    self.diagram(),
                    title="Diagram",
                    sizing_mode='stretch_width',
                    collapsed=True
                )
            )
        elements.extend([
            pn.pane.Str(self.param.computation_info),
            pn.widgets.FileDownload(
                callback=self._df_to_csv_callback,
                filename=self.filename + ".csv"
            ),
            self._tabulator
        ])
        col = pn.Column(*elements)
        if self.wrap_in_card:
            return pn.Card(
                col,
                title=self.title,
                sizing_mode='stretch_width',
                collapsed=False
            )
        return col


class FlowSection(TabulatorSection):
    """Base class for sections contained in the following pages:

    * IsentropicPage
    * FannoPage
    * RayleighPage
    * NormalShockPagePage

    """
    gamma = param.Array(np.array([1.4]))

    input_parameter = param.String("m")

    input_value = param.Array(np.array([2]))

    @param.depends("gamma", "input_parameter", "input_value", watch=True)
    def compute(self):
        values = product(self.input_value, self.gamma)

        list_of_results = []
        info = [""]

        # NOTE:
        # isentropic_solver/fanno_solver/rayleigh_solver/normal_shockwave_solver
        # uses numpy vectorization for performance.
        # Here, I sacrifice performance in favor of functionality. For example,
        # suppose the user inputs "0.5, 2" for the pressure ratio. If I were to
        # use vectorized operations, an error will be raised and no results
        # will be returned because all pressure ratios must be 0 <= P/P0 <= 1.
        # Instead, by looping over each provided value, I can catch the faulty
        # value and insert an appropriate row in the dataframe.
        for v, g in values:
            try:
                results = self.solver(
                    self.input_parameter, v, gamma=g, to_dict=True)
            except ValueError as err:
                results = {k: np.nan for k in self.columns_map}
                current_key = self.internal_map.get(
                    self.input_parameter,
                    self.input_parameter,
                )
                results[current_key] = v
                error_msg = "%s" % err
                info.append(
                    "ValueError: %s Instead, %s was received." % (
                        err, g if "specific heats" in error_msg else v
                ))
            results["gamma"] = g * np.ones_like(list(results.values())[0])
            list_of_results.append(results)

        results = _combine(list_of_results)
        rows = [results[k] for k in self.columns_map]
        data = np.vstack(rows).T
        self.computation_info = "\n".join(info)
        self.results = pd.DataFrame(data, columns=self.columns_map.values())


class ShockSection(TabulatorSection):
    """Base class for sections contained in the following pages:

    * ObliqueShockPage
    * ConicalShockPage

    """

    input_parameter_1 = param.String("m1")

    input_value_1 = param.Array(np.array([2]))

    input_parameter_2 = param.String("theta")

    input_value_2 = param.Array(np.array([15]))

    gamma = param.Array(np.array([1.4]))

    input_flag = param.String("both")

    @param.depends(
        "input_parameter_1", "input_value_1",
        "input_parameter_2", "input_value_2",
        "gamma", "input_flag",
        watch=True, on_init=True
    )
    def compute(self):
        values = product(self.input_value_1, self.input_value_2, self.gamma)

        list_of_results = []
        info = [""]
        flags = ["weak"]
        if self.input_flag == "strong":
            flags = ["strong"]
        elif self.input_flag == "both":
            flags = ["weak", "strong"]

        postprocess_results = False
        if (
            ((self.input_parameter_1 == "theta")
                and (self.input_parameter_2 == "beta"))
            or ((self.input_parameter_1 == "beta")
                and (self.input_parameter_2 == "theta"))
        ):
            # Here, we are considering a specific point in the Mach-beta-theta
            # diagram. The important thing is that flag != "both", so only one
            # entry will show up on the dataframe.
            flags = ["weak"]
            # We want to postprocess the results and find in which region
            # (weak or strong) the actual point lies in the diagram.
            postprocess_results = True

        for v1, v2, g in values:
            for f in flags:
                try:
                    results = self.run_solver(v1, v2, g, f)
                    if postprocess_results:
                        m1 = results["m1"]
                        beta_crit = beta_from_mach_max_theta(m1, gamma=g)
                        sol = "weak" if results["beta"] <= beta_crit else "strong"
                        results["Solution"] = [sol]
                except ValueError as err:
                    results = {k: np.nan for k in self.columns_map}
                    current_key_1 = self.internal_map.get(
                        self.input_parameter_1,
                        self.input_parameter_1,
                    )
                    current_key_2 = self.internal_map.get(
                        self.input_parameter_2,
                        self.input_parameter_2,
                    )
                    results[current_key_1] = v1
                    results[current_key_2] = v2
                    error_msg = "%s" % err
                    info.append("ValueError: %s" % err)

                results["gamma"] = g * np.ones_like(results["m1"])
                if not postprocess_results:
                    results["Solution"] = [f] * len(np.atleast_1d(results["m1"]))
                list_of_results.append(results)

        results = _combine(list_of_results)
        rows = [results[k] for k in self.columns_map]
        data = np.vstack(rows).T
        self.computation_info = "\n".join(info)
        results = pd.DataFrame(
            data, columns=self.columns_map.values()
        )
        # this step is necessary, otherwise all columns would be of dtype
        # str, requiring more steps in testing
        dtypes = {}
        for c in results.columns:
            dtypes[c] = np.float64 if c != "Solution" else str
        results = results.astype(dtypes)
        self.results = results


class Common(param.Parameterized):
    """Base class for pages.
    """

    page_title = param.String("", doc="Title of the page.")

    page_description = param.String("", doc="""
        Brief description of what a page will help to compute.""")

    num_decimal_places = param.Integer(
        4, bounds=(2, None), label="Number of decimal places:",
        doc="Controls the number of decimal places shown on tabulators."
    )

    sections = param.List(doc="Store sections to be shown on the page.")

    theme = param.String("default")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_sidebar_controls()

    @param.depends("num_decimal_places", watch=True, on_init=True)
    def _update_formatters(self):
        for s in self.sections:
            s.num_decimal_places = self.num_decimal_places


class FlowPage(Common, pn.viewable.Viewer):
    """Base class for the following pages:

    * IsentropicPage
    * FannoPage
    * RayleighPage
    * NormalShockPagePage

    """

    input_value = param.String(
        default="2",
        label="Parameter values:",
        doc="Comma separated list of values."
    )

    gamma = param.String(
        default="1.4",
        label="Ratio of specific heats, γ:",
        doc="""
            Comma separated list of ratio of specific heats.
            Each value must be greater than 1."""
    )

    def _create_sidebar_controls(self):
        self.controls = pn.Column(
            self.param.input_parameter,
            self.param.input_value,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.gamma,
            pn.layout.Divider(stylesheets=[stylesheet]),
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


class ShockPage(Common, pn.viewable.Viewer):
    """Base class for the following pages:

    * ObliqueShockPage
    * ConicalShockPage

    """

    input_flag = param.Selector(
        label="Solution to compute:",
        objects=["weak", "strong", "both"],
        default="both"
    )

    gamma = param.String(
        default="1.4",
        label="Ratio of specific heats, γ:",
        doc="""
            Comma separated list of ratio of specific heats.
            Each value must be greater than 1."""
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
