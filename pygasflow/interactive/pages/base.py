import numpy as np
import param
import pandas as pd
import panel as pn
from io import StringIO
from itertools import product
from bokeh.models.widgets.tables import NumberFormatter
from pygasflow.interactive.diagrams.flow_base import BasePlot


pn.extension()


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


class Common(param.Parameterized):
    page_title = param.String("", doc="Title of the page.")

    page_description = param.String("", doc="""
        Brief description of what a page will help to compute.""")

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

    results = param.DataFrame(
        doc="Stores the results of the computation."
    )

    num_decimal_places = param.Integer(
        4, bounds=(2, None), label="Number of decimal places:"
    )

    computation_info = param.String(default="")

    _filename = param.String("", doc="File name for the CSV-file download.")

    _solver = param.Callable(doc="""
        Solver to be used to compute numerical results.""")

    _diagram = param.ClassSelector(class_=BasePlot, is_instance=False,
        doc="The class responsible to create a particular diagram.")

    _theme = param.String("default", doc="""
        Theme used by this page. Useful to apply custom stylesheet
        to sub-components.""")

    # list of column names to be excluded by float formatter
    _float_formatters_exclusion = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_sidebar_controls()
        self._create_tabulator()

    def _parse_input_string(self, s):
        return np.fromstring(s, sep=",")

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
                " {background-color: %s; !important}" % _get_tab_header_hover_bg(self._theme)
            ],
            formatters={
                name: NumberFormatter(
                    format="0." + "".join(["0"] * self.num_decimal_places))
                for name in self._columns.values()
                if name not in self._float_formatters_exclusion
            }
        )

    @param.depends("num_decimal_places", watch=True, on_init=True)
    def _update_formatters(self):
        if hasattr(self, "_tabulator"):
            self._tabulator.formatters = {
                name: NumberFormatter(format="0." + "".join(
                    ["0"] * self.num_decimal_places))
                for name in self._columns.values()
                if name not in self._float_formatters_exclusion
            }

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
            pn.Card(
                self._diagram(),
                title="Diagram",
                sizing_mode='stretch_width',
                collapsed=True
            ),
            pn.pane.Str(self.param.computation_info),
            pn.widgets.FileDownload(
                callback=self._df_to_csv_callback,
                filename=self._filename + ".csv"
            ),
            self._tabulator
        )
        return pn.panel(self.param)

    def _compute(self):
        value = self._parse_input_string(self.input_value)
        gamma = self._parse_input_string(self.gamma)
        values = product(value, gamma)

        list_of_results = []
        info = [""]

        # NOTE: isentropic_solver uses numpy vectorization for performance.
        # Here, I sacrifice performance in favor of functionality. For example,
        # suppose the user inputs "0.5, 2" for the pressure ratio. If I were to
        # use vectorized operations, an error will be raised and no results
        # will be returned because all pressure ratios must be 0 <= P/P0 <= 1.
        # Instead, by looping over each provided value, I can catch the faulty
        # value and insert an appropriate row in the dataframe.
        for v, g in values:
            try:
                results = self._solver(
                    self.input_parameter, v, gamma=g, to_dict=True)
            except ValueError as err:
                results = {k: np.nan for k in self._columns}
                results[self._internal_map[self.input_parameter]] = v
                error_msg = "%s" % err
                info.append(
                    "ValueError: %s Instead, %s was received." % (
                        err, g if "specific heats" in error_msg else v
                ))
            results["gamma"] = g * np.ones_like(results["m"])
            list_of_results.append(results)

        results = _combine(list_of_results)
        rows = [results[k] for k in self._columns]
        data = np.vstack(rows).T
        self.computation_info = "\n".join(info)
        self.results = pd.DataFrame(data, columns=self._columns.values())
