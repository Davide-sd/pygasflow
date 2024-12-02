import numpy as np
import panel as pn
from pygasflow.interactive.diagrams import (
    IsentropicDiagram,
    FannoDiagram,
    RayleighDiagram,
    NormalShockDiagram,
    ObliqueShockDiagram,
    ConicalShockDiagram,
    GasDiagram,
    DeLavalDiagram
)
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
from pygasflow.solvers import (
    isentropic_solver,
    fanno_solver,
    rayleigh_solver,
    normal_shockwave_solver,
    shockwave_solver,
    conical_shockwave_solver,
    gas_solver,
    ideal_gas_solver,
    De_Laval_Solver
)
from pygasflow.nozzles import CD_Conical_Nozzle
from pygasflow.interactive.pages.isentropic import IsentropicSection
from pygasflow.interactive.pages.fanno import FannoSection
from pygasflow.interactive.pages.rayleigh import RayleighSection
from pygasflow.interactive.pages.normal_shock import NormalShockSection
from pygasflow.interactive.pages.oblique_shock import ObliqueShockSection
from pygasflow.interactive.pages.conical_shock import ConicalShockSection
from pygasflow.interactive.pages.gas import GasSection, IdealGasSection
from pygasflow.interactive.pages.de_laval import DeLavalSection
import pytest


expected = {
    IsentropicPage: {
        "page_title": "Isentropic",
        "page_description": "Adiabatic and reversible 1D flow.",
        "num_sections": 1,
        "sections": (IsentropicSection, ),
    },
    FannoPage: {
        "page_title": "Fanno",
        "page_description": "1D flow with friction.",
        "num_sections": 1,
        "sections": (FannoSection, ),
    },
    RayleighPage: {
        "page_title": "Rayleigh",
        "page_description": "1D flow with heat addition.",
        "num_sections": 1,
        "sections": (RayleighSection, ),
    },
    NormalShockPage: {
        "page_title": "Normal Shock",
        "page_description": "Change in properties caused by a shock wave perpendicular to a 1D flow.",
        "num_sections": 1,
        "sections": (NormalShockSection, ),
    },
    ObliqueShockPage: {
        "page_title": "Oblique Shock",
        "page_description": "Change in properties of a 1D flow caused by an oblique shock wave.",
        "num_sections": 1,
        "sections": (ObliqueShockSection, ),
    },
    ConicalShockPage: {
        "page_title": "Conical Shock",
        "page_description": "Change in properties of an axisymmetric supersonic flow over a sharp cone",
        "num_sections": 1,
        "sections": (ConicalShockSection, ),
    },
    GasPage: {
        "page_title": "Gas",
        "page_description": "Compute gas-related quantities.",
        "num_sections": 2,
        "sections": (GasSection, IdealGasSection, ),
    },
    NozzlesPage: {
        "page_title": "Nozzles",
        "page_description": "Flow in a convergent-divergent nozzle.",
        "num_sections": 1,
        "sections": (DeLavalSection, ),
    },
}

expected_sections = {
    IsentropicSection: {
        "title": "Isentropic Section",
        "filename": "isentropic",
        "diagram": IsentropicDiagram,
        "solver": isentropic_solver,
    },
    FannoSection: {
        "title": "Fanno Section",
        "filename": "fanno",
        "diagram": FannoDiagram,
        "solver": fanno_solver,
    },
    RayleighSection: {
        "title": "Rayleigh Section",
        "filename": "rayleigh",
        "diagram": RayleighDiagram,
        "solver": rayleigh_solver,
    },
    NormalShockSection: {
        "title": "Normal Shock Section",
        "filename": "normal_shock",
        "diagram": NormalShockDiagram,
        "solver": normal_shockwave_solver,
    },
    ObliqueShockSection: {
        "title": "Oblique Shock Wave Section",
        "filename": "oblique_shock",
        "diagram": ObliqueShockDiagram,
        "solver": shockwave_solver,
    },
    ConicalShockSection: {
        "title": "Conical Shock Wave Section",
        "filename": "conical_shock",
        "diagram": ConicalShockDiagram,
        "solver": conical_shockwave_solver,
    },
    GasSection: {
        "title": "Gas",
        "filename": "gas",
        "diagram": GasDiagram,
        "solver": gas_solver,
    },
    IdealGasSection: {
        "title": "Ideal Gas",
        "filename": "ideal_gas",
        "diagram": None,
        "solver": ideal_gas_solver,
    },
}


@pytest.mark.parametrize("SectionClass", [
    IsentropicSection,
    FannoSection,
    RayleighSection,
    NormalShockSection,
    ObliqueShockSection,
    ConicalShockSection,
    GasSection,
    IdealGasSection
])
def test_sections_instantiation(SectionClass):
    s = SectionClass()
    assert s.title == expected_sections[SectionClass]["title"]
    assert s.tabulators[0].filename == expected_sections[SectionClass]["filename"]
    if expected_sections[SectionClass]["diagram"] is not None:
        assert s.diagrams[0] is expected_sections[SectionClass]["diagram"]
    else:
        assert len(s.diagrams) == 0
    assert s.solver is expected_sections[SectionClass]["solver"]


def fix_numerical_errors(column_values, expected_values):
    """When an input value is given to a solver, the returned dictionary
    may contain slight variations of the input value, caused by numerical
    errors in the solver.

    This function take the values of a dataframe column, and compare it with
    the expected values. It returns "rounded" values according
    to expected_values.

    Examples
    --------
    >>> expected = [1, 2, 3]
    >>> col = [t + 1e-08 for t in [3, 1, 2, 1, 3, 2]]
    >>> fix_numerical_errors(col, expected)
    [3, 1, 2, 1, 3, 2]
    """
    output_values = np.zeros_like(column_values)
    for i in range(len(column_values)):
        for v in expected_values:
            if np.isclose(v - column_values[i], 0):
                output_values[i] = v
                break
    return output_values


@pytest.mark.parametrize("SectionClass", [
    IsentropicSection,
    FannoSection,
    RayleighSection,
    NormalShockSection,
])
def test_update_flow_related_sections(SectionClass):
    s = SectionClass()
    assert len(s.results) == 1

    with pytest.raises(ValueError):
        # input value must be a numpy array
        s.input_value = "2, 4, 6"

    m = np.array([2, 4, 6])
    s.input_value = m
    assert len(s.results) == 3
    k = "Upstream Mach" if SectionClass is NormalShockSection else "Mach"
    assert len(set(s.results[k]).difference(m)) == 0
    assert s.error_log == ""

    s.gamma = np.array([1.1, 1.2, 1.3, 1.4])
    assert len(s.results) == 12
    assert s.error_log == ""

    r = np.array([0.1, 0.2])
    s.input_value = r
    s.input_parameter = "total_pressure" if SectionClass is NormalShockSection else "pressure"
    assert len(s.results) == 8
    if SectionClass is NormalShockSection:
        k = "P02/P01"
    elif SectionClass is IsentropicSection:
        k = "P/P0"
    else:
        k = "P/P*"
    # need to account for the numerical errors...
    values = s.results[k].values
    sorted_values = list(sorted(set(fix_numerical_errors(values, r))))
    assert np.allclose(sorted_values, r)

    s.param.update(dict(
        input_parameter = "m",
        gamma = np.array([1.4, 0.8]),
        input_value = m
    ))
    assert len(s.results) == 6
    # errors because gamma=0.8 is < 1
    assert s.error_log != ""


@pytest.mark.parametrize("SectionClass", [
    ObliqueShockSection,
    ConicalShockSection,
])
def test_update_shock_related_sections(SectionClass):
    s = SectionClass()
    assert len(s.results) == 2 # weak and strong solutions are shown

    with pytest.raises(ValueError):
        # input value must be a numpy array
        s.input_value_1 = "2, 4, 6"

    with pytest.raises(ValueError):
        # input value must be a numpy array
        s.input_value_2 = "2, 4, 6"

    # test for input_parameter_1="m1" and input_parameter_2="theta"
    k_theta = "theta" if SectionClass is ObliqueShockSection else "theta_c"
    m1 = np.array([2, 4, 6])
    theta = np.array([15, 18, 20])
    # NOTE: one shot update to speed up test
    s.param.update(dict(
        input_parameter_1 = "m1",
        input_value_1 = m1,
        input_parameter_2 = k_theta,
        input_value_2 = theta,
        gamma = np.array([1.1, 1.2, 1.3, 1.4]),
        input_flag = "weak"
    ))
    assert len(s.results) == 36
    assert s.error_log == ""
    assert np.allclose(
        list(sorted(set(
            fix_numerical_errors(s.results["Upstream Mach"].values, m1)))),
        m1
    )
    k = "θ [deg]" if SectionClass is ObliqueShockSection else "θ_c [deg]"
    assert np.allclose(
        list(sorted(set(fix_numerical_errors(s.results[k].values, theta)))),
        theta
    )

    s.param.update(dict(
        input_parameter_1 = "m1",
        input_value_1 = np.array([2, 3, 4]),
        input_parameter_2 = "theta",
        input_value_2 = np.array([15, 20]),
        gamma = np.array([1.4, 0.8]),
        input_flag = "both"
    ))
    assert len(s.results) == 24
    # errors because gamma=0.8 is < 1
    assert s.error_log != ""


def test_update_oblique_shock_section_theta_beta():
    # test for input_parameter_1="theta", input_parameter_2="beta"
    # here, input_flag has no meaning: strong or weak will be determined
    # by the algorithm

    s = ObliqueShockSection()
    s.gamma = np.array([1.4])
    s.input_parameter_1 = "theta"
    s.input_parameter_2 = "beta"
    theta = np.array([10, 20])
    beta = np.array([20, 40, 60, 80, 90])
    s.input_value_1 = theta
    s.input_value_2 = beta
    assert len(s.results) == 10
    k = "θ [deg]"
    assert np.allclose(
        list(sorted(set(
            fix_numerical_errors(s.results[k].values, theta)))),
        theta
    )
    assert np.allclose(
        list(sorted(set(
            fix_numerical_errors(s.results["β [deg]"].values, beta)))),
        beta
    )
    # nan indicates there is no solution for the specified pairs
    # of theta, beta
    expected_solutions = [
        "weak", "weak", "weak", "strong", "nan", "nan",
        "weak", "weak", "strong", "nan"
    ]
    for sol1, sol2 in zip(s.results["Solution"], expected_solutions):
        assert sol1 == sol2


@pytest.mark.parametrize("SectionClass", [
    IsentropicSection,
    FannoSection,
    RayleighSection,
    NormalShockSection,
    ObliqueShockSection,
    ConicalShockSection,
    GasSection,
    IdealGasSection
])
def test_section_columns_names(SectionClass):
    # verify that appropriate column names are used in the dataframe
    s = SectionClass()
    if SectionClass not in [ObliqueShockSection, ConicalShockSection]:
        assert s.results.shape == (1, len(s.tabulators[0].columns_map))
    else:
        assert s.results.shape == (2, len(s.tabulators[0].columns_map))
    assert len(set(s.tabulators[0].columns_map.values()).difference(
        s.results.columns)) == 0
    # verify the order is correct
    for name1, name2 in zip(
        s.tabulators[0].columns_map.values(), s.results.columns):
        assert name1 == name2


def test_de_laval_section_errors():
    n = CD_Conical_Nozzle(R0=0, Rj=0, Ri=0.4, Rt=0.2, Re=0.6)
    solver = De_Laval_Solver(R=287.05, gamma=1.4, P0=101325, T0=298,
        Pb_P0_ratio=0.1, nozzle=n)
    s = DeLavalSection(solver=solver)
    assert len(s.tabulators) == 2
    assert len(s.diagrams) == 1
    assert isinstance(s.diagrams[0](), DeLavalDiagram)
    assert isinstance(s.solver.nozzle, CD_Conical_Nozzle)
    assert s.error_log == ""
    n.throat_radius = 0.7
    assert s.error_log != ""


@pytest.mark.parametrize("PageClass", [
    IsentropicPage,
    FannoPage,
    RayleighPage,
    NormalShockPage,
    ObliqueShockPage,
    ConicalShockPage,
    GasPage,
    NozzlesPage
])
def test_page_instantiation(PageClass):
    p = PageClass()
    assert len(p.sections) == expected[PageClass]["num_sections"]
    assert all(
        isinstance(s, expected[PageClass]["sections"])
        for s in p.sections
    )
    assert p.page_title == expected[PageClass]["page_title"]
    assert (
        (p.page_description == expected[PageClass]["page_description"])
        or (expected[PageClass]["page_description"] in p.page_description)
    )


@pytest.mark.parametrize("PageClass", [
    IsentropicPage,
    FannoPage,
    RayleighPage,
    NormalShockPage,
    ObliqueShockPage,
    ConicalShockPage,
    GasPage,
    NozzlesPage
])
def test_page_content(PageClass):
    p = PageClass()
    panel = p.__panel__()
    assert isinstance(panel, pn.Column)
    # page description
    assert isinstance(panel.objects[0], pn.pane.Markdown)
    # section
    assert isinstance(panel.objects[1], pn.Column)


@pytest.mark.parametrize("PageClass", [
    IsentropicPage,
    FannoPage,
    RayleighPage,
    NormalShockPage,
])
def test_flow_related_page_ui_controls(PageClass):
    p = PageClass()
    sidebar_controls = p.controls
    assert isinstance(sidebar_controls, pn.Column)
    assert len(sidebar_controls.objects) == 6
    assert isinstance(sidebar_controls.objects[0], pn.widgets.Select)
    assert isinstance(sidebar_controls.objects[1], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[2], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[3], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[4], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[5], pn.widgets.IntInput)


def test_oblique_shock_related_page_ui_controls():
    p = ObliqueShockPage()
    sidebar_controls = p.controls
    assert isinstance(sidebar_controls, pn.Column)
    assert len(sidebar_controls.objects) == 10
    assert isinstance(sidebar_controls.objects[0], pn.widgets.Select)
    assert isinstance(sidebar_controls.objects[1], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[2], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[3], pn.widgets.Select)
    assert isinstance(sidebar_controls.objects[4], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[5], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[6], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[7], pn.widgets.Select)
    assert isinstance(sidebar_controls.objects[8], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[9], pn.widgets.IntInput)


def test_conical_shock_related_page_ui_controls():
    p = ConicalShockPage()
    sidebar_controls = p.controls
    assert isinstance(sidebar_controls, pn.Column)
    assert len(sidebar_controls.objects) == 9
    assert isinstance(sidebar_controls.objects[0], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[1], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[2], pn.widgets.Select)
    assert isinstance(sidebar_controls.objects[3], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[4], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[5], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[6], pn.widgets.Select)
    assert isinstance(sidebar_controls.objects[7], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[8], pn.widgets.IntInput)


def test_gas_page_ui_controls():
    p = GasPage()
    sidebar_controls = p.controls
    assert isinstance(sidebar_controls, pn.Column)
    assert len(sidebar_controls.objects) == 15
    assert isinstance(sidebar_controls.objects[0], pn.pane.Markdown)
    assert isinstance(sidebar_controls.objects[1], pn.widgets.MultiChoice)
    assert isinstance(sidebar_controls.objects[2], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[3], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[4], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[5], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[6], pn.pane.Markdown)
    assert isinstance(sidebar_controls.objects[7], pn.widgets.Select)
    assert isinstance(sidebar_controls.objects[8], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[9], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[10], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[11], pn.widgets.TextInput)
    assert isinstance(sidebar_controls.objects[12], pn.layout.Divider)
    assert isinstance(sidebar_controls.objects[13], pn.pane.Markdown)
    assert isinstance(sidebar_controls.objects[14], pn.widgets.IntInput)


@pytest.mark.parametrize("PageClass", [
    IsentropicPage,
    FannoPage,
    RayleighPage,
    NormalShockPage,
])
def test_flow_related_pages_parameter_propagation(PageClass):
    # verify that parameters set on pages are correctly passed down
    # to sections

    p = PageClass()
    s = p.sections[0]
    param_name = "m1" if PageClass is NormalShockPage else "m"
    p.input_parameter = param_name
    p.input_value = "2, 3, 5, 8"
    p.gamma = "1.1, 1.3, 1.5"
    assert s.input_parameter == param_name
    assert np.allclose(s.input_value, [2, 3, 5, 8])
    assert np.allclose(s.gamma, [1.1, 1.3, 1.5])

    p.input_parameter = "pressure"
    p.input_value = "0.1, 0.2, 0.3"
    assert s.input_parameter == "pressure"
    assert np.allclose(s.input_value, [0.1, 0.2, 0.3])


def test_oblique_shock_page_parameter_propagation():
    # verify that parameters set on pages are correctly passed down
    # to sections

    p = ObliqueShockPage()
    s = p.sections[0]
    param_name1 = "m1"
    param_name2 = "theta"
    p.input_parameter_1 = param_name1
    p.input_parameter_2 = param_name2
    p.input_value_1 = "2, 3, 5, 8"
    p.input_value_2 = "10, 20, 30"
    p.gamma = "1.1, 1.3, 1.5"
    assert s.input_parameter_1 == param_name1
    assert s.input_parameter_2 == param_name2
    assert np.allclose(s.input_value_1, [2, 3, 5, 8])
    assert np.allclose(s.input_value_2, [10, 20, 30])
    assert np.allclose(s.gamma, [1.1, 1.3, 1.5])

    param_name1 = "pressure"
    param_name2 = "beta"
    p.input_parameter_1 = param_name1
    p.input_parameter_2 = param_name2
    p.input_value_1 = "5, 6"
    p.input_value_2 = "10, 20, 30, 40"
    assert s.input_parameter_1 == param_name1
    assert s.input_parameter_2 == param_name2
    assert np.allclose(s.input_value_1, [5, 6])
    assert np.allclose(s.input_value_2, [10, 20, 30, 40])
    assert np.allclose(s.gamma, [1.1, 1.3, 1.5])


def test_conical_shock_page_parameter_propagation():
    # verify that parameters set on pages are correctly passed down
    # to sections

    p = ConicalShockPage()
    s = p.sections[0]
    param_name2 = "theta_c"
    p.input_parameter_2 = param_name2
    p.input_value_1 = "2, 3, 5, 8"
    p.input_value_2 = "10, 20, 30"
    p.gamma = "1.1, 1.3, 1.5"
    assert s.input_parameter_1 == "m1"
    assert s.input_parameter_2 == param_name2
    assert np.allclose(s.input_value_1, [2, 3, 5, 8])
    assert np.allclose(s.input_value_2, [10, 20, 30])
    assert np.allclose(s.gamma, [1.1, 1.3, 1.5])

    param_name2 = "beta"
    p.input_parameter_2 = param_name2
    p.input_value_1 = "5, 6"
    p.input_value_2 = "10, 20, 30, 40"
    assert s.input_parameter_1 == "m1"
    assert s.input_parameter_2 == param_name2
    assert np.allclose(s.input_value_1, [5, 6])
    assert np.allclose(s.input_value_2, [10, 20, 30, 40])
    assert np.allclose(s.gamma, [1.1, 1.3, 1.5])


def test_gas_page_parameter_propagation():
    p = GasPage()
    s1, s2 = p.sections
    assert len(s1.results) == 1
    assert len(s2.results) == 1

    # gas section
    p.input_value_1 = "1000, 2000"
    p.input_value_2 = "500, 700, 900"
    p.input_temperature = "298, 400"
    assert len(s1.results) == 12

    p.input_parameter_1 = ["cp", "cv"]
    assert len(s1.results) == 2

    p.input_value_3 = "100, 200"
    p.input_value_4 = "1, 2, 3"
    p.input_value_5 = "300, 400"
    p.gamma = "1.1, 1.2"
    assert len(s2.results) == 24

    p.input_wanted = "rho"
    assert len(s2.results) == 2


def test_nozzles_page_ui_controls():
    p = NozzlesPage()
    sidebar_controls = p.controls
    assert isinstance(sidebar_controls, pn.Column)
    assert len(sidebar_controls.objects) == 3
    assert isinstance(sidebar_controls.objects[0], pn.widgets.Select)
    assert isinstance(sidebar_controls.objects[1], pn.pane.Markdown)
    assert isinstance(sidebar_controls.objects[2], pn.widgets.IntInput)
