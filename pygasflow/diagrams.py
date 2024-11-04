from pygasflow.solvers import (
    isentropic_solver, fanno_solver, rayleigh_solver,
    normal_shockwave_solver
)

from pygasflow.shockwave import (
    theta_from_mach_beta,
    beta_from_mach_max_theta,
    beta_theta_max_for_unit_mach_downstream,
    mach_cone_angle_from_shock_angle,
    max_theta_c_from_mach,
    beta_theta_c_for_unit_mach_downstream,
    load_data
)
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import (
    Legend, HoverTool, Range1d, LinearAxis, VeeHead, Arrow,
    LabelSet, ColumnDataSource
)
from bokeh.palettes import Category10
import itertools


# TODO: Ideally, I would love to use holoviews or hvplot, giving the user
# the abitiy to chose the backend (Matplotlib, Bokeh, Plotly).
# However, using holoviews/hvplot I'm unable to create a proper figure with
# multiple y-axis (necessary for the isentropic diagram).
# So, for now I'll use Bokeh.

tooltips=[("Variable", "@v"), ("Mach", "@xs"), ("value", "@ys")]


def _create_figure(**fig_kwargs):
    fig_kwargs.setdefault("x_axis_label", "M")
    fig_kwargs.setdefault("y_axis_label", "Ratios")
    fig_kwargs.setdefault("y_range", (0, 3))
    fig_kwargs.setdefault("height", 300)
    fig_kwargs.setdefault("width", 800)
    # fig_kwargs.setdefault("sizing_mode", "stretch_width")
    return figure(**fig_kwargs)


def _place_legend_outside(fig):
    # hide original legend and create a new one outside of the plot area
    fig.legend.visible = False
    legend_items = fig.legend.items
    legend = Legend(items=legend_items)
    # interactive legend
    legend.click_policy = "hide"
    fig.add_layout(legend, "right")


def isentropic(
    min_mach=1e-05, max_mach=5, gamma=1.4, n=100,
    show_title=False, **fig_kwargs
):
    """
    Parameters
    ----------
    min_mach : float
        Minimum upstream Mach number.
    max_mach : float
        Maximum upstream Mach number.
    gamma : float
        Ratio of specific heats.
    n : int
        Number of points for each curve.
    show_title : bool
        Wheter to show the title of the plot.
    """
    M = np.linspace(min_mach, max_mach, n)
    results = isentropic_solver("m", M, gamma)
    labels = [
        "M", "P/P0", "rho/rho0", "T/T0",
        "p/p*", "rho/rho*", "T/T*", "U/U*", "A/A*",
        "Mach Angle", "Prandtl-Meyer Angle"
    ]

    if show_title:
        fig_kwargs.setdefault("title", "Isentropic Flow")

    fig = _create_figure(**fig_kwargs)
    colors = itertools.cycle(Category10[10])

    for l, r in zip(labels[1:-2], results[1:-2]):
        source = {"xs": results[0], "ys": r, "v": [l] * len(r)}
        line = fig.line(
            "xs", "ys",
            source=source,
            legend_label=l,
            line_width=2,
            line_color=next(colors)
        )
        fig.add_tools(HoverTool(
            tooltips=tooltips,
            renderers=[line]
        ))

    fig.extra_y_ranges['deg'] = Range1d(0, 90)
    for l, r in zip(labels[-2:], results[-2:]):
        source = {"xs": results[0], "ys": r, "v": [l] * len(r)}
        line = fig.line("xs", "ys", source=source, legend_label=l, line_width=2, line_color=next(colors), line_dash="dashed",
            y_range_name="deg",
        )
        fig.add_tools(HoverTool(
            tooltips=tooltips,
            renderers=[line]
        ))

    # create new y-axis
    y_deg = LinearAxis(
        axis_label="Angles [deg]",
        y_range_name="deg",
    )
    fig.add_layout(y_deg, 'right')

    _place_legend_outside(fig)
    return fig


def fanno(
    min_mach=1e-05, max_mach=5, gamma=1.4, n=100,
    show_title=False, **fig_kwargs
):
    """
    Parameters
    ----------
    min_mach : float
        Minimum upstream Mach number.
    max_mach : float
        Maximum upstream Mach number.
    gamma : float
        Ratio of specific heats.
    n : int
        Number of points for each curve.
    show_title : bool
        Wheter to show the title of the plot.
    """
    M = np.linspace(min_mach, max_mach, n)
    results = fanno_solver("m", M, gamma)
    labels = [
        "M", "p/p*", "rho/rho*", "T/T*",
        "P0/P0*", "U/U*", "4fL*/D", "(s*-s)/R"
    ]

    if show_title:
        fig_kwargs.setdefault("title", "Fanno Flow")

    fig = _create_figure(**fig_kwargs)
    colors = itertools.cycle(Category10[10])

    for l, r in zip(labels[1:], results[1:]):
        source = {"xs": results[0], "ys": r, "v": [l] * len(r)}
        line = fig.line(
            "xs", "ys",
            source=source,
            legend_label=l,
            line_width=2,
            line_color=next(colors)
        )
        fig.add_tools(HoverTool(
            tooltips=tooltips,
            renderers=[line]
        ))

    _place_legend_outside(fig)
    return fig


def rayleigh(
    min_mach=1e-05, max_mach=5, gamma=1.4, n=100,
    show_title=False, **fig_kwargs
):
    """
    Parameters
    ----------
    min_mach : float
        Minimum upstream Mach number.
    max_mach : float
        Maximum upstream Mach number.
    gamma : float
        Ratio of specific heats.
    n : int
        Number of points for each curve.
    show_title : bool
        Wheter to show the title of the plot.
    """
    M = np.linspace(min_mach, max_mach, n)
    results = rayleigh_solver("m", M, gamma)
    if np.isclose(min_mach, 1e-05):
        # rho/rho* will be so huge (on the order of 1e09) that Bokeh is
        # unable to plot it
        results[2][0] = np.nan
    labels = [
        "M", "p/p*", "rho/rho*", "T/T*",
        "P0/P0*", "T0/T0*", "U/U*", "(s*-s)/R"
    ]

    if show_title:
        fig_kwargs.setdefault("title", "Rayleigh Flow")

    fig = _create_figure(**fig_kwargs)
    colors = itertools.cycle(Category10[10])

    for l, r in zip(labels[1:], results[1:]):
        source = {"xs": results[0], "ys": r, "v": [l] * len(r)}
        line = fig.line(
            "xs", "ys",
            source=source,
            legend_label=l,
            line_width=2,
            line_color=next(colors)
        )
        fig.add_tools(HoverTool(
            tooltips=tooltips,
            renderers=[line]
        ))

    _place_legend_outside(fig)
    return fig


def normal_shockwave(
    min_mach=1, max_mach=8, gamma=1.4, n=100,
    dividers=[1, 100, 10, 10, 1],
    show_title=False, **fig_kwargs
):
    """
    Parameters
    ----------
    min_mach : float
        Minimum upstream Mach number.
    max_mach : float
        Maximum upstream Mach number.
    gamma : float
        Ratio of specific heats.
    n : int
        Number of points for each curve.
    show_title : bool
        Wheter to show the title of the plot.
    dividers : list
        Some ratio are going to be much bigger than others at the same
        upstream Mach number. Each number of this list represents a
        quotient for a particular ratio in order to "normalize" the
        visualization.
    """
    M = np.linspace(min_mach, max_mach, n)
    results = normal_shockwave_solver("m1", M, gamma=gamma)
    labels = [
        "M1", "M2", "p2/p1", "rho2/rho1", "T2/T1", "P02/P01*"
    ]

    if show_title:
        fig_kwargs.setdefault("title", "Normal Shock Properties")
    fig_kwargs.setdefault("x_axis_label", "Upstream Mach, M1")
    fig_kwargs.setdefault("x_range", (min_mach, max_mach))
    fig_kwargs.setdefault("y_range", (0, 1.5))

    fig = _create_figure(**fig_kwargs)
    colors = itertools.cycle(Category10[10])

    for i, (l, r) in enumerate(zip(labels[1:], results[1:])):
        current_label = l
        if not np.isclose(dividers[i], 1):
            current_label = "(%s) / %s" % (current_label, dividers[i])
        source = {
            "xs": results[0],
            "ys": r / dividers[i],
            "v": [current_label] * len(r)
        }
        line = fig.line(
            "xs", "ys",
            source=source,
            legend_label=current_label,
            line_width=2,
            line_color=next(colors)
        )
        fig.add_tools(HoverTool(
            tooltips=tooltips,
            renderers=[line]
        ))

    _place_legend_outside(fig)
    return fig


def oblique_shock_wave(
    M=[1.1, 1.5, 2, 3, 5, 10, 1e9],
    gamma=1.4,
    N=100,
    show_title=False,
    show_sonic_line=True,
    show_region_line=True,
    **fig_kwargs
):
    """
    Parameters
    ----------
    M : list
        List of upstream Mach numbers.
    gamma : float
        Ratio of specific heats.
    N : int
        Number of points for each Mach curve.
    show_title : bool
        Wheter to show the title of the plot.
    show_sonic_line : bool
        Show the line where the downstream Mach number is M2=1.
    show_region_line : bool
        Show a separation line between "strong" solution and "weak" solution.
    """

    fig_kwargs.setdefault("x_axis_label", "Flow Deflection Angle, θ [deg]")
    fig_kwargs.setdefault("y_axis_label", "Shock Wave Angle, β [deg]")
    fig_kwargs.setdefault("x_range", (0, 50))
    fig_kwargs.setdefault("y_range", (0, 90))
    fig_kwargs.setdefault("width", 700)
    fig_kwargs.setdefault("height", 400)
    if show_title:
        fig_kwargs.setdefault("title", "Oblique Shock Properties: Mach - β - θ")

    # labels
    lbls = ["M1 = " + str(M[i]) for  i in range(len(M))]
    lbls[-1] = "M1 = ∞"

    fig = _create_figure(**fig_kwargs)
    colors = itertools.cycle(Category10[10])
    tooltips=[("Variable", "@v"), ("θ", "@xs"), ("β", "@ys")]

    ############################### PART 1 ###############################

    # plot the Mach curves
    for i, m in enumerate(M):
        beta_min = np.rad2deg(np.arcsin(1 / m))
        betas = np.linspace(beta_min, 90, N)
        thetas = theta_from_mach_beta(m, betas, gamma)
        source = {"xs": thetas, "ys": betas, "v": [lbls[i]] * len(betas)}
        line = fig.line(
            "xs", "ys",
            source=source,
            line_color=next(colors),
            line_width=2,
            legend_label=lbls[i]
        )
        fig.add_tools(HoverTool(
            tooltips=tooltips,
            renderers=[line]
        ))

    ############################### PART 2 ###############################

    vh = VeeHead(size=6, fill_color="#000000")
    offset_1 = 1
    offset_2 = 8
    offset_3 = 11

    sonic_annotations_x = [np.nan, np.nan]
    sonic_annotations_y = [np.nan, np.nan]
    sonic_annotations = ["", ""]

    # compute the line M2 = 1
    M1 = np.logspace(0, 3, 5 * N)
    beta_M2_equal_1, theta_max = beta_theta_max_for_unit_mach_downstream(M1, gamma)

    if show_sonic_line:
        source = {"xs": theta_max, "ys": beta_M2_equal_1, "v": [""] * len(M1)}
        fig.line(
            "xs", "ys",
            source=source,
            line_dash="dotted",
            color="#000000",
            line_width=1
        )

        # select an index where to put the annotation (chosen by trial and error)
        i1 = 20
        sonic_annotations_x = [theta_max[i1], theta_max[i1]]
        sonic_annotations_y = [
            beta_M2_equal_1[i1] + offset_3, beta_M2_equal_1[i1] - offset_3]
        sonic_annotations = ["M2 < 1", "M2 > 1"]

        fig.add_layout(
            Arrow(
                end=vh,
                x_start=theta_max[i1], y_start=beta_M2_equal_1[i1] + offset_1,
                x_end=theta_max[i1], y_end=beta_M2_equal_1[i1] + offset_2
            )
        )
        fig.add_layout(
            Arrow(
                end=vh,
                x_start=theta_max[i1], y_start=beta_M2_equal_1[i1] - offset_1,
                x_end=theta_max[i1], y_end=beta_M2_equal_1[i1] - offset_2
            )
        )

    ############################### PART 3 ###############################

    region_annotations_x = [np.nan, np.nan]
    region_annotations_y = [np.nan, np.nan]
    region_annotations = ["", ""]

    if show_region_line:
        # compute the line passing through (M,theta_max)
        beta = beta_from_mach_max_theta(M1, gamma)

        source = {"xs": theta_max, "ys": beta, "v": [""] * len(M1)}
        fig.line(
            "xs", "ys",
            source=source,
            line_dash="dashed",
            color="#000000",
            line_width=1
        )

        # # select an index where to put the annotation (chosen by trial and error)
        i2 = 50
        region_annotations_x = [theta_max[i2], theta_max[i2]]
        region_annotations_y = [beta[i2] + offset_3, beta[i2] - offset_3]
        region_annotations = ["strong", "weak"]

        fig.add_layout(
            Arrow(
                end=vh,
                x_start=theta_max[i2], y_start=beta[i2] + offset_1,
                x_end=theta_max[i2], y_end=beta[i2] + offset_2
            )
        )
        fig.add_layout(
            Arrow(
                end=vh,
                x_start=theta_max[i2], y_start=beta[i2] - offset_1,
                x_end=theta_max[i2], y_end=beta[i2] - offset_2
            )
        )

    source = ColumnDataSource(data={
        "x": sonic_annotations_x + region_annotations_x,
        "y": sonic_annotations_y + region_annotations_y,
        "labels": sonic_annotations + region_annotations
    })
    fig.add_layout(
        LabelSet(
            x="x", y="y", text="labels",
            x_offset="x_offset", y_offset="y_offset", source=source,
            text_baseline="middle", text_align="center",
            text_color="#000000", text_font_size="12px"
        )
    )

    _place_legend_outside(fig)
    return fig


def conical_shock_wave(
    M=[1.05, 1.2, 1.5, 2, 5, 10000],
    gamma=1.4,
    N=200,
    show_title=False,
    show_sonic_line=True,
    show_region_line=True,
    **fig_kwargs
):
    """
    Parameters
    ----------
    M : list
        List of upstream Mach numbers.
    gamma : float
        Ratio of specific heats.
    N : int
        Number of points for each Mach curve.
    show_title : bool
        Wheter to show the title of the plot.
    show_sonic_line : bool
        Show the line where the downstream Mach number is M2=1.
    show_region_line : bool
        Show a separation line between "strong" solution and "weak" solution.
    """

    fig_kwargs.setdefault("x_axis_label", "Half cone angle, $$θ_{c}$$ [deg]")
    fig_kwargs.setdefault("y_axis_label", "Shock Wave Angle, β [deg]")
    fig_kwargs.setdefault("x_range", (0, 60))
    fig_kwargs.setdefault("y_range", (0, 90))
    fig_kwargs.setdefault("width", 700)
    fig_kwargs.setdefault("height", 400)
    if show_title:
        fig_kwargs.setdefault("title", "Conical Shock Properties: Mach - $$β$$ - $$θ_{c}$$")

    # labels
    lbls = ["M1 = " + str(M[i]) for  i in range(len(M))]
    lbls[-1] = "M1 = ∞"

    fig = _create_figure(**fig_kwargs)
    colors = itertools.cycle(Category10[10])
    tooltips=[("Variable", "@v"), ("θ", "@xs"), ("β", "@ys")]

    ############################### PART 1 ###############################

    max_theta_c = 0
    for j, M1 in enumerate(M):
        theta_c = np.zeros(N)
        # NOTE: to avoid errors in the integration process of Taylor-Maccoll
        # equation, beta should be different than Mach angle and 90deg,
        # hence an offset is applied.
        offset = 1e-08
        theta_s = np.linspace(np.rad2deg(np.arcsin(1 / M1)) + offset, 90 - offset, N)
        for i, ts in enumerate(theta_s):
            Mc, tc = mach_cone_angle_from_shock_angle(M1, ts, gamma)
            theta_c[i] = tc
        theta_c = np.insert(theta_c, 0, 0)
        theta_c = np.append(theta_c, 0)
        theta_s = np.insert(theta_s, 0, np.rad2deg(np.arcsin(1 / M1)))
        theta_s = np.append(theta_s, 90)
        max_theta_c = max(max_theta_c, theta_c.max())

        source = {"xs": theta_c, "ys": theta_s, "v": [lbls[j]] * len(theta_s)}
        line = fig.line(
            "xs", "ys",
            source=source,
            line_color=next(colors),
            line_width=2,
            legend_label=lbls[j]
        )
        fig.add_tools(HoverTool(
            tooltips=tooltips,
            renderers=[line]
        ))

    # adjust x_range
    max_theta_c = round(max_theta_c + (max_theta_c % 5))
    fig.x_range = Range1d(0, max_theta_c)

    ############################### PART 2 ###############################

    vh = VeeHead(size=6, fill_color="#000000")
    offset_1 = 1
    offset_2 = 8
    offset_3 = 11
    region_annotations_x = [np.nan, np.nan]
    region_annotations_y = [np.nan, np.nan]
    region_annotations = ["", ""]

    if show_region_line:
        # Compute the line passing through theta_c_max
        M = np.asarray([1.0005, 1.0025, 1.005, 1.025, 1.05, 1.07, 1.09,
                        1.12, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5,
                        1.6, 1.75, 2, 2.25, 3, 4, 5, 10, 100, 10000])
        b = np.zeros_like(M)
        tc = np.zeros_like(M)
        for i, m in enumerate(M):
            _, tc[i], b[i] = max_theta_c_from_mach(m, gamma)
        tc = np.insert(tc, 0, 0)
        b = np.insert(b, 0, 90)
        fig.line(
            tc, b,
            line_color="#000000",
            line_width=1,
            line_dash="dashed"
        )

        # select an index where to put the annotation
        # (chosen by trial and error)
        i1 = 16
        region_annotations_x = [tc[i1], tc[i1]]
        region_annotations_y = [b[i1] + offset_3, b[i1] - offset_3]
        region_annotations = ["strong", "weak"]

        fig.add_layout(
            Arrow(
                end=vh,
                x_start=tc[i1], y_start=b[i1] + offset_1,
                x_end=tc[i1], y_end=b[i1] + offset_2
            )
        )
        fig.add_layout(
            Arrow(
                end=vh,
                x_start=tc[i1], y_start=b[i1] - offset_1,
                x_end=tc[i1], y_end=b[i1] - offset_2
            )
        )

    ############################### PART 3 ###############################

    sonic_annotations_x = [np.nan, np.nan]
    sonic_annotations_y = [np.nan, np.nan]
    sonic_annotations = ["", ""]

    if show_sonic_line:
        try:
            M, beta, theta_c = load_data(gamma)
            i2 = 54
        except FileNotFoundError:
            # If there is no data for M2=1, we just need to generate it.
            # IT IS SLOW!!!
            M = np.asarray([
                1, 1.005, 1.05, 1.2, 1.3, 1.4, 1.5,
                1.65, 1.8, 2, 3, 4, 5, 10, 10000])
            beta = np.zeros_like(M)
            theta_c = np.zeros_like(M)
            for i, m in enumerate(M):
                try:
                    beta[i], theta_c[i] = beta_theta_c_for_unit_mach_downstream(m, gamma)
                except ValueError:
                    beta[i], theta_c[i] = np.nan, np.nan
            i2 = int(len(beta) / 4)

        sonic_annotations_x = [theta_c[i2], theta_c[i2]]
        sonic_annotations_y = [beta[i2] + offset_3, beta[i2] - offset_3]
        sonic_annotations = ["M2 < 1", "M2 > 1"]

        fig.line(
            np.asarray(theta_c), np.asarray(beta),
            line_color="#000000",
            line_width=1,
            line_dash="dotted"
        )

        fig.add_layout(
            Arrow(
                end=vh,
                x_start=theta_c[i2], y_start=beta[i2] + offset_1,
                x_end=theta_c[i2], y_end=beta[i2] + offset_2
            )
        )
        fig.add_layout(
            Arrow(
                end=vh,
                x_start=theta_c[i2], y_start=beta[i2] - offset_1,
                x_end=theta_c[i2], y_end=beta[i2] - offset_2
            )
        )

    source = ColumnDataSource(data={
        "x": region_annotations_x + sonic_annotations_x,
        "y": region_annotations_y + sonic_annotations_y,
        "labels": region_annotations + sonic_annotations
    })
    fig.add_layout(
        LabelSet(
            x="x", y="y", text="labels",
            x_offset="x_offset", y_offset="y_offset", source=source,
            text_baseline="middle", text_align="center",
            text_color="#000000", text_font_size="12px"
        )
    )

    _place_legend_outside(fig)
    return fig
