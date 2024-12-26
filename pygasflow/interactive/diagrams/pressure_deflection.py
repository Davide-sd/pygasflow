from bokeh.models import (
    ColumnDataSource, Arrow, VeeHead, Label, CustomJS, Circle
)
import itertools
import numpy as np
import panel as pn
from pygasflow.interactive.diagrams.flow_base import BasePlot
from pygasflow.shockwave import PressureDeflectionLocus


def _compute_arrows_position(x, y, num_arrows=1, dir=1):
    """
    Compute the position of arrows along the coordinates x, y at
    selected locations.

    Parameters:
    -----------
    x: x-coordinate
    y: y-coordinate
    num_arrows: number of arrows to be added
    dir: direction of the arrows. +1 along the line, -1 in opposite direction.

    Returns:
    --------
    source: a dictionary containing the data for bokeh.models.Arrow
    """

    # Compute the arc length along the curve
    s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    # remove duplicates
    idx = ~np.isclose(s - np.roll(s, 1), 0)
    s = s[idx]

    arrow_locs = np.linspace(0, 1, num_arrows + 2)[1:-1]
    x_start, y_start, x_end, y_end = [], [], [], []
    for loc in arrow_locs:
        n = np.searchsorted(s, s[-1] * loc)

        # Figure out what direction to paint the arrow
        if dir == 1:
            arrow_tail = (x[n], y[n])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        elif dir == -1:
            # Orient the arrow in the other direction on the segment
            arrow_tail = (x[n + 1], y[n + 1])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        else:
            raise ValueError("unknown value for keyword 'dir'")

        x_start.append(arrow_tail[0])
        y_start.append(arrow_tail[1])
        x_end.append(arrow_head[0])
        y_end.append(arrow_head[1])

    source = {
        "x_start": x_start, "x_end": x_end, "y_start": y_start, "y_end": y_end
    }
    return source


class PressureDeflectionDiagram(BasePlot, pn.viewable.Viewer):
    """Creates a pressure-deflection diagram.

    Examples
    --------

    .. panel-screenshot::
        :large-size: 700,450

        from pygasflow.shockwave import PressureDeflectionLocus
        from pygasflow.interactive import PressureDeflectionDiagram

        M1 = 3
        theta_2 = 20
        theta_3 = -15

        loc1 = PressureDeflectionLocus(M=M1, label="1")
        loc2 = loc1.new_locus_from_shockwave(theta_2, label="2")
        loc3 = loc1.new_locus_from_shockwave(theta_3, label="3")

        phi, p4_p1 = loc2.intersection(loc3)
        print("Intersection between locus M2 and locus M3 happens at:")
        print("Deflection Angle [deg]:", phi)
        print("Pressure ratio to freestream:", p4_p1)

        d = PressureDeflectionDiagram(
            title="Intersection of shocks of opposite families"
        )
        d.add_locus(loc1)
        d.add_locus(loc2)
        d.add_locus(loc3)
        d.add_path((loc1, theta_2), (loc2, phi))
        d.add_path((loc1, theta_3), (loc3, phi))
        d.add_state(
            phi, p4_p1, "4=4'",
            background_fill_color="white",
            background_fill_alpha=0.8)
        d.move_legend_outside()
        d.y_range = (0, 18)
        d.show_figure()

    """
    # TODO:
    # 1. add tooltips to states
    # 2. create some class that renders the locus, so that we can monitor
    #    the attributes of the locus, and update it when they change.
    #    This would allow interactive pressure-deflection diagrams.

    def __init__(self, **params):
        params.setdefault("x_label", "Deflection angle, θ [deg]")
        params.setdefault("y_label", "Pressure Ratio to Freestream")
        params.setdefault("size", (600, 400))
        params.setdefault("show_minor_grid", True)
        super().__init__(**params)
        self.color_iterator = itertools.cycle(self.colors)

    def update(self):
        pass

    def add_state(
        self, theta, pr, text, primary_line=None, circle_kw=None, **kwargs
    ):
        """Add `text` to the specified point `(x, y)` in the diagram.

        Parameters
        ----------
        theta : float
            Flow deflection angle in degrees.
        pr : float
            Pressure ratio to freestream.
        text : str
        primary_line : None or Renderer
            If a Bokeh renderer is provided, its visibility will be linked to
            the label's visibility. Useful to hide the label when the
            visibility of ``primary_line`` is toggled on the legend.
        circle_kw : None or dict
            Keyword arguments to :class:`bokeh.models.Circle`.
        **kwargs :
            Keyword arguments passed to :class:`bokeh.models.Label`.

        Returns
        -------
        lbl : :class:`bokeh.models.Label`
        circle_rend : :class:`bokeh.models.GlyphRenderer`
        """
        kwargs.setdefault("x_offset", 0)
        kwargs.setdefault("y_offset", 15)
        kwargs.setdefault("text_align", "center")
        kwargs.setdefault("text_baseline", "middle")
        lbl = Label(x=theta, y=pr, text=text, **kwargs)
        circle_kw = {} if circle_kw is None else circle_kw
        circle_kw.setdefault("radius", 4)
        circle_kw.setdefault("radius_units", "screen")
        circle_kw.setdefault("line_color", "#000000")
        circle_kw.setdefault("fill_color", "#000000")
        self.figure.add_layout(lbl)
        circle_rend = self.figure.add_glyph(Circle(x=theta, y=pr, **circle_kw))

        if primary_line is not None:
            # bind all handles' visibility to the primary_line's visibility
            callback = CustomJS(
                args=dict(
                    primary_line=primary_line, label=lbl, circle_rend=circle_rend
                ),
                code="""
                label.visible = primary_line.visible;
                circle_rend.visible = primary_line.visible;
                """
            )
            primary_line.js_on_change("visible", callback)
        return lbl, circle_rend

    def add_locus(self, locus, show_state=True, N=100, include_mirror=True, **kwargs):
        """Add the locus to the diagram, with a single line.

        Parameters
        ----------
        locus : :class:`~pygasflow.shockwave.PressureDeflectionLocus`
        show_state : bool
            If True, also add a text on the diagram at the start of the locus.
        N : int
            Number of discretization points.
        include_mirror : bool
            If False, only plot the locus for theta >= 0.
        **kwargs :
            Keyword arguments passed to :class:`bokeh.models.Line`

        Returns
        -------
        line_rend : :class:`bokeh.models.GlyphRenderer`
        lbl : :class:`bokeh.models.Label` or None
        circle_rend : :class:`bokeh.models.GlyphRenderer` or None
        """
        if locus.label:
            kwargs.setdefault("legend_label", f"M{locus.label}")
        kwargs.setdefault("line_dash", "dotted")
        kwargs.setdefault("line_width", 2)
        kwargs.setdefault("line_color", next(self.color_iterator))
        theta, pr = locus.pressure_deflection(N=N, include_mirror=include_mirror)
        source = ColumnDataSource({"x": theta, "y": pr})
        line_rend = self.figure.line("x", "y", source=source, **kwargs)
        lbl, circle_rend = None, None
        if show_state and locus.label:
            lbl, circle_rend = self.add_state(
                locus.theta_origin,
                locus.pr_to_fs_at_origin,
                locus.label,
                primary_line=line_rend
            )
        return line_rend, lbl, circle_rend

    def add_locus_split(
        self, locus, weak_kwargs={}, strong_kwargs={}, same_color=True,
        show_state=True, mode="region", N=100, include_mirror=True
    ):
        """Add the locus to the diagram, with two lines, one for the weak
        region and the other for the strong region.

        Parameters
        ----------
        locus : :class:`~pygasflow.shockwave.PressureDeflectionLocus`
        show_state : bool
            If True, also add a text on the diagram at the start of the locus.
        weak_kwargs : dict
            Keyword arguments passed to :class:`bokeh.models.Line` in order to
            customize the line for the weak region.
        strong_kwargs : dict
            Keyword arguments passed to :class:`bokeh.models.Line` in order to
            customize the line for the strong region.
        same_color : bool
            Wheter the two lines uses the same color.
        show_state : bool
            If True, also add a text on the diagram at the start of the locus.
        mode : str
            Split the locus at some point. It can be:

            * ``"region"``: the locus is splitted at the detachment point,
              where ``theta=theta_max``.
            * ``"sonic"``: the locus is splitted at the sonic point (where
              the downstream Mach number is 1).
        N : int
            Number of discretization points.
        include_mirror : bool
            If False, only plot the locus for theta >= 0.

        Returns
        -------
        line_rend_weak : :class:`bokeh.models.GlyphRenderer`
        line_rend_strong : :class:`bokeh.models.GlyphRenderer`
        lbl : :class:`bokeh.models.Label` or None
        circle_rend : :class:`bokeh.models.GlyphRenderer` or None
        """
        weak_kwargs = weak_kwargs.copy()
        strong_kwargs = strong_kwargs.copy()
        if locus.label:
            weak_kwargs.setdefault("legend_label", f"M{locus.label} weak")
            strong_kwargs.setdefault("legend_label", f"M{locus.label} strong")
        weak_kwargs.setdefault("line_dash", "dotted")
        strong_kwargs.setdefault("line_dash", "dashed")
        weak_kwargs.setdefault("line_width", 2)
        strong_kwargs.setdefault("line_width", 2)
        color = next(self.color_iterator)
        weak_kwargs.setdefault("line_color", color)
        strong_kwargs.setdefault(
            "line_color", color if same_color else next(self.color_iterator))

        theta_w, pr_w, theta_s, pr_s = locus.pressure_deflection_split(
            mode=mode, N=N, include_mirror=include_mirror)
        source_w = ColumnDataSource({"x": theta_w, "y": pr_w})
        source_s = ColumnDataSource({"x": theta_s, "y": pr_s})
        line_rend_weak = self.figure.line(
            "x", "y", source=source_w, **weak_kwargs)
        line_rend_strong = self.figure.line(
            "x", "y", source=source_s, **strong_kwargs)
        lbl, circle_rend = None, None
        if show_state and locus.label:
            lbl, circle_rend = self.add_state(
                locus.theta_origin,
                locus.pr_to_fs_at_origin,
                locus.label,
                primary_line=line_rend_weak
            )
        return line_rend_weak, line_rend_strong, lbl, circle_rend

    def add_path(self, *segments, num_arrows=2, **kwargs):
        """Add a path connecting one or more segments.

        Parameters
        ----------
        segments : tuples
            Each segment is a 2-elements tuple where the first element is a
            PressureDeflectionLocus, and the second element is the deflection
            angle [degrees] of the end of the segment.
        num_arrows : int
            Number of arrows to add over the each segment.
        **kwargs :
            Keyword arguments passed to :class:`bokeh.models.Line`.

        Returns
        -------
        rend : :class:`bokeh.models.GlyphRenderer`
        arrows : :class:`bokeh.models.Arrow`
        """
        theta, pr = PressureDeflectionLocus.create_path(
            *segments, concatenate=False)
        kwargs.setdefault("line_width", 2)
        kwargs.setdefault("line_color", next(self.color_iterator))
        source = ColumnDataSource({
            "x": np.concatenate(theta),
            "y": np.concatenate(pr)
        })
        rend = self.figure.line("x", "y", source=source, **kwargs)

        source_dicts = []
        for t, p in zip(theta, pr):
            source_dicts.append(_compute_arrows_position(
                t, p, num_arrows=num_arrows, dir=1))
        arrow_source = ColumnDataSource(data={
            k: np.concatenate([d[k] for d in source_dicts])
            for k in source_dicts[0]
        })
        arrow_style = VeeHead(
            line_color=kwargs["line_color"],
            fill_color=kwargs["line_color"],
            size=10
        )
        arrows = Arrow(
            source=arrow_source,
            line_color=kwargs["line_color"],
            end=arrow_style
        )
        self.figure.add_layout(arrows)
        return rend, arrows

    def __panel__(self):
        return pn.pane.Bokeh(self.figure)
