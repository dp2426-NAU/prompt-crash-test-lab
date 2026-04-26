"""
Core 3D drawing utilities for isometric workflow diagrams.
Uses matplotlib mplot3d with orthographic projection for a clean 3D isometric look.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Professional colour palette ───────────────────────────────────────────────

PALETTE = {
    "input":      "#4ECDC4",   # teal
    "variants":   "#45B7D1",   # sky-blue
    "models":     "#7EC8E3",   # cornflower
    "evaluation": "#C9A7EB",   # lavender
    "storage":    "#FFD93D",   # gold
    "viz":        "#6BCB77",   # mint-green
    "bg_dark":    "#0D1B2A",   # dark navy
    "bg_mid":     "#1B2A3B",   # mid navy
    "grid":       "#1E3A5F",   # grid lines
    "text_light": "#FFFFFF",
    "text_dark":  "#0D1B2A",
    "accent":     "#FF6B6B",   # coral accent
    "shadow":     "#050D17",   # deep shadow
    "connector":  "#7FB3D3",   # arrow colour
    # Component colours
    "gpt4":       "#74AA9C",   # OpenAI green
    "claude":     "#CC785C",   # Anthropic orange
    "gemini":     "#4285F4",   # Google blue
    "llama":      "#7B68EE",   # Meta purple
    "metric_r":   "#FF6B6B",   # robustness
    "metric_s":   "#4ECDC4",   # semantic
    "metric_f":   "#45B7D1",   # format
    "metric_a":   "#FFD93D",   # answer
    "metric_c":   "#6BCB77",   # citation
    "metric_cost":"#C9A7EB",   # cost
}


# ── Colour helpers ────────────────────────────────────────────────────────────

def _shade(color: str, factor: float) -> tuple:
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _lighten(color: str, factor: float) -> tuple:
    r, g, b = to_rgb(color)
    return (
        r + (1 - r) * factor,
        g + (1 - g) * factor,
        b + (1 - b) * factor,
    )


# ── Figure setup ──────────────────────────────────────────────────────────────

def make_iso_figure(
    figsize: tuple = (14, 14),
    dpi: int = 150,
    bg: str = "#0D1B2A",
    elev: float = 22,
    azim: float = -55,
) -> tuple[plt.Figure, Axes3D]:
    """Create a figure with an isometric-style 3D axis."""
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)
    ax.patch.set_facecolor(bg)

    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")     # true isometric (matplotlib ≥ 3.1)
    except AttributeError:
        pass

    # Remove default axes decorations
    ax.set_axis_off()
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")

    return fig, ax


# ── 3D grid / platform base ───────────────────────────────────────────────────

def draw_platform(
    ax: Axes3D,
    x0: float, y0: float,
    width: float, depth: float,
    z: float = -0.2,
    color: str = "#1E3A5F",
    alpha: float = 0.4,
) -> None:
    """Draw a subtle grid platform beneath the diagram."""
    # Platform surface
    xs = [x0, x0 + width, x0 + width, x0]
    ys = [y0, y0, y0 + depth, y0 + depth]
    zs = [z, z, z, z]
    verts = [list(zip(xs, ys, zs))]
    poly = Poly3DCollection(verts, alpha=alpha)
    poly.set_facecolor(color)
    poly.set_edgecolor("#2A5298")
    poly.set_linewidth(0.5)
    ax.add_collection3d(poly)

    # Grid lines
    step = 1.0
    lc = "#1A3A6A"
    x = x0
    while x <= x0 + width:
        ax.plot([x, x], [y0, y0 + depth], [z, z], color=lc, lw=0.3, alpha=0.5)
        x += step
    y = y0
    while y <= y0 + depth:
        ax.plot([x0, x0 + width], [y, y], [z, z], color=lc, lw=0.3, alpha=0.5)
        y += step


# ── Core 3D box ───────────────────────────────────────────────────────────────

def draw_box(
    ax: Axes3D,
    pos: tuple,
    size: tuple,
    color: str,
    alpha: float = 0.92,
    edge_color: str = "#FFFFFF",
    edge_width: float = 0.6,
    shadow: bool = True,
) -> np.ndarray:
    """
    Draw a 3D box with face-based lighting simulation.

    Args:
        pos:   (x, y, z) bottom-front-left corner
        size:  (w, d, h) width, depth, height
        color: hex fill colour (top face = full brightness)

    Returns:
        Centre point of the box as np.ndarray([cx, cy, cz]).
    """
    x, y, z = pos
    w, d, h = size

    # Shadow (slightly offset, very dark)
    if shadow:
        sh = 0.15
        sv = -0.08
        s_verts = [
            [x + sh,     y + sh,     z + sv],
            [x + w + sh, y + sh,     z + sv],
            [x + w + sh, y + d + sh, z + sv],
            [x + sh,     y + d + sh, z + sv],
        ]
        sp = Poly3DCollection([s_verts], alpha=0.35)
        sp.set_facecolor(PALETTE["shadow"])
        sp.set_edgecolor("none")
        ax.add_collection3d(sp)

    v = np.array([
        [x,   y,   z  ],   # 0
        [x+w, y,   z  ],   # 1
        [x+w, y+d, z  ],   # 2
        [x,   y+d, z  ],   # 3
        [x,   y,   z+h],   # 4
        [x+w, y,   z+h],   # 5
        [x+w, y+d, z+h],   # 6
        [x,   y+d, z+h],   # 7
    ])

    # Faces: (vertices, brightness-factor)
    faces = [
        ([v[4], v[5], v[6], v[7]], 1.00),   # top
        ([v[0], v[1], v[5], v[4]], 0.78),   # front-y
        ([v[1], v[2], v[6], v[5]], 0.62),   # right-x  (most shadow)
        ([v[0], v[3], v[7], v[4]], 0.72),   # left-x
        ([v[3], v[2], v[6], v[7]], 0.68),   # back-y
        ([v[0], v[1], v[2], v[3]], 0.45),   # bottom
    ]

    for verts, bright in faces:
        fc = _shade(color, bright)
        poly = Poly3DCollection([verts], alpha=alpha)
        poly.set_facecolor(fc)
        poly.set_edgecolor(edge_color)
        poly.set_linewidth(edge_width)
        ax.add_collection3d(poly)

    return np.array([x + w / 2, y + d / 2, z + h / 2])


# ── Glowing top cap ───────────────────────────────────────────────────────────

def draw_glow_cap(
    ax: Axes3D,
    pos: tuple,
    size: tuple,
    color: str,
    alpha: float = 0.25,
    layers: int = 3,
) -> None:
    """Draw faint expanded rectangles above the box top to simulate glow."""
    x, y, z = pos
    w, d, h = size
    top_z = z + h
    for i in range(1, layers + 1):
        expand = i * 0.06
        gverts = [
            [x - expand,     y - expand,     top_z + i * 0.01],
            [x + w + expand, y - expand,     top_z + i * 0.01],
            [x + w + expand, y + d + expand, top_z + i * 0.01],
            [x - expand,     y + d + expand, top_z + i * 0.01],
        ]
        gp = Poly3DCollection([gverts], alpha=alpha / i)
        gp.set_facecolor(color)
        gp.set_edgecolor("none")
        ax.add_collection3d(gp)


# ── Label rendering ───────────────────────────────────────────────────────────

def label_box(
    ax: Axes3D,
    center: np.ndarray,
    title: str,
    subtitle: str = "",
    icon: str = "",
    text_color: str = "#FFFFFF",
    title_size: float = 9,
    sub_size: float = 7,
    icon_size: float = 14,
) -> None:
    """Place a title (+ optional icon + subtitle) on top of a box."""
    cx, cy, cz = center

    # Icon
    if icon:
        ax.text(
            cx, cy, cz + 0.28,
            icon,
            ha="center", va="bottom",
            fontsize=icon_size,
            color=text_color,
            zorder=10,
        )

    # Title
    ax.text(
        cx, cy, cz + 0.08,
        title,
        ha="center", va="center",
        fontsize=title_size,
        fontweight="bold",
        color=text_color,
        zorder=10,
    )

    # Subtitle
    if subtitle:
        ax.text(
            cx, cy, cz - 0.12,
            subtitle,
            ha="center", va="center",
            fontsize=sub_size,
            color=text_color,
            alpha=0.80,
            zorder=10,
        )


# ── 3D connector arrow ────────────────────────────────────────────────────────

def draw_arrow(
    ax: Axes3D,
    start: np.ndarray,
    end: np.ndarray,
    color: str = "#7FB3D3",
    lw: float = 1.8,
    label: str = "",
    label_size: float = 7.5,
) -> None:
    """Draw a line connector with an arrowhead approximated by a cone tip."""
    xs = [start[0], end[0]]
    ys = [start[1], end[1]]
    zs = [start[2], end[2]]

    ax.plot(xs, ys, zs, color=color, lw=lw, alpha=0.85, zorder=5)

    # Arrowhead: small cone drawn as a triangle at the end
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return
    unit = direction / length
    tip_len = 0.18
    tip_base = end - unit * tip_len

    perp = np.cross(unit, [0, 0, 1])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(unit, [0, 1, 0])
    perp = perp / np.linalg.norm(perp) * tip_len * 0.4

    tri = [
        end,
        tip_base + perp,
        tip_base - perp,
    ]
    arrow_poly = Poly3DCollection([tri], alpha=0.9)
    arrow_poly.set_facecolor(color)
    arrow_poly.set_edgecolor(color)
    ax.add_collection3d(arrow_poly)

    # Midpoint label
    if label:
        mid = (start + end) / 2
        ax.text(
            mid[0], mid[1], mid[2] + 0.1,
            label,
            ha="center", va="bottom",
            fontsize=label_size,
            color=color,
            alpha=0.9,
            zorder=10,
        )


# ── Axis limits helper ────────────────────────────────────────────────────────

def set_axes_limits(ax: Axes3D, margin: float = 0.5) -> None:
    """Auto-set equal-aspect axis limits from plotted data."""
    try:
        x_lims = ax.get_xlim3d()
        y_lims = ax.get_ylim3d()
        z_lims = ax.get_zlim3d()
        max_range = max(
            x_lims[1] - x_lims[0],
            y_lims[1] - y_lims[0],
            z_lims[1] - z_lims[0],
        ) / 2
        cx = (x_lims[0] + x_lims[1]) / 2
        cy = (y_lims[0] + y_lims[1]) / 2
        cz = (z_lims[0] + z_lims[1]) / 2
        ax.set_xlim3d(cx - max_range - margin, cx + max_range + margin)
        ax.set_ylim3d(cy - max_range - margin, cy + max_range + margin)
        ax.set_zlim3d(cz - max_range - margin, cz + max_range + margin)
    except Exception:
        pass


# ── Title card ────────────────────────────────────────────────────────────────

def add_title(
    fig: plt.Figure,
    title: str,
    subtitle: str = "",
    color: str = "#FFFFFF",
    bg: str = "#0D1B2A",
) -> None:
    """Add a styled title to the figure."""
    fig.text(
        0.5, 0.97, title,
        ha="center", va="top",
        fontsize=18, fontweight="bold",
        color=color,
        path_effects=[pe.withStroke(linewidth=3, foreground=bg)],
    )
    if subtitle:
        fig.text(
            0.5, 0.94, subtitle,
            ha="center", va="top",
            fontsize=10, color="#7FB3D3",
        )


# ── Legend helper ─────────────────────────────────────────────────────────────

def add_legend(
    fig: plt.Figure,
    items: list[tuple[str, str]],
    bg: str = "#0D1B2A",
    x: float = 0.02,
    y: float = 0.12,
) -> None:
    """Add a colour-coded legend. items = [(color, label), ...]"""
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(facecolor=c, edgecolor="#FFFFFF", linewidth=0.5, label=lbl)
        for c, lbl in items
    ]
    legend = fig.legend(
        handles=patches,
        loc="lower left",
        bbox_to_anchor=(x, y),
        framealpha=0.25,
        facecolor=bg,
        edgecolor="#2A5298",
        labelcolor="#FFFFFF",
        fontsize=8,
        title="Pipeline Stages",
        title_fontsize=9,
    )
    legend.get_title().set_color("#7FB3D3")
