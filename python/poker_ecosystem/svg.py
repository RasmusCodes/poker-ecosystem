"""Small dependency-free SVG charts that read on light and dark backgrounds.

Conventions
-----------
* Text, axes and gridlines use ``currentColor``: an SVG inlined in a page is
  transparent and inherits the page's text colour. When the file is the
  document itself (an ``<img>`` in a markdown report) it paints its own
  rounded surface with matching ink, light or dark from
  ``prefers-color-scheme``, so it stays legible whatever the page behind it.
* Series colours come from a validated 8-slot categorical palette with
  separate light and dark steps. Inline, the dark steps follow the OS scheme
  unless the page sets ``data-theme="light"`` on its root element, and
  ``data-theme="dark"`` forces them. Slots are assigned in a fixed order and
  never cycled; more than eight series raise ``ValueError``. Scatter charts
  also vary the marker shape, because only the first three hues stay
  distinguishable when every pair of groups can touch.
* Marks: 2px lines with round joins, markers of radius 4 with a 2px ring in
  the surface colour, bars at most 24px wide with a 4px rounded data end and
  a 2px gap between neighbours, solid hairline gridlines. Charts with two or
  more series get a legend; every mark carries a ``<title>`` tooltip.
* Fonts: the system UI sans stack, 12px ticks, 13px axis titles, 15px title.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

PALETTE_LIGHT: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PALETTE_DARK: tuple[str, ...] = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767")
SURFACE_LIGHT, SURFACE_DARK = "#fcfcfb", "#1a1a19"
INK_LIGHT, INK_DARK = "#0b0b0b", "#f0efec"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
MARKERS: tuple[str, ...] = ("circle", "square", "triangle", "diamond", "triangle-down", "plus", "cross", "star")


@dataclass(frozen=True)
class Series:
    """One line: x values, y values (``None`` breaks the line), optional band."""

    name: str
    xs: Sequence[float]
    ys: Sequence[float | None]
    lo: Sequence[float | None] | None = None
    hi: Sequence[float | None] | None = None


# --------------------------------------------------------------------------- helpers


def format_number(value: float) -> str:
    """Tick label: full numbers below 10,000, compact (10K, 2.5M) above."""
    v = float(f"{value:.12g}")
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:g}M"
    if a >= 1e4:
        return f"{v / 1e3:g}K"
    if v.is_integer():
        return f"{int(v):,}"
    return f"{v:.4g}"


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]; the step is the 1, 2, 2.5 or 5 x 10^k
    value closest (in ratio) to span / (target - 1)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("tick range must be finite")
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        pad = abs(lo) * 0.5 if lo else 1.0
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / max(target - 1, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1.0, 2.0, 2.5, 5.0, 10.0)), key=lambda s: abs(math.log(s / raw)))
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    count = int(round((end - start) / step))
    return [float(f"{start + i * step:.12g}") for i in range(count + 1)]


def log_ticks(lo: float, hi: float) -> list[float]:
    """Powers of ten in [lo, hi], with 2 and 5 multiples when the span is short."""
    if lo <= 0 or hi <= 0:
        raise ValueError("log axes need positive values")
    e0, e1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    ticks = [10.0 ** e for e in range(e0, e1 + 1) if lo * (1 - 1e-9) <= 10.0 ** e <= hi * (1 + 1e-9)]
    if len(ticks) < 3:
        ticks = sorted(
            m * 10.0 ** e for e in range(e0, e1 + 1) for m in (1, 2, 5) if lo * (1 - 1e-9) <= m * 10.0 ** e <= hi * (1 + 1e-9)
        )
    return [float(f"{t:.12g}") for t in ticks]


def _text_width(text: str, size: float) -> float:
    return len(text) * size * 0.58


def _num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


class _Scale:
    def __init__(self, lo: float, hi: float, a: float, b: float, log: bool = False) -> None:
        if log and (lo <= 0 or hi <= 0):
            raise ValueError("log scale needs positive bounds")
        self.lo, self.hi, self.a, self.b, self.log = lo, hi, a, b, log
        self._l0 = math.log10(lo) if log else lo
        self._l1 = math.log10(hi) if log else hi
        if self._l1 == self._l0:
            self._l1 = self._l0 + 1.0

    def __call__(self, v: float) -> float:
        t = ((math.log10(v) if self.log else v) - self._l0) / (self._l1 - self._l0)
        return self.a + t * (self.b - self.a)


def _dark_rules(slots: int, scopes: Sequence[str]) -> str:
    """Dark-step overrides for every selector scope in ``scopes``."""

    def rule(suffix: str, prop: str, value: str) -> str:
        return ",".join(f"{scope} {suffix}" for scope in scopes) + f"{{{prop}:{value}}}"

    parts = [rule(".pe-ring", "stroke", SURFACE_DARK)]
    for i, colour in enumerate(PALETTE_DARK[:slots]):
        parts.append(rule(f".pe-l{i}", "stroke", colour))
        parts.append(rule(f".pe-f{i}", "fill", colour))
    return "".join(parts)


def _style(slots: int) -> str:
    """Theme rules.

    Standalone (the SVG is the document, e.g. an ``<img>`` in a report): the
    chart paints its own surface and ink from ``prefers-color-scheme``, so it
    stays legible on any page. Inline: transparent, text inherits the page's
    ``currentColor``, and series follow the OS scheme unless the page stamps
    ``data-theme="light"``; ``data-theme="dark"`` on the root forces dark steps.
    """
    standalone = "svg.pe-chart:root"
    inline = ':root:not([data-theme="light"]) .pe-chart'
    forced = ':root[data-theme="dark"] .pe-chart'
    return (
        "<style>"
        f"{standalone}{{color:{INK_LIGHT}}}{standalone}>.pe-bg{{fill:{SURFACE_LIGHT}}}"
        f"@media (prefers-color-scheme: dark){{{standalone}{{color:{INK_DARK}}}{standalone}>.pe-bg{{fill:{SURFACE_DARK}}}"
        f"{_dark_rules(slots, (standalone, inline))}}}"
        f"{_dark_rules(slots, (forced,))}"
        "</style>"
    )


def _check_slots(count: int) -> None:
    if count > len(PALETTE_LIGHT):
        raise ValueError(f"{count} series exceed the {len(PALETTE_LIGHT)} categorical slots; facet or fold into 'Other'")


class _Canvas:
    """Accumulates SVG elements and the shared frame (title, legend, axes)."""

    def __init__(self, width: int, height: int, title: str, description: str, slots: int) -> None:
        _check_slots(slots)
        self.width, self.height = width, height
        self.title = title
        uid = hashlib.sha1(f"{title}|{description}|{width}x{height}".encode()).hexdigest()[:8]
        self.uid = f"pe-{uid}"
        self.parts: list[str] = []
        self.description = description or title
        self.slots = slots

    def add(self, element: str) -> None:
        self.parts.append(element)

    def text(self, x: float, y: float, content: str, *, size: int = 12, anchor: str = "start", weight: int = 400, opacity: float = 1.0, rotate: float | None = None) -> None:
        transform = f' transform="rotate({_num(rotate)} {_num(x)} {_num(y)})"' if rotate is not None else ""
        alpha = f' fill-opacity="{opacity}"' if opacity < 1 else ""
        self.add(
            f'<text x="{_num(x)}" y="{_num(y)}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="currentColor"{alpha}{transform}>{escape(content)}</text>'
        )

    def render(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" class="pe-chart" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="{self.uid}-t {self.uid}-d" '
            f'font-family="{FONT}">'
            f'<title id="{self.uid}-t">{escape(self.title)}</title><desc id="{self.uid}-d">{escape(self.description)}</desc>'
            f"{_style(self.slots)}"
            f'<rect class="pe-bg" width="{self.width}" height="{self.height}" rx="8" fill="none"/>'
        )
        return head + "".join(self.parts) + "</svg>"


def _legend(canvas: _Canvas, names: Sequence[str], kind: str, x0: float, y: float) -> float:
    """Draw a legend row (wrapping); returns the y below it."""
    if len(names) < 2:
        return y
    x = x0
    for i, name in enumerate(names):
        item_w = 22 + _text_width(name, 12) + 18
        if x + item_w > canvas.width - 16 and x > x0:
            x, y = x0, y + 18
        colour = PALETTE_LIGHT[i]
        if kind == "line":
            canvas.add(f'<path class="pe-l{i}" d="M{_num(x)},{_num(y - 4)}h16" stroke="{colour}" stroke-width="2" stroke-linecap="round" fill="none"/>')
        elif kind == "marker":
            canvas.add(_marker(MARKERS[i], x + 8, y - 4, 4.5, i))
        else:
            canvas.add(f'<rect class="pe-f{i}" x="{_num(x + 3)}" y="{_num(y - 9)}" width="10" height="10" rx="2" fill="{colour}"/>')
        canvas.text(x + 22, y, name)
        x += item_w
    return y + 18


def _marker(shape: str, cx: float, cy: float, r: float, slot: int, tooltip: str = "") -> str:
    colour = PALETTE_LIGHT[slot]
    common = f'class="pe-f{slot} pe-ring" fill="{colour}" stroke="{SURFACE_LIGHT}" stroke-width="2"'
    tip = f"<title>{escape(tooltip)}</title>" if tooltip else ""
    if shape == "circle":
        return f'<circle cx="{_num(cx)}" cy="{_num(cy)}" r="{_num(r)}" {common}>{tip}</circle>'
    pts: list[tuple[float, float]]
    if shape == "square":
        s = r * 0.9
        pts = [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
    elif shape == "triangle":
        pts = [(cx, cy - r * 1.2), (cx + r * 1.1, cy + r * 0.8), (cx - r * 1.1, cy + r * 0.8)]
    elif shape == "triangle-down":
        pts = [(cx, cy + r * 1.2), (cx + r * 1.1, cy - r * 0.8), (cx - r * 1.1, cy - r * 0.8)]
    elif shape == "diamond":
        pts = [(cx, cy - r * 1.25), (cx + r * 1.25, cy), (cx, cy + r * 1.25), (cx - r * 1.25, cy)]
    elif shape in ("plus", "cross"):
        a, b = r * 1.2, r * 0.45
        base = [(-b, -a), (b, -a), (b, -b), (a, -b), (a, b), (b, b), (b, a), (-b, a), (-b, b), (-a, b), (-a, -b), (-b, -b)]
        if shape == "cross":
            c = math.cos(math.pi / 4)
            base = [(c * (x - y), c * (x + y)) for x, y in base]
        pts = [(cx + x, cy + y) for x, y in base]
    else:  # star
        pts = [
            (cx + (r * 1.3 if i % 2 == 0 else r * 0.6) * math.sin(i * math.pi / 5),
             cy - (r * 1.3 if i % 2 == 0 else r * 0.6) * math.cos(i * math.pi / 5))
            for i in range(10)
        ]
    path = " ".join(f"{_num(x)},{_num(y)}" for x, y in pts)
    return f'<polygon points="{path}" {common}>{tip}</polygon>'


@dataclass
class _Frame:
    left: float
    right: float
    top: float
    bottom: float


def _frame(canvas: _Canvas, title: str, legend_names: Sequence[str], legend_kind: str, x_label: str, y_label: str, bottom_extra: float = 0.0) -> _Frame:
    y = 24.0
    if title:
        canvas.text(16, y, title, size=15, weight=600)
        y += 22
    y = _legend(canvas, legend_names, legend_kind, 16, y) if len(legend_names) >= 2 else y
    top = y + 6
    left = 64.0 if y_label else 52.0
    bottom = canvas.height - (44.0 if x_label else 28.0) - bottom_extra
    if x_label:
        canvas.text((left + canvas.width - 20) / 2, canvas.height - 10, x_label, size=13, anchor="middle", opacity=0.8)
    if y_label:
        canvas.text(16, (top + bottom) / 2, y_label, size=13, anchor="middle", opacity=0.8, rotate=-90)
    return _Frame(left, canvas.width - 20.0, top, bottom)


def _y_axis(canvas: _Canvas, frame: _Frame, scale: _Scale, ticks: Sequence[float]) -> None:
    for t in ticks:
        y = scale(t)
        canvas.add(f'<path d="M{_num(frame.left)},{_num(y)}H{_num(frame.right)}" stroke="currentColor" stroke-opacity="0.12" stroke-width="1" fill="none"/>')
        canvas.text(frame.left - 8, y + 4, format_number(t), anchor="end", opacity=0.75)


def _x_axis(canvas: _Canvas, frame: _Frame, scale: _Scale, ticks: Sequence[float]) -> None:
    canvas.add(f'<path d="M{_num(frame.left)},{_num(frame.bottom)}H{_num(frame.right)}" stroke="currentColor" stroke-opacity="0.35" stroke-width="1" fill="none"/>')
    for t in ticks:
        x = scale(t)
        canvas.add(f'<path d="M{_num(x)},{_num(frame.bottom)}v4" stroke="currentColor" stroke-opacity="0.35" stroke-width="1"/>')
        canvas.text(x, frame.bottom + 18, format_number(t), anchor="middle", opacity=0.75)


def _domain(values: Sequence[float], log: bool, lo: float | None, hi: float | None, include_zero: bool) -> tuple[float, float, list[float]]:
    finite = [v for v in values if v is not None and math.isfinite(v) and (v > 0 or not log)]
    if log:
        d_lo = lo if lo is not None else (min(finite) if finite else 1.0)
        d_hi = hi if hi is not None else (max(finite) if finite else 10.0)
        if d_hi <= d_lo:
            d_lo, d_hi = d_lo / 2.0, d_hi * 2.0
        return d_lo, d_hi, log_ticks(d_lo, d_hi)
    d_lo = lo if lo is not None else (min(finite) if finite else 0.0)
    d_hi = hi if hi is not None else (max(finite) if finite else 1.0)
    if include_zero and lo is None:
        d_lo = min(d_lo, 0.0)
    ticks = nice_ticks(d_lo, d_hi)
    t_lo = d_lo if lo is not None else ticks[0]
    t_hi = d_hi if hi is not None else ticks[-1]
    return t_lo, t_hi, [t for t in ticks if t_lo - 1e-12 <= t <= t_hi + 1e-12]


def _no_data(canvas: _Canvas, frame: _Frame) -> str:
    canvas.text((frame.left + frame.right) / 2, (frame.top + frame.bottom) / 2, "no data", anchor="middle", opacity=0.6)
    return canvas.render()


# --------------------------------------------------------------------------- charts


def line_chart(
    series: Sequence[Series],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    log_x: bool = False,
    log_y: bool = False,
    width: int = 640,
    height: int = 380,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    reference: float | None = None,
    reference_label: str = "",
    description: str = "",
) -> str:
    """Line chart with markers, optional shaded bands, log axes and a reference line."""
    canvas = _Canvas(width, height, title, description, len(series))
    frame = _frame(canvas, title, [s.name for s in series], "line", x_label, y_label)
    xs = [x for s in series for x in s.xs]
    ys = [y for s in series for y in [*s.ys, *(s.lo or []), *(s.hi or [])] if y is not None]
    if reference is not None:
        ys.append(reference)
    if not xs or not ys:
        return _no_data(canvas, frame)
    x_lo, x_hi, x_ticks = _domain(xs, log_x, x_min, x_max, include_zero=False)
    y_lo, y_hi, y_ticks = _domain(ys, log_y, y_min, y_max, include_zero=not log_y)
    sx = _Scale(x_lo, x_hi, frame.left + 8, frame.right - 8, log_x)
    sy = _Scale(y_lo, y_hi, frame.bottom, frame.top, log_y)
    _y_axis(canvas, frame, sy, y_ticks)
    _x_axis(canvas, frame, sx, x_ticks)
    for i, s in enumerate(series):
        _draw_band(canvas, s, sx, sy, i, log_x, log_y)
    if reference is not None:
        y = sy(reference)
        canvas.add(f'<path d="M{_num(frame.left)},{_num(y)}H{_num(frame.right)}" stroke="currentColor" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="4 4" fill="none"/>')
        if reference_label:
            canvas.text(frame.right, y - 6, reference_label, anchor="end", opacity=0.75)
    for i, s in enumerate(series):
        _draw_line(canvas, s, sx, sy, i, log_x, log_y)
    return canvas.render()


def _usable(v: float | None, log: bool) -> bool:
    return v is not None and math.isfinite(v) and (v > 0 or not log)


def _draw_band(canvas: _Canvas, s: Series, sx: _Scale, sy: _Scale, slot: int, log_x: bool, log_y: bool) -> None:
    if s.lo is None or s.hi is None:
        return
    pts = [(x, lo, hi) for x, lo, hi in zip(s.xs, s.lo, s.hi) if _usable(x, log_x) and _usable(lo, log_y) and _usable(hi, log_y)]
    if len(pts) < 2:
        return
    upper = " ".join(f"{_num(sx(x))},{_num(sy(hi))}" for x, _, hi in pts)
    lower = " ".join(f"{_num(sx(x))},{_num(sy(lo))}" for x, lo, _ in reversed(pts))
    canvas.add(f'<polygon class="pe-f{slot}" points="{upper} {lower}" fill="{PALETTE_LIGHT[slot]}" fill-opacity="0.12" stroke="none"/>')


def _draw_line(canvas: _Canvas, s: Series, sx: _Scale, sy: _Scale, slot: int, log_x: bool, log_y: bool) -> None:
    colour = PALETTE_LIGHT[slot]
    segments: list[list[tuple[float, float, float, float]]] = [[]]
    for x, y in zip(s.xs, s.ys):
        if _usable(x, log_x) and _usable(y, log_y):
            segments[-1].append((sx(x), sy(y), x, y))
        elif segments[-1]:
            segments.append([])
    for seg in segments:
        if len(seg) >= 2:
            d = "M" + "L".join(f"{_num(px)},{_num(py)}" for px, py, _, _ in seg)
            canvas.add(f'<path class="pe-l{slot}" d="{d}" stroke="{colour}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" fill="none"/>')
    for seg in segments:
        for px, py, x, y in seg:
            canvas.add(_marker("circle", px, py, 4, slot, f"{s.name}: {format_number(x)}, {y:.4g}"))


def _bar_path(x: float, base: float, end: float, w: float, radius: float = 4.0) -> str:
    """Bar with a rounded data end (top for positive bars, bottom for negative)."""
    h = abs(base - end)
    r = min(radius, w / 2.0, h)
    if end <= base:
        return (
            f"M{_num(x)},{_num(base)}V{_num(end + r)}A{_num(r)},{_num(r)} 0 0 1 {_num(x + r)},{_num(end)}"
            f"H{_num(x + w - r)}A{_num(r)},{_num(r)} 0 0 1 {_num(x + w)},{_num(end + r)}V{_num(base)}Z"
        )
    return (
        f"M{_num(x)},{_num(base)}V{_num(end - r)}A{_num(r)},{_num(r)} 0 0 0 {_num(x + r)},{_num(end)}"
        f"H{_num(x + w - r)}A{_num(r)},{_num(r)} 0 0 0 {_num(x + w)},{_num(end - r)}V{_num(base)}Z"
    )


def grouped_bar_chart(
    categories: Sequence[str],
    groups: Sequence[tuple[str, Sequence[float | None]]],
    *,
    errors: Sequence[Sequence[tuple[float, float] | None] | None] | None = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    width: int = 640,
    height: int = 380,
    y_min: float | None = None,
    y_max: float | None = None,
    description: str = "",
) -> str:
    """Grouped bars (one colour per group) with optional (lo, hi) error bars."""
    canvas = _Canvas(width, height, title, description, len(groups))
    longest = max((_text_width(c, 12) for c in categories), default=0.0)
    band_guess = (width - 84) / max(len(categories), 1)
    rotate = longest > band_guess - 6
    extra = min(longest * 0.6, 90.0) if rotate else 0.0
    frame = _frame(canvas, title, [g[0] for g in groups], "bar", x_label, y_label, bottom_extra=extra)
    values = [v for _, vals in groups for v in vals if v is not None]
    for err_list in errors or []:
        for e in err_list or []:
            if e is not None:
                values.extend(e)
    if not categories or not values:
        return _no_data(canvas, frame)
    y_lo, y_hi, y_ticks = _domain(values, False, y_min, y_max, include_zero=True)
    sy = _Scale(y_lo, y_hi, frame.bottom, frame.top)
    _y_axis(canvas, frame, sy, y_ticks)
    base = sy(min(max(0.0, y_lo), y_hi))
    canvas.add(f'<path d="M{_num(frame.left)},{_num(base)}H{_num(frame.right)}" stroke="currentColor" stroke-opacity="0.35" stroke-width="1" fill="none"/>')
    band = (frame.right - frame.left) / len(categories)
    n_groups = max(len(groups), 1)
    bar_w = max(2.0, min(24.0, (band * 0.8 - 2.0 * (n_groups - 1)) / n_groups))
    cluster_w = n_groups * bar_w + 2.0 * (n_groups - 1)
    for ci, cat in enumerate(categories):
        x0 = frame.left + ci * band + (band - cluster_w) / 2.0
        for gi, (gname, vals) in enumerate(groups):
            v = vals[ci] if ci < len(vals) else None
            if v is None or not math.isfinite(v):
                continue
            x = x0 + gi * (bar_w + 2.0)
            canvas.add(
                f'<path class="pe-f{gi}" d="{_bar_path(x, base, sy(v), bar_w)}" fill="{PALETTE_LIGHT[gi]}">'
                f"<title>{escape(f'{gname} / {cat}: {v:.4g}')}</title></path>"
            )
            err = errors[gi][ci] if errors and gi < len(errors) and errors[gi] and ci < len(errors[gi]) else None
            if err is not None:
                _error_bar(canvas, x + bar_w / 2.0, sy(err[0]), sy(err[1]))
        cx = frame.left + (ci + 0.5) * band
        if rotate:
            canvas.text(cx, frame.bottom + 14, cat, anchor="end", rotate=-35, opacity=0.75)
        else:
            canvas.text(cx, frame.bottom + 18, cat, anchor="middle", opacity=0.75)
    return canvas.render()


def _error_bar(canvas: _Canvas, x: float, y_lo: float, y_hi: float) -> None:
    canvas.add(
        f'<path d="M{_num(x)},{_num(y_lo)}V{_num(y_hi)}M{_num(x - 3)},{_num(y_lo)}h6M{_num(x - 3)},{_num(y_hi)}h6" '
        'stroke="currentColor" stroke-opacity="0.7" stroke-width="1.5" fill="none"/>'
    )


def scatter_chart(
    groups: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    width: int = 640,
    height: int = 420,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    point_labels: Sequence[Sequence[str]] | None = None,
    description: str = "",
) -> str:
    """Scatter plot; each group gets its own colour slot and marker shape."""
    canvas = _Canvas(width, height, title, description, len(groups))
    frame = _frame(canvas, title, [g[0] for g in groups], "marker", x_label, y_label)
    xs = [p[0] for _, pts in groups for p in pts]
    ys = [p[1] for _, pts in groups for p in pts]
    if not xs:
        return _no_data(canvas, frame)
    x_lo, x_hi, x_ticks = _domain(xs, False, x_min, x_max, include_zero=False)
    y_lo, y_hi, y_ticks = _domain(ys, False, y_min, y_max, include_zero=False)
    sx = _Scale(x_lo, x_hi, frame.left + 8, frame.right - 8)
    sy = _Scale(y_lo, y_hi, frame.bottom - 6, frame.top + 6)
    _y_axis(canvas, frame, sy, y_ticks)
    _x_axis(canvas, frame, sx, x_ticks)
    for gi, (gname, pts) in enumerate(groups):
        labels = point_labels[gi] if point_labels and gi < len(point_labels) else None
        for pi, (x, y) in enumerate(pts):
            who = f"{labels[pi]} " if labels and pi < len(labels) else ""
            tip = f"{who}({gname}): {x:.4g}, {y:.4g}"
            canvas.add(_marker(MARKERS[gi], sx(x), sy(y), 4.5, gi, tip))
    return canvas.render()


def histogram(
    values: Sequence[float],
    *,
    bins: int = 20,
    value_range: tuple[float, float] | None = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "count",
    width: int = 640,
    height: int = 360,
    description: str = "",
) -> str:
    """Histogram of one series (no legend), equal-width bins over ``value_range``."""
    canvas = _Canvas(width, height, title, description, 1)
    frame = _frame(canvas, title, [], "bar", x_label, y_label)
    data = [v for v in values if v is not None and math.isfinite(v)]
    if not data:
        return _no_data(canvas, frame)
    lo, hi = value_range if value_range else (min(data), max(data))
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    bins = max(1, bins)
    width_bin = (hi - lo) / bins
    counts = [0] * bins
    for v in data:
        if lo <= v <= hi:
            counts[min(int((v - lo) / width_bin), bins - 1)] += 1
    y_lo, y_hi, y_ticks = _domain([float(c) for c in counts], False, 0.0, None, include_zero=True)
    sy = _Scale(y_lo, y_hi, frame.bottom, frame.top)
    sx = _Scale(lo, hi, frame.left + 4, frame.right - 4)
    _y_axis(canvas, frame, sy, y_ticks)
    _x_axis(canvas, frame, sx, [t for t in nice_ticks(lo, hi) if lo - 1e-12 <= t <= hi + 1e-12])
    for i, c in enumerate(counts):
        if c == 0:
            continue
        x0, x1 = sx(lo + i * width_bin), sx(lo + (i + 1) * width_bin)
        w = max(x1 - x0 - 2.0, 1.0)
        tip = f"[{lo + i * width_bin:.4g}, {lo + (i + 1) * width_bin:.4g}): {c}"
        canvas.add(f'<path class="pe-f0" d="{_bar_path(x0 + 1.0, sy(0.0), sy(c), w)}" fill="{PALETTE_LIGHT[0]}"><title>{escape(tip)}</title></path>')
    return canvas.render()


def save(svg: str, path: str | Path) -> Path:
    """Write an SVG string to ``path`` (parents created)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg, encoding="utf-8")
    return target
