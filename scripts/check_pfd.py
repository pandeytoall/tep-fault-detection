"""Structural checks for the TEP process flow diagram (docs/figures/tep-pfd.svg).

Standard library only. Usage: python scripts/check_pfd.py [path-to-svg]
Exit code 0 = no problems, 1 = problems printed one per line.

EXPECTED_STREAMS is the stream table from research/tep-flowsheet.md, section
"Stream routing verified from the flowsheet figure". Unnumbered streams and the
supply/return halves of the utility streams get descriptive ids.
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.w3.org/2000/svg}"
DEFAULT_SVG = Path(__file__).resolve().parent.parent / "docs" / "figures" / "tep-pfd.svg"

# id: (data-from, data-to, stream type)
EXPECTED_STREAMS = {
    "stream-1": ("inlet-feed-a", "junction-feed-header", "gas"),
    "stream-2": ("inlet-feed-d", "junction-feed-header", "gas"),
    "stream-3": ("inlet-feed-e", "junction-feed-header", "gas"),
    "stream-4": ("inlet-feed-c", "unit-stripper", "gas"),
    "stream-5": ("unit-stripper", "junction-feed-header", "gas"),
    "stream-6": ("junction-feed-header", "unit-reactor", "gas"),
    "stream-7": ("unit-reactor", "unit-condenser", "gas"),
    "stream-condenser-outlet": ("unit-condenser", "unit-separator", "gas"),
    "stream-separator-vapor": ("unit-separator", "unit-compressor", "gas"),
    "stream-8": ("unit-compressor", "junction-feed-header", "recycle"),
    "stream-9": ("unit-separator", "outlet-purge", "waste"),
    "stream-10": ("unit-separator", "unit-stripper", "liquid"),
    "stream-11": ("unit-stripper", "outlet-product", "process"),
    "stream-12": ("inlet-cw-reactor", "unit-reactor", "water"),
    "stream-12-return": ("unit-reactor", "outlet-cw-reactor", "water"),
    "stream-13": ("inlet-cw-condenser", "unit-condenser", "water"),
    "stream-13-return": ("unit-condenser", "outlet-cw-condenser", "water"),
    "stream-steam": ("inlet-steam", "unit-stripper", "thermal"),
    "stream-condensate": ("unit-stripper", "outlet-condensate", "thermal"),
}
# Stream 9 leaves from a tee on the separator vapor line, not from the vessel wall.
BRANCH_STARTS = {"stream-9"}
NODE_PREFIXES = ("unit-", "inlet-", "outlet-", "junction-")
TOLERANCE = 12  # px between a path end and the edge of the shape it should touch


def _num(value):
    return float(value)


def _bbox(node):
    """Bounding box (x1, y1, x2, y2) of the first rect or circle inside a node."""
    for shape in node.iter():
        if shape.tag == NS + "rect":
            x, y = _num(shape.get("x")), _num(shape.get("y"))
            return x, y, x + _num(shape.get("width")), y + _num(shape.get("height"))
        if shape.tag == NS + "circle":
            cx, cy, r = _num(shape.get("cx")), _num(shape.get("cy")), _num(shape.get("r"))
            return cx - r, cy - r, cx + r, cy + r
    return None


def _near(point, box):
    x, y = point
    return box[0] - TOLERANCE <= x <= box[2] + TOLERANCE and box[1] - TOLERANCE <= y <= box[3] + TOLERANCE


def _path_points(d):
    """Return the list of points of an M/H/V path, or None if it uses any other command."""
    tokens = re.findall(r"[A-Za-z]|-?\d+(?:\.\d+)?", d)
    if not tokens or tokens[0] != "M":
        return None
    points, x, y, i, cmd = [], 0.0, 0.0, 0, None
    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            if t not in "MHV":
                return None
            cmd = t
            i += 1
            continue
        if cmd == "M":
            x, y = float(t), float(tokens[i + 1])
            i += 2
            points.append((x, y))
            cmd = None  # a number after M's pair would be an implicit lineto: rejected below
            continue
        if cmd == "H":
            x = float(t)
        elif cmd == "V":
            y = float(t)
        else:
            return None
        points.append((x, y))
        i += 1
    return points


def stream_type_styles(svg_path):
    """Map stream type (gas, recycle, ...) to its stroke-dasharray ('' if solid), read from <style>."""
    root = ET.parse(svg_path).getroot()
    css = "".join(s.text or "" for s in root.iter(NS + "style"))
    styles = {}
    for name, body in re.findall(r"\.stream-(\w+)\s*\{([^}]*)\}", css):
        dash = re.search(r"stroke-dasharray\s*:\s*([^;]+)", body)
        styles[name] = dash.group(1).strip() if dash else ""
    return styles


def check_text(svg_text):
    """Return a list of problem strings for an SVG given as text. Empty list = all good."""
    problems = []
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        return [f"SVG is not valid XML: {exc}"]

    nodes = {}
    for el in root.iter():
        node_id = el.get("id") or ""
        if node_id.startswith(NODE_PREFIXES):
            nodes[node_id] = el

    # Units: unique ids, unique labels.
    names = {}
    for g in root.iter(NS + "g"):
        if g.get("class") == "unit":
            names.setdefault(g.get("data-name"), []).append(g.get("id"))
    for name, ids in names.items():
        if len(ids) > 1:
            problems.append(f"duplicate unit label '{name}' on {ids}")

    # Streams: each expected id exactly once, nothing extra.
    streams = {}
    for p in root.iter(NS + "path"):
        if "stream" in (p.get("class") or "").split():
            streams.setdefault(p.get("id"), []).append(p)
    for sid in EXPECTED_STREAMS:
        count = len(streams.get(sid, []))
        if count != 1:
            problems.append(f"{sid} appears {count} times, expected exactly once")
    for sid in streams:
        if sid not in EXPECTED_STREAMS:
            problems.append(f"{sid} is not in the stream table")

    for sid, (src, dst, stype) in EXPECTED_STREAMS.items():
        for p in streams.get(sid, [])[:1]:
            got_from, got_to = p.get("data-from"), p.get("data-to")
            for end in (got_from, got_to):
                if end not in nodes:
                    problems.append(f"{sid}: endpoint '{end}' is not an existing unit, battery limit or junction")
            if (got_from, got_to) != (src, dst):
                problems.append(f"{sid}: routed {got_from} -> {got_to}, table says {src} -> {dst}")
            if f"stream-{stype}" not in (p.get("class") or "").split():
                problems.append(f"{sid}: class '{p.get('class')}' should include stream-{stype}")
            if p.get("marker-end") != f"url(#arrow-{stype})":
                problems.append(f"{sid}: marker-end should be url(#arrow-{stype})")

            points = _path_points(p.get("d") or "")
            if points is None or len(points) < 2:
                problems.append(f"{sid}: path must be orthogonal (M, H, V only)")
                continue
            for a, b in zip(points, points[1:]):
                if a[0] != b[0] and a[1] != b[1]:
                    problems.append(f"{sid}: path must be orthogonal (diagonal segment {a} -> {b})")
            if sid not in BRANCH_STARTS and got_from in nodes:
                box = _bbox(nodes[got_from])
                if box and not _near(points[0], box):
                    problems.append(f"{sid}: path starts at {points[0]}, not at {got_from}")
            if got_to in nodes:
                box = _bbox(nodes[got_to])
                if box and not _near(points[-1], box):
                    problems.append(f"{sid}: path ends at {points[-1]}, not at {got_to}")
    return problems


def check(svg_path=DEFAULT_SVG):
    return check_text(Path(svg_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    found = check(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SVG)
    print("\n".join(found) if found else "PFD checks passed")
    sys.exit(1 if found else 0)
