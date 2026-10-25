"""The 52 process variables: D&V name, unit, plant unit and kind.

Names and units: Downs & Vogel (1993) as compiled in research/tep-flowsheet.md, section 4 (variable map),
columns "D&V name", "Unit" and "Unit / stream". Stream routing: the same file, sections 2 and
"Stream routing verified from the flowsheet figure" (Bathelt, Ricker & Jelali 2015, Fig. 3).

plant_unit, one of feed, reactor, condenser, separator, compressor, stripper, purge, product.
A row that names a stream goes to the closest unit:
- streams 1-4 (fresh feeds, xmeas_1-4 and xmv_1-4): feed. Stream 4 enters the stripper, but it is a fresh feed.
- stream 6 (reactor feed, xmeas_6 and analyzers xmeas_23-28): reactor.
- stream 8 (recycle, xmeas_5): compressor, the unit that drives it.
- stream 9 (purge, xmeas_10, xmv_6, analyzers xmeas_29-36): purge.
- stream 10 (separator underflow, xmeas_14, xmv_7): separator.
- stream 11 (product, xmeas_17, xmv_8, analyzers xmeas_37-41): product.
- stream 12 (reactor cooling water, xmeas_21, xmv_10): reactor. Stream 13 (xmeas_22, xmv_11): condenser.
- stripper steam (xmeas_19, xmv_9): stripper.

Rows the flowsheet marks as not fully verified:
- xmeas_22: D&V name says "separator cooling water outlet temperature"; the figure shows the only
  cooling-water outlet downstream of the reactor on the condenser return. Condenser is an inference.
- xmv_5: the recycle valve exists (CONFIRMED); its exact piping (bypass around the compressor) is UNVERIFIED.
- xmv_9: "reboiler" wording UNVERIFIED; the steam goes to the stripper.
- Units of the xmv rows are not in the flowsheet file: D&V give valve positions in %.
"""
from typing import NamedTuple

from src.data import VARIABLES


class Variable(NamedTuple):
    tag: str
    name: str
    unit: str
    plant_unit: str
    kind: str  # measured (xmeas 1-22), analyzer (xmeas 23-41) or manipulated (xmv 1-11)


# (name, unit, plant_unit) in column order xmeas_1..41, then xmv_1..11
_XMEAS = [
    ("A feed", "kscmh", "feed"), ("D feed", "kg/h", "feed"), ("E feed", "kg/h", "feed"),
    ("A and C feed", "kscmh", "feed"), ("Recycle flow", "kscmh", "compressor"),
    ("Reactor feed rate", "kscmh", "reactor"), ("Reactor pressure", "kPa gauge", "reactor"),
    ("Reactor level", "%", "reactor"), ("Reactor temperature", "deg C", "reactor"),
    ("Purge rate", "kscmh", "purge"), ("Product separator temperature", "deg C", "separator"),
    ("Product separator level", "%", "separator"), ("Product separator pressure", "kPa gauge", "separator"),
    ("Product separator underflow", "m3/h", "separator"), ("Stripper level", "%", "stripper"),
    ("Stripper pressure", "kPa gauge", "stripper"), ("Stripper underflow", "m3/h", "product"),
    ("Stripper temperature", "deg C", "stripper"), ("Stripper steam flow", "kg/h", "stripper"),
    ("Compressor work", "kW", "compressor"), ("Reactor cooling water outlet temperature", "deg C", "reactor"),
    ("Separator cooling water outlet temperature", "deg C", "condenser"),
]
_ANALYZERS = (
    [(f"Reactor feed analysis, component {c}", "mol %", "reactor") for c in "ABCDEF"]
    + [(f"Purge gas analysis, component {c}", "mol %", "purge") for c in "ABCDEFGH"]
    + [(f"Product analysis, component {c}", "mol %", "product") for c in "DEFGH"]
)
_XMV = [
    ("D feed flow", "%", "feed"), ("E feed flow", "%", "feed"), ("A feed flow", "%", "feed"),
    ("A and C feed flow", "%", "feed"), ("Compressor recycle valve", "%", "compressor"),
    ("Purge valve", "%", "purge"), ("Separator pot liquid flow", "%", "separator"),
    ("Stripper liquid product flow", "%", "product"), ("Stripper steam valve", "%", "stripper"),
    ("Reactor cooling water flow", "%", "reactor"), ("Condenser cooling water flow", "%", "condenser"),
]
_KINDS = ["measured"] * 22 + ["analyzer"] * 19 + ["manipulated"] * 11

VARIABLE_INFO = {
    tag: Variable(tag, name, unit, plant_unit, kind)
    for tag, (name, unit, plant_unit), kind in zip(VARIABLES, _XMEAS + _ANALYZERS + _XMV, _KINDS)
}


# --- scoring units for the assistant (phase 2b): the 8 plant units folded into 5 ---------------------------------
SCORING_UNITS = ["feeds", "reactor", "condenser/separator", "compressor/recycle", "stripper"]
PLANT_TO_SCORING_UNIT = {
    "feed": "feeds", "reactor": "reactor", "condenser": "condenser/separator", "separator": "condenser/separator",
    "compressor": "compressor/recycle", "purge": "compressor/recycle", "stripper": "stripper", "product": "stripper",
}


def scoring_unit(tag):
    """One of SCORING_UNITS for a tag (xmeas_N / xmv_N)."""
    return PLANT_TO_SCORING_UNIT[VARIABLE_INFO[tag].plant_unit]
