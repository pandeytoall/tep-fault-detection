"""The assistant's final answer block, and how it is parsed.

The prompt asks for exactly one block at the end of the reply:

    <answer>
    status: fault | normal | can't tell
    unit: feeds | reactor | condenser/separator | compressor/recycle | stripper | unknown
    idv: up to 3 fault numbers 1-20, best first (or none)
    evidence: a short text
    </answer>

parse_answer never guesses: a missing or invalid field makes the whole answer unparseable (scored as wrong).
"""
import re

from src.variables import SCORING_UNITS

STATUSES = ("fault", "normal", "can't tell")
UNITS = tuple(SCORING_UNITS) + ("unknown",)
MAX_IDV = 3
_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_FIELD = re.compile(r"^\s*(status|unit|idv|evidence)\s*:\s*(.*)$", re.IGNORECASE)
_STATUS_ALIASES = {"cannot tell": "can't tell", "cant tell": "can't tell", "can not tell": "can't tell"}


def _fail(error):
    return {"ok": False, "error": error}


def parse_answer(text):
    """{'ok': True, status, unit, idv: [ints], evidence} or {'ok': False, 'error': reason}. The last block wins."""
    blocks = _BLOCK.findall(text or "")
    if not blocks:
        return _fail("no <answer> block")
    fields, current = {}, None
    for line in blocks[-1].strip().splitlines():
        m = _FIELD.match(line)
        if m:
            current = m.group(1).lower()
            fields[current] = m.group(2).strip()
        elif current == "evidence" and line.strip():
            fields["evidence"] += " " + line.strip()

    status = fields.get("status", "").lower().replace("\u2019", "'").strip()
    status = _STATUS_ALIASES.get(status, status)
    if status not in STATUSES:
        return _fail(f"status must be one of {STATUSES}, got {fields.get('status')!r}")
    unit = fields.get("unit", "").lower().strip()
    if unit not in UNITS:
        return _fail(f"unit must be one of {UNITS}, got {fields.get('unit')!r}")
    if "idv" not in fields:
        return _fail("idv missing")
    raw = fields["idv"].strip().lower()
    idv = []
    if raw not in ("", "none"):
        parts = [p for p in re.split(r"[,\s]+", raw) if p]
        if not all(p.isdigit() for p in parts):
            return _fail(f"idv must be numbers 1-20 or none, got {fields['idv']!r}")
        idv = [int(p) for p in parts]
        if len(idv) > MAX_IDV or len(set(idv)) != len(idv) or not all(1 <= i <= 20 for i in idv):
            return _fail(f"idv must be at most {MAX_IDV} different numbers from 1 to 20, got {fields['idv']!r}")
    return {"ok": True, "status": status, "unit": unit, "idv": idv, "evidence": fields.get("evidence", "")}
