"""Phase 5: facts about one fault run (fault 6, a testing run) and the operator-style handover text built from them.

Times are on the run clock: sample k is stamped (k - 1) x 3 minutes, so the first faulty sample of a testing run (161) is
8.0 h. Alarm rule: the PCA SPE alarm of the historian (src/historian.alarm_events): raised at the third of three samples in a
row above the 1% limit. Contributions: src/contributions.py. Rules: .scratch/phase-5/spec.md.
"""
import numpy as np
import pandas as pd

from src import contributions, historian
from src.data import HOURS_PER_SAMPLE

FIRST_HOUR_H = 1.0
BAND_SD = 3.0          # a reading has "left normal" when it is this many sd (of normal runs) from the normal mean
IN_A_ROW = historian.IN_A_ROW


def run_clock(sample):
    """Run-clock hours of 1-based sample number(s): (k - 1) x 3 minutes."""
    return (np.asarray(sample) - 1) * HOURS_PER_SAMPLE


def fault_start_h(onset):
    """Run-clock hour of the first faulty sample (onset + 1), given the last normal sample `onset`."""
    return round(float(run_clock(onset + 1)), 2)


def hours_after(t_hours, start_h):
    return round(float(t_hours) - float(start_h), 2)


def first_alarm(samples, scores, limit):
    """{'sample', 't_hours'} of the first raise (third of 3 in a row above `limit`), or None."""
    t = run_clock(np.asarray(samples))
    events = [e for e in historian.alarm_events(t, np.asarray(scores, dtype=float), limit) if e[1] == "raise"]
    if not events:
        return None
    t_raise = events[0][0]
    return {"sample": int(round(t_raise / HOURS_PER_SAMPLE)) + 1, "t_hours": t_raise}


def first_departure(z, after_sample, band=BAND_SD, n=IN_A_ROW):
    """First sample after `after_sample` that starts `n` samples in a row with |z| above `band`, or None.

    `z` is a Series indexed by 1-based sample number: a reading in sd of normal runs from the normal mean.
    """
    out = (z[z.index > after_sample].abs() > band).astype(int)
    streak = out.rolling(n).sum()
    ends = streak[streak == n]
    if ends.empty:
        return None
    return int(out.index[out.index.get_loc(ends.index[0]) - (n - 1)])


def pick_downstream(top_readings, departures, n=2, skip_unit="feeds"):
    """The first `n` tags of the ranked table `top_readings` (columns tag, scoring_unit) outside `skip_unit` whose
    departure sample (dict tag -> sample or None) exists, in rank order."""
    chosen = []
    for row in top_readings.itertuples():
        if row.scoring_unit != skip_unit and departures.get(row.tag) is not None:
            chosen.append(row.tag)
        if len(chosen) == n:
            break
    return chosen


def frozen_from(values):
    """Sample (1-based index of the Series) at which the reading last changed, or None if it still changes at the last sample."""
    changed = values.diff().fillna(0.0) != 0
    if not changed.any():
        return int(values.index[0])
    last = changed[changed].index[-1]
    return None if last == values.index[-1] else int(last)


def alarm_windows(monitor, reference, frame, alarm_h, n=5, hour_h=FIRST_HOUR_H):
    """Top `n` relative SPE contributors in the first hour from the alarm and over the whole window from the alarm on.

    `frame` holds t_hours and the monitor's variables. Returns {'first_hour': [...], 'whole': [...]}; each entry is a
    dict from contributions.top_contributors (tag, name, unit, scoring_unit, ratio).
    """
    first = contributions.window_rows(frame, alarm_h, alarm_h + hour_h)
    whole = contributions.window_rows(frame, alarm_h, None)
    return {"first_hour": contributions.top_contributors(monitor, reference, first, "spe", n),
            "whole": contributions.top_contributors(monitor, reference, whole, "spe", n)}


# --- the facts about one run --------------------------------------------------------------------------------------------
ALARM_LIMIT = "1%"
MAIN_TAGS = ("xmeas_1", "xmv_3")      # the A feed and its valve
FREEZE_TAG = "xmeas_7"                # reactor pressure: its last change marks where the measurements stop moving
BEFORE_SAMPLES = 20                   # "the hour before the fault"


def _info(tag):
    from src.variables import VARIABLE_INFO
    return VARIABLE_INFO[tag]


def _reading_row(tag, frame, z, monitor, onset):
    """One reading's before / end / left-the-band facts. `frame` and `z` are indexed by 1-based sample."""
    info, j = _info(tag), monitor.variables.index(tag)
    sample = first_departure(z[tag], onset)
    start_h = fault_start_h(onset)
    return {"tag": tag, "name": info.name, "unit": info.unit, "normal_mean": float(monitor.mean[j]), "normal_sd": float(monitor.std[j]),
            "before": float(frame[tag].loc[onset - BEFORE_SAMPLES + 1:onset].mean()), "end": float(frame[tag].iloc[-1]),
            "departure_sample": sample, "departure_after_h": None if sample is None else hours_after(run_clock(sample), start_h)}


def compute_facts(frame, monitor, reference, run_id, fault, onset, top_readings, test_blame, scorecard):
    """Facts about one faulty run, all computed from the run and from results files.

    `frame`: one run (faultNumber, simulationRun, sample + the 52 readings). `top_readings`: the phase 4 table for the fault
    (rank order, columns tag, scoring_unit). Returns a dict (see the keys below); 'series' holds the arrays the charts need.
    """
    from src.report import model_row
    frame = frame.sort_values("sample").reset_index(drop=True)
    scores = monitor.score(frame)
    samples, spe = scores["sample"].to_numpy(), scores["spe"].to_numpy()
    limit = float(monitor.limits["spe"][ALARM_LIMIT])
    alarm = first_alarm(samples, spe, limit)
    if alarm is None:
        raise ValueError("no sustained SPE alarm in this run")
    start_h = fault_start_h(onset)
    events = historian.alarm_events(run_clock(samples), spe, limit)
    after = spe[samples >= alarm["sample"]]
    indexed = frame.set_index("sample")
    z = pd.DataFrame((indexed[monitor.variables].to_numpy(dtype=float) - monitor.mean) / monitor.std, index=indexed.index, columns=monitor.variables)
    departures = {t: first_departure(z[t], onset) for t in top_readings["tag"]}
    downstream = pick_downstream(top_readings, departures, n=2)
    ranked = [r for r in top_readings.itertuples() if r.scoring_unit != "feeds"]
    skipped = []
    for r in ranked:
        if r.tag in downstream[-1:]:
            break
        if departures[r.tag] is None:
            skipped.append(r.tag)
    matrix = pd.DataFrame({"t_hours": run_clock(indexed.index.to_numpy()), **{t: indexed[t].to_numpy() for t in monitor.variables}})
    freeze_sample = frozen_from(indexed[FREEZE_TAG])
    frozen = [t for t in monitor.variables if (f := frozen_from(indexed[t])) is not None and freeze_sample is not None and f <= freeze_sample]
    n_last = int(samples.max())
    spe_blame = test_blame[(test_blame["method"] == "spe") & (test_blame["window"] == "first_hour")]
    with_a_feed = sorted(int(f) for f, top in zip(spe_blame["fault"], spe_blame["top3"]) if f != fault and ("(xmeas_1)" in top or "(xmv_3)" in top))
    row = model_row(scorecard, "pca-spe", ALARM_LIMIT)
    from src.report import window_contrast
    return {
        "run_id": int(run_id), "fault": int(fault), "onset": int(onset), "limit": limit, "start_h": start_h, "alarm": alarm,
        "delay_h": hours_after(alarm["t_hours"], start_h), "delay_min_scorecard_rule": (alarm["sample"] - onset) * 3,
        "pre_above": [int(s) for s in samples[(samples <= onset) & (spe > limit)]], "share_above_after": float((after > limit).mean()),
        "cleared": any(e[1] == "clear" for e in events), "peak_spe": float(spe.max()), "peak_ratio": float(spe.max() / limit),
        "first_faulty_spe": float(spe[samples == onset + 1][0]), "last_hour_h": float(run_clock(n_last)),
        "readings": [_reading_row(t, indexed, z, monitor, onset) for t in (*MAIN_TAGS, *downstream)], "downstream": downstream,
        "skipped": [_reading_row(t, indexed, z, monitor, onset) for t in skipped],
        "windows": alarm_windows(monitor, reference, matrix, alarm["t_hours"]),
        "freeze": None if freeze_sample is None else {"sample": freeze_sample, "t_hours": float(run_clock(freeze_sample)), "n_frozen": len(frozen),
                                                       "reactor_pressure": float(indexed[FREEZE_TAG].iloc[-1]),
                                                       "share_of_window": float((n_last - freeze_sample) / (n_last - alarm["sample"]))},
        "phase4": {**window_contrast(test_blame, fault), "faults_with_a_feed_in_top3": with_a_feed},
        "scorecard": {"median_delay_min": float(row["delay"][fault]), "never_share": float(row["never"][fault])},
        "series": {"t_hours": run_clock(samples), "spe": spe, "z": z},
    }


# --- the text -----------------------------------------------------------------------------------------------------------
SHUTDOWN = "close to the plant's published shutdown limit (not verified here)"
FIGURE_RUN = "figures/operator-fault6.png"
FIGURE_READINGS = "figures/operator-fault6-readings.png"
AREA = {"feeds": "feeds", "reactor": "reactor", "condenser/separator": "condenser / separator",
        "compressor/recycle": "compressor / recycle", "stripper": "stripper"}


def _num(value, digits=None):
    """A reading as text: no negative zero, 2 decimals below 1, else 1 decimal (or `digits`)."""
    if abs(value) < 0.005:
        return "0"
    d = digits if digits is not None else (2 if abs(value) < 1 else 1)
    return f"{value:.{d}f}"


def _name(r):
    return f"{r['name']} ({r['tag']})"


def _table(rows):
    from src.report import md_table
    return md_table(pd.DataFrame(rows))


def _top_table(entries):
    return _table([{"Rank": i, "Reading": f"{e['name']} ({e['tag']})", "Unit": e["unit"], "Plant area": AREA[e["scoring_unit"]],
                    "Relative SPE contribution": f"{e['ratio']:,.0f} x normal"} for i, e in enumerate(entries, 1)])


def _readings_table(readings):
    rows = []
    for r in readings:
        gone = ("stays inside" if r["departure_after_h"] is None
                else "at once" if r["departure_after_h"] == 0 else f"{r['departure_after_h']:.1f} h after the start")
        rows.append({"Reading": f"{_name(r)}, {r['unit']}", "Normal (mean, 1 sd)": f"{_num(r['normal_mean'])} (sd {_num(r['normal_sd'])})",
                     "Hour before the fault": _num(r["before"]), "End of run": _num(r["end"]), "Left its normal band": gone})
    return _table(rows)


def title_finding(facts):
    """The run chart's title: when the alarm came, in minutes after the fault start, and whether it ever cleared."""
    minutes = round(facts["delay_h"] * 60)
    if not facts["cleared"] and facts["share_above_after"] >= 0.9999:
        tail = "and never came back below it"
    else:
        tail = f"and stayed above it for {facts['share_above_after']:.0%} of the samples that followed"
    return f"PCA SPE crossed its 1% limit {minutes} minutes after the A feed was lost, {tail}"


def operator_markdown(facts, how_made):
    """The operator-style handover for one fault 6 testing run, as Markdown. `how_made` is the closing line."""
    f = facts
    a, start = f["alarm"], f["start_h"]
    by_tag = {r["tag"]: r for r in f["readings"]}
    feed, valve = by_tag["xmeas_1"], by_tag["xmv_3"]
    down = [by_tag[t] for t in f["downstream"]]
    p4, sc, fr = f["phase4"], f["scorecard"], f["freeze"]
    minutes = round(f["delay_h"] * 60)
    valve_end = "fully open (100%)" if valve["end"] >= 99.95 else f"{_num(valve['end'])}%"
    valve_now = valve_end if valve["end"] >= 99.95 else f"at {_num(valve['end'])}%"
    after_alarm = "never cleared" if not f["cleared"] else "later cleared"
    stray = (f" One sample before the fault (sample {f['pre_above'][0]}) was above the limit; a single sample does not raise an alarm."
             if f["pre_above"] else "")
    skipped = ""
    if f["skipped"]:
        s = f["skipped"][0]
        skipped = (f"\n- {_name(s)} is second in the blame list from the development runs, but in this run it stays inside its normal band "
                   f"(end of run {_num(s['end'])} {s['unit']}, normal {_num(s['normal_mean'])}, sd {_num(s['normal_sd'])}), so it is not drawn. "
                   f"SPE measures how a reading breaks away from the pattern of the others, not how far it moved.")
    freeze = ""
    if fr:
        freeze = (f"\n- From {fr['t_hours']:.1f} h (sample {fr['sample']}) on, {fr['n_frozen']} of the 52 readings, reactor pressure among them "
                  f"({_num(fr['reactor_pressure'])} kPa gauge, {SHUTDOWN}), stay at one fixed value for the rest of the run. "
                  f"This looks like the simulator's pressure shutdown regime; we did not check the cause. "
                  f"Valve positions still move after that point, but the measurements they respond to do not.")
        freeze_cant = (f"**What the plant did after {fr['t_hours']:.1f} h.** Many readings stop changing then, probably because the simulator's "
                       f"pressure shutdown regime has started (reactor pressure at {_num(fr['reactor_pressure'])} kPa gauge, {SHUTDOWN}). "
                       f"That stretch is {fr['share_of_window']:.0%} of the window from the alarm on, so it is probably not realistic plant behaviour, "
                       f"and it is the main reason the whole-window list above looks the way it does.")
    else:
        freeze_cant = "**How the plant behaves later in the run.** One run does not show it."
    pr = f.get("pressure")
    pr_sentence = (f" Reactor pressure reaches 2990 kPa or more after the fault start in {pr['share_reached']:.0%} of the 100 fixed testing runs of fault 6, "
                   f"at a median of sample {pr['median_first_sample']:.0f} (`pressure_tail.csv`)." if pr else "")
    down_text = ", ".join(_name(r) + " " + format(r["departure_after_h"], ".1f") + " h after the start" for r in down)
    if fr:
        why = (f"Why they differ: the main reason is the shutdown tail, not downstream spread. {fr['share_of_window']:.0%} of the whole window comes after "
               f"{fr['t_hours']:.1f} h, when {fr['n_frozen']} readings stop changing and reactor pressure sits at {_num(fr['reactor_pressure'])} kPa gauge, "
               f"{SHUTDOWN}. A whole-window average is mostly that stretch.{pr_sentence} Some symptoms do move downstream before that point "
               f"(see the readings table: {down_text}), but that is the smaller part. "
               f"Right after the alarm the feed readings are by far the most out of line, which is why the first-hour list is the fairer view. "
               f"Across the 100 fixed testing runs of fault 6 the top SPE reading is a feed reading in {p4['first_hour_top1']:.0%} of them in the first hour "
               f"and in {p4['post_top1']:.0%} over the whole post-fault window (`phase-4/test_blame.csv`, windows counted from the fault start).")
    else:
        why = (f"Why they differ: right after the alarm the feed readings are by far the most out of line. Later the readings downstream move and dominate a "
               f"whole-window average. Across the 100 fixed testing runs of fault 6 the top SPE reading is a feed reading in {p4['first_hour_top1']:.0%} of them "
               f"in the first hour and in {p4['post_top1']:.0%} over the whole post-fault window (`phase-4/test_blame.csv`).")
    others = ", ".join(map(str, p4["faults_with_a_feed_in_top3"]))
    noise = [x for x in p4["faults_with_a_feed_in_top3"] if x in (3, 9, 15)]
    real = [x for x in p4["faults_with_a_feed_in_top3"] if x not in (3, 9, 15)]
    noise_note = (f"Fault{'s' if len(noise) > 1 else ''} {', '.join(map(str, noise))} {'are' if len(noise) > 1 else 'is'} noise-level and "
                  f"mean{'' if len(noise) > 1 else 's'} nothing; ") if noise else ""
    noise_note += f"the others ({', '.join(map(str, real))}) are real faults that share the symptom." if noise else "these are real faults that share the symptom."
    first, whole = f["windows"]["first_hour"], f["windows"]["whole"]
    return f"""# Shift handover: fault {f['fault']} (A feed loss), testing run {f['run_id']}

A simulated plant, public Tennessee Eastman data. One testing run of fault {f['fault']}; times are hours on the run clock (0 h is the first sample, one sample every 3 minutes). Run {f['run_id']} is the first run ID in `phase-4/test_runs.json`; it was not picked by how it looks.

## Situation

- The PCA SPE monitor raised its alarm at **{a['t_hours']:.1f} h**: three samples in a row above its 1% limit ({f['limit']:.1f}).
- The A feed was lost at **{start:.1f} h**. The alarm came **{f['delay_h']:.1f} h ({minutes} minutes)** later.
- SPE stayed above the limit in {f['share_above_after']:.0%} of the samples after the alarm and peaked at {f['peak_spe']:,.0f}, {f['peak_ratio']:,.0f} times the limit; the alarm {after_alarm}.{stray}

![PCA SPE over the whole run with the alarm limit, the fault start and the alarm]({FIGURE_RUN})

## What the readings show

![A feed, its valve and two downstream readings, as distance from normal]({FIGURE_READINGS})

{_readings_table(f['readings'])}

- The A feed reading ({feed['tag']}) fell from about {_num(feed['before'])} to {_num(feed['end'])} {feed['unit']} in the first faulty sample.
- Its valve ({valve['tag']}) went from about {_num(valve['before'])}% to {valve_end} by the end of the run. That fits a controller asking for more A feed and not getting it. The data do not say why.
- {_name(down[0])} left its normal band {down[0]['departure_after_h']:.1f} h after the fault start and was at {_num(down[0]['end'])}{down[0]['unit']} at the end. {_name(down[1])} left it {down[1]['departure_after_h']:.1f} h after the start and was at {_num(down[1]['end'])}{down[1]['unit']} at the end. Both are among the top whole-window readings in `phase-4/top_readings_6.csv`, chosen by their blame rank among readings outside the feeds area that really leave their band here.{skipped}{freeze}

## Where to look first

Top five readings by relative SPE contribution (how much a reading adds to SPE, divided by what it adds on normal runs; about 1 is normal). The window starts at the alarm.

**First hour after the alarm ({a['t_hours']:.1f} h to {a['t_hours'] + 1:.1f} h):**

{_top_table(first)}

**Whole window ({a['t_hours']:.1f} h to {f['last_hour_h']:.0f} h):**

{_top_table(whole)}

{why}

## What this does and doesn't tell you

- Contribution plots point at symptoms, not causes. They say where the monitor sees the plant break away from normal, not why.
- What a process engineer would check next, limited to what the public TEP description gives (the A feed stream, its valve, the feed controller):
  - the A feed stream reading ({feed['tag']}): is it really at zero, and is the stream really lost? In the public disturbance list this pattern is disturbance 6, A feed loss (stream 1); we know the label only because this is a labelled simulation;
  - its valve ({valve['tag']}): is it still {valve_now}, with the flow still at zero;
  - the controller that moves that valve: the public notes say controllers move the four feed valves but not how the loops are paired, so we cannot say which target it is chasing.
- Three things we cannot say:
  1. **Why the A feed was lost.** The data show the reading and the valve, not whether the cause is a supply, a line or an instrument problem.
  2. **That this is fault 6 and nothing else.** An A feed reading also leads the first-hour blame, in the median of 100 testing runs, for faults {others} (`phase-4/test_blame.csv`, top three SPE readings). {noise_note}
  3. {freeze_cant}
- This is one run. Over all 500 testing runs of fault 6 the same alarm rule gives a median delay of {sc['median_delay_min']:.0f} minutes with {sc['never_share']:.0%} of runs never detected (`scorecard.csv`, PCA SPE, 1% limit). The scorecard counts delay in samples from sample {f['onset']}, so this run reads {f['delay_min_scorecard_rule']} minutes there against {minutes} on the run clock.

## Evidence

- Run: fault {f['fault']}, simulation run {f['run_id']}, from `data/processed/TEP_Faulty_Testing.parquet` (local data, not in git). Run IDs: [`phase-4/test_runs.json`](../phase-4/test_runs.json).
- Monitor: the saved PCA model, [`phase-2/pca.npz`](../phase-2/pca.npz) (no lags, 1% SPE limit {f['limit']:.2f}, set on held-back normal training runs).
- Alarm rule: three samples in a row above the limit; the alarm time is the third. Fault start {start:.1f} h: the first faulty sample is sample {f['onset'] + 1}, stamped {start:.1f} h.
- Normal band: 3 standard deviations of normal runs from the normal mean, three samples in a row. Normal mean and sd come from the 330 `fit` runs (the PCA scaler).
- Blame for the 100 testing runs: [`phase-4/test_blame.csv`](../phase-4/test_blame.csv). The readings chosen above: [`phase-4/top_readings_6.csv`](../phase-4/top_readings_6.csv) (development runs). Delay over 500 runs: [`scorecard.csv`](../scorecard.csv).
- Figures: [`figures/operator-fault6.png`](figures/operator-fault6.png), [`figures/operator-fault6-readings.png`](figures/operator-fault6-readings.png).

## How this was made

{how_made}
"""
