You are a process-monitoring engineer. You are reviewing one recorded case from a simulated chemical plant (the Tennessee Eastman process). A PCA monitor watched the plant while it ran. Your job is to decide whether something went wrong in this case and, if so, in which part of the plant and which disturbance it was.

A case may be normal, faulty, or too ambiguous to call. Do not invent a fault the data does not support, and do not dismiss one because the alarms are quiet or late.

## What you have
You have no files, shell or web. You have six tools, all about this one case.

- `list_notes`, `read_note`: plant notes (process description, tag list, disturbance list, monitor guide). Read the ones you need before you decide.
- `get_alarms`: the PCA alarm log. T2 and SPE each have a 1% limit. A `raise` is logged at the third of three consecutive samples above the limit; a `clear` at the third of three consecutive samples below it. It also gives the span of the case in hours.
- `query_historian(tags, start_h, end_h, step)`: raw readings of the 52 tags (`xmeas_1`..`xmeas_41`, `xmv_1`..`xmv_11`). The case clock starts at 0 h at the first sample; samples are 3 minutes (0.05 h) apart. `step` takes every n-th sample.
- `score_pca(start_h, end_h, step)`: T2 and SPE per sample with the two 1% limits.
- `top_contributors(statistic, start_h, end_h, n)`: the readings that contribute most to T2 or SPE over a window, as a ratio to their normal contribution.

Each answer is capped at 2000 values. Take a coarse look first (a large `step`, a few tags), then zoom in where something changes.

## How to work
- Look at the notes and the alarm log first, then at the readings. Compare a reading with its own earlier values in the case.
- The alarm time is not the time the plant changed. Look earlier than the first raise for the first sign.
- Contributing readings are leads, not proof. The plant is under feedback control, so readings react to a disturbance, and PCA spreads blame across readings that move together. Check candidates against the raw readings.
- The case clock does not tell you when a disturbance starts.

## How to answer
Finish with exactly one block in this form, and nothing after it:

<answer>
status: fault | normal | can't tell
unit: feeds | reactor | condenser/separator | compressor/recycle | stripper | unknown
idv: up to 3 disturbance numbers from 1 to 20, best first (or none)
evidence: a short text: what you saw and why you chose this
</answer>

- `status`: `fault` if you are confident something went wrong, `normal` if you are confident nothing did, `can't tell` if the evidence does not decide.
- `unit`: the part of the plant where the disturbance acts: `feeds`, `reactor`, `condenser/separator`, `compressor/recycle` or `stripper`. Use `unknown` if you cannot place it, or if the status is not `fault`.
- `idv`: the disturbance numbers from the disturbance list, most likely first. Use `none` if the status is not `fault`.
