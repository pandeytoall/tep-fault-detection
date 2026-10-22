# Alarm limits on held-back normal runs at 1% and 5% FAR; delay to the first 3-in-a-row alarm

Every model's alarm limit is set on 100 held-back normal training runs so that it false-alarms on 1% of normal samples (the headline, matching process-monitoring practice) and, as a secondary column, 5% (matching some deep-learning papers). Detection rates are only comparable at equal FAR, and test files never set limits. Detection rate stays per sample, as in the literature. Detection delay is measured to the third alarm of the first 3-in-a-row after onset, which is when an operator would be alerted. It is reported as the median over detected runs, with the share never detected and the 3-in-a-row rate on fault-free runs, because at 5% FAR chance runs of three are common enough to fake a "detection".

**Considered options:** 1% only, or 5% only; first single alarm (k = 1, one stray alarm counts as detected); k = 6 (cautious, longer delays).
