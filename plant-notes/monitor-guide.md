# Monitor guide: the PCA monitor and its tools

## What the monitor is
A PCA model was fitted on normal operation only. It sees the 52 readings, each scaled to its normal spread. PCA finds a few directions in which the readings normally vary together (the principal components). Two summary numbers are computed for every sample.

- **T-squared (T²).** How far the sample is from the normal centre, measured inside the retained directions. A high T² means the usual patterns are present but at an unusual size.
- **SPE** (squared prediction error). How much of the sample the retained directions fail to explain. A high SPE means the readings broke the normal correlations between each other.

Either can alarm, and a fault may raise one, both or neither.

## Alarm limit
Each statistic has a limit set so that about 1% of samples from normal runs exceed it. So an occasional isolated exceedance is expected in a normal run. A 1% limit is a statistical threshold and not proof of a fault.

## Alarm log rule
The alarm log records events per statistic.
- `raise`: at the third of three consecutive samples above the limit.
- `clear`: at the third of three consecutive samples below the limit, after a raise.

Samples are 3 minutes apart, so a raise is logged 9 minutes (the third sample) after the exceedances begin. The raise time is therefore later than the moment the readings first moved. A case with no raise has no logged alarm; that does not by itself prove it is normal or faulty.

## Relative contribution
For each statistic the total value is split across the 52 readings, giving each reading a contribution. The tools average these over a time window (by default from the first raise to the end of the case, or the whole case if there was no raise) and divide by that reading's average contribution on normal data.

- A relative contribution near 1 means that reading contributes as much as it normally does.
- A value well above 1 means it contributes more than usual in that window.
- Top-contributor lists show the 5 largest per statistic.

Caution: contributions smear across correlated readings. When one reading goes wrong, PCA spreads part of the blame to other readings that normally move with it, so the top reading is a lead and not a diagnosis. Compare raw readings from the historian before concluding anything. Because the plant is under feedback control, readings that react to a disturbance can also show up high.

## Tools
Use `list_notes` and `read_note` for these notes. Use the historian, alarm, score and contribution tools for the case data. Each case clock starts at 0 h.

## Sources
- Standard PCA monitoring: Kourti and MacGregor (1995), "Process analysis, monitoring and diagnosis, using multivariate projection methods", Chemometrics and Intelligent Laboratory Systems 28, 3-21.
- T² contribution decomposition: Alcala and Qin (2011), "Analysis and generalization of fault diagnosis methods for process monitoring", Journal of Process Control 21, 322-330.
