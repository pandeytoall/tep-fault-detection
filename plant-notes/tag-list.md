# Tag list: the 52 readings

Sampling: one value every 3 minutes. Tags `xmeas_1` to `xmeas_41` are measurements; `xmv_1` to `xmv_11` are manipulated variables (valve or flow positions set by the controllers). A twelfth manipulated variable in the original design (agitator speed) is not in the data.

Analyzer tags `xmeas_23` to `xmeas_41` come from sampled analyzers (composition, mol %). They update less often than every 3 minutes: the reactor feed and purge analyzers about every 6 minutes (0.1 h), the product analyzer about every 15 minutes (0.25 h), each with a delay (dead time) of the same length. Between updates the value stays the same. (Interval and delay are the D&V figures; the "held value" behaviour is our reading of them.)

Scoring unit mapping (from the plant unit): feed -> feeds; reactor -> reactor; condenser, separator -> condenser/separator; compressor, purge -> compressor/recycle; stripper, product -> stripper.

Name and unit follow D&V. "kscmh" is thousand standard cubic metres per hour. The `xmv` units are percent of valve or flow range.

| Tag | D&V name | Unit | Plant unit | Scoring unit |
|---|---|---|---|---|
| xmeas_1 | A feed | kscmh | feed | feeds |
| xmeas_2 | D feed | kg/h | feed | feeds |
| xmeas_3 | E feed | kg/h | feed | feeds |
| xmeas_4 | A and C feed | kscmh | feed | feeds |
| xmeas_5 | Recycle flow | kscmh | compressor | compressor/recycle |
| xmeas_6 | Reactor feed rate | kscmh | reactor | reactor |
| xmeas_7 | Reactor pressure | kPa gauge | reactor | reactor |
| xmeas_8 | Reactor level | % | reactor | reactor |
| xmeas_9 | Reactor temperature | deg C | reactor | reactor |
| xmeas_10 | Purge rate | kscmh | purge | compressor/recycle |
| xmeas_11 | Product separator temperature | deg C | separator | condenser/separator |
| xmeas_12 | Product separator level | % | separator | condenser/separator |
| xmeas_13 | Product separator pressure | kPa gauge | separator | condenser/separator |
| xmeas_14 | Product separator underflow | m3/h | separator | condenser/separator |
| xmeas_15 | Stripper level | % | stripper | stripper |
| xmeas_16 | Stripper pressure | kPa gauge | stripper | stripper |
| xmeas_17 | Stripper underflow | m3/h | product | stripper |
| xmeas_18 | Stripper temperature | deg C | stripper | stripper |
| xmeas_19 | Stripper steam flow | kg/h | stripper | stripper |
| xmeas_20 | Compressor work | kW | compressor | compressor/recycle |
| xmeas_21 | Reactor cooling water outlet temperature | deg C | reactor | reactor |
| xmeas_22 | Separator cooling water outlet temperature | deg C | condenser | condenser/separator |
| xmeas_23 | Reactor feed analysis, component A | mol % | reactor | reactor |
| xmeas_24 | Reactor feed analysis, component B | mol % | reactor | reactor |
| xmeas_25 | Reactor feed analysis, component C | mol % | reactor | reactor |
| xmeas_26 | Reactor feed analysis, component D | mol % | reactor | reactor |
| xmeas_27 | Reactor feed analysis, component E | mol % | reactor | reactor |
| xmeas_28 | Reactor feed analysis, component F | mol % | reactor | reactor |
| xmeas_29 | Purge gas analysis, component A | mol % | purge | compressor/recycle |
| xmeas_30 | Purge gas analysis, component B | mol % | purge | compressor/recycle |
| xmeas_31 | Purge gas analysis, component C | mol % | purge | compressor/recycle |
| xmeas_32 | Purge gas analysis, component D | mol % | purge | compressor/recycle |
| xmeas_33 | Purge gas analysis, component E | mol % | purge | compressor/recycle |
| xmeas_34 | Purge gas analysis, component F | mol % | purge | compressor/recycle |
| xmeas_35 | Purge gas analysis, component G | mol % | purge | compressor/recycle |
| xmeas_36 | Purge gas analysis, component H | mol % | purge | compressor/recycle |
| xmeas_37 | Product analysis, component D | mol % | product | stripper |
| xmeas_38 | Product analysis, component E | mol % | product | stripper |
| xmeas_39 | Product analysis, component F | mol % | product | stripper |
| xmeas_40 | Product analysis, component G | mol % | product | stripper |
| xmeas_41 | Product analysis, component H | mol % | product | stripper |
| xmv_1 | D feed flow | % | feed | feeds |
| xmv_2 | E feed flow | % | feed | feeds |
| xmv_3 | A feed flow | % | feed | feeds |
| xmv_4 | A and C feed flow | % | feed | feeds |
| xmv_5 | Compressor recycle valve | % | compressor | compressor/recycle |
| xmv_6 | Purge valve | % | purge | compressor/recycle |
| xmv_7 | Separator pot liquid flow | % | separator | condenser/separator |
| xmv_8 | Stripper liquid product flow | % | product | stripper |
| xmv_9 | Stripper steam valve | % | stripper | stripper |
| xmv_10 | Reactor cooling water flow | % | reactor | reactor |
| xmv_11 | Condenser cooling water flow | % | condenser | condenser/separator |

Notes on the mapping
- Order trap: `xmeas_1..4` run A, D, E, A+C, but `xmv_1..4` run D, E, A, A+C.
- Stream 4 ("A and C feed") enters the stripper, but it is a fresh feed, so it is placed in feeds.
- `xmeas_22` is named "separator cooling water" by D&V. In the flowsheet the cooling water outlet after the reactor is on the condenser, so we place it with the condenser. This placement is our inference.
- `xmeas_5` (recycle flow) is placed with the compressor that drives it. `xmeas_17` and `xmeas_37-41` (product stream) are placed with the stripper whose bottom they come from.
- `xmv_5`: the recycle valve exists; its exact piping is not confirmed in our sources.

## Sources
- J. J. Downs and E. F. Vogel (1993), "A plant-wide industrial process control problem", Computers & Chemical Engineering 17(3), 245-255, via the variable tables reproduced at https://bibmon.readthedocs.io/en/latest/tutorial_tep.html
- Bathelt, Ricker and Jelali (2015), IFAC-PapersOnLine 48(8), 309-314 (flowsheet routing).
