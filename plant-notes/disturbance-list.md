# Disturbance list (IDV 1 to 20)

The 20 programmed disturbances of the Tennessee Eastman simulation, as Downs & Vogel describe them. This list says only what each disturbance is. A fault number in our data corresponds to the IDV number of the same value.

Types used: step (sudden change to a new constant value), random variation (a value that wanders randomly), slow drift (gradual change over time), sticking (a valve that sticks and moves unevenly), unknown (D&V give no description).

Stream numbers are the D&V stream numbers (see `process-description.md`).

| IDV | What it is | Type |
|---|---|---|
| 1 | A/C feed ratio changes while the B composition stays constant (stream 4) | step |
| 2 | B composition changes while the A/C ratio stays constant (stream 4) | step |
| 3 | D feed temperature (stream 2) | step |
| 4 | Reactor cooling water inlet temperature | step |
| 5 | Condenser cooling water inlet temperature | step |
| 6 | A feed loss (stream 1) | step |
| 7 | C header pressure loss, i.e. reduced availability (stream 4) | step |
| 8 | A, B and C composition (stream 4) | random variation |
| 9 | D feed temperature (stream 2) | random variation |
| 10 | C feed temperature (stream 4) | random variation |
| 11 | Reactor cooling water inlet temperature | random variation |
| 12 | Condenser cooling water inlet temperature | random variation |
| 13 | Reaction kinetics | slow drift |
| 14 | Reactor cooling water valve | sticking |
| 15 | Condenser cooling water valve | sticking |
| 16 | Unknown | unknown |
| 17 | Unknown | unknown |
| 18 | Unknown | unknown |
| 19 | Unknown | unknown |
| 20 | Unknown | unknown |

Remarks
- D&V name a unit or stream only where shown above (the reactor cooling water, condenser cooling water and the numbered streams). They do not assign IDV 16 to 20 to any equipment.
- The original simulation also has an IDV 21 (a stream 4 valve held at a fixed position). It is not in our data.
- The type of IDV 8 to 12 is "random variation" in the D&V table. Treat reports that add more detail on IDV 16 to 20 as unconfirmed.

## Sources
- J. J. Downs and E. F. Vogel (1993), "A plant-wide industrial process control problem", Computers & Chemical Engineering 17(3), 245-255, disturbance table, as reproduced in https://bibmon.readthedocs.io/en/latest/tutorial_tep.html and https://arxiv.org/html/2510.23817v1
