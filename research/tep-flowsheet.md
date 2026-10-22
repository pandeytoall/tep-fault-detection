# TEP flowsheet, stream table, variable map, fault list

Compiled 2026-10-22. Limits of this pass:
- I could NOT read the Downs & Vogel (1993) paper itself. I also could not read the Bathelt/Ricker/Jelali 2015 PDF or any other PDF.
- The fetch tool returns compressed PDFs as unreadable binary. Springer, MDPI, PMC, ScienceDirect and ResearchGate returned 403 or a login redirect.
- Everything below comes from secondary sources that reproduce Downs & Vogel (D&V) tables. Each is cited per item.
- The figure itself (stream routing) is not described in text anywhere I could read. Routing items are therefore marked CONFIRMED (2 or more sources), SINGLE-SOURCE, or UNVERIFIED.
- To close the gaps, someone with a PDF reader should open D&V 1993 Fig. 1 and Tables 1-8 and Bathelt 2015 (https://skoge.folk.ntnu.no/prost/proceedings/adchem2015/media/papers/0010.pdf).

Source keys (all URLs fetched this session):
- [BIB] bibmon tutorial, reproduces the D&V variable tables: https://bibmon.readthedocs.io/en/latest/tutorial_tep.html
- [HOV] Norwegian thesis process description based on D&V (Hovland, NTNU): https://skoge.folk.ntnu.no/diplom/diplom00/Hovland/Diplom/Tekstdok/Prosbeskrivelse.doc
- [RICKER] Ricker TE page: https://depts.washington.edu/control/LARRY/TE/download.html
- [SHAP] arXiv 2510.23817 HTML: https://arxiv.org/html/2510.23817v1
- [PMIM] arXiv 2007.10692 HTML: https://arxiv.org/html/2007.10692
- [SSCAN] arXiv 2208.08879 HTML, Appendix A: https://arxiv.org/html/2208.08879
- [DOM] arXiv 2308.11247 HTML (domain adaptation benchmark): https://arxiv.org/html/2308.11247
- [OKID] arXiv 2210.08538 HTML: https://arxiv.org/html/2210.08538v1
- [DVRS] Rieth Dataverse record: https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/6C3JR1
- [FDDB] FDDBenchmark: https://github.com/AIRI-Institute/fddbenchmark
- Search-result snippets only (not opened in full): Wu & Yu material-balance table (stream 5 = stripper overhead) https://scholars.lib.ntu.edu.tw/bitstream/123456789/67258/1/21.pdf ; Texas Tech thesis (reactor has 5 mixed feed streams) https://ttu-ir.tdl.org/bitstreams/32c7263b-4c51-4879-ba1e-a7a87a99dcf1/download ; EURASIP stream table https://asp-eurasipjournals.springeropen.com/articles/10.1186/1687-6180-2011-83/tables/2

## 1. Units

Five major units: reactor, product condenser, vapor-liquid separator, recycle compressor, product stripper.
- Sources: [SSCAN], [DOM], [PMIM], [SHAP]. The order "reactor, condenser, compressor, separator, stripper" is the one [SHAP] uses.
- Process: gaseous A, C, D, E plus inert B make liquid products G and H and byproduct F. Eight components A-H. [DOM], [SHAP], [BIB]

Flow in words, per [BIB], [SHAP], [DOM]:
- Reactor product goes through the condenser, then the separator.
- Non-condensables are recycled by a centrifugal compressor.
- Separator liquid goes to the stripper. Product leaves the stripper bottom.
- Vapor is purged from the separator to remove inert B and byproduct F.

Other items on the flowsheet, with the variables that touch them (variables per [BIB], [HOV]):

| Item | Evidence | Status |
|---|---|---|
| Purge valve (stream 9) | XMV(6), XMEAS(10) | CONFIRMED |
| Compressor recycle valve | XMV(5) (a valve around the compressor); XMEAS(20) compressor work | CONFIRMED that it exists. Exact piping (spill-back around the compressor) UNVERIFIED |
| Separator pot liquid flow valve (stream 10) | XMV(7), XMEAS(14) | CONFIRMED |
| Stripper liquid product valve (stream 11) | XMV(8), XMEAS(17) | CONFIRMED |
| Stripper steam valve | XMV(9), XMEAS(19) steam flow | CONFIRMED. "Reboiler" wording UNVERIFIED |
| Reactor cooling water (stream 12) | XMV(10), XMEAS(21) outlet temp | CONFIRMED, stream number SINGLE-SOURCE [HOV] |
| Condenser cooling water (stream 13) | XMV(11), XMEAS(22) | CONFIRMED, stream number SINGLE-SOURCE [HOV] |
| Reactor agitator | XMV(12), speed in rpm [HOV] | CONFIRMED in D&V list. It is NOT in the Rieth data (52 columns = 41 + 11) |

Likely wrong in drawings: [DVRS] says 52 process variables, so the 12th manipulated variable (agitator) is absent from the Rieth files.

## 2. Streams

D&V stream numbers. Routing for 5 and 7 was not found in text form and is therefore UNVERIFIED unless noted.

| # | Name / content | From | To | Type | Evidence |
|---|---|---|---|---|---|
| 1 | A feed (nearly pure A) | feed header | reactor feed mix (stream 6) | feed | CONFIRMED name [BIB], [HOV]; "nearly pure" per Texas Tech snippet. Mixing point (into stream 6) per [HOV] and the Texas Tech snippet |
| 2 | D feed | feed header | reactor feed mix | feed | CONFIRMED [BIB], [HOV] |
| 3 | E feed | feed header | reactor feed mix | feed | CONFIRMED [BIB], [HOV] |
| 4 | A and C feed (contains inert B) | feed header | stripper (stripping medium) | feed | Name CONFIRMED [BIB]. To-unit = stripper is SINGLE-SOURCE [HOV]; the ScienceDirect snippet agrees ("stripped with stream 4") but I could not open it (403) |
| 5 | Stripper overhead (gas) | stripper | reactor feed mix | recycle | Name SINGLE-SOURCE (Wu & Yu snippet). [HOV] says stripper gas joins the recycle. Texas Tech snippet lists stream 5 among the 5 mixed reactor feeds |
| 6 | Reactor feed (mix of 1, 2, 3, 5, 8) | feed mix point | reactor | gas | Name CONFIRMED (XMEAS(6) in [BIB], [HOV]). Composition of mix: SINGLE-SOURCE (Texas Tech snippet) |
| 7 | Reactor product / effluent | reactor | condenser | gas | UNVERIFIED (no source text). Inferred only from the unit sequence in [BIB] |
| 8 | Recycle gas | separator vapor, via compressor | reactor feed mix | recycle | CONFIRMED name [BIB], [HOV], [SSCAN]. Separator -> compressor -> reactor feed order per [BIB] narrative |
| 9 | Purge | separator vapor | out of plant | purge | CONFIRMED [BIB], [HOV] |
| 10 | Separator underflow (liquid) | separator | stripper | liquid | CONFIRMED [BIB], [HOV] |
| 11 | Stripper underflow = product (G, H) | stripper bottom | downstream refining (out of scope) | product | CONFIRMED [BIB], [HOV], [SSCAN] |
| 12 | Reactor cooling water | utility | reactor | utility | SINGLE-SOURCE [HOV] |
| 13 | Condenser cooling water | utility | condenser | utility | SINGLE-SOURCE [HOV] |
| (steam) | Stripper steam | utility | stripper | utility | Existence CONFIRMED (XMV(9), XMEAS(19)); stream number UNVERIFIED |

Where drawings often go wrong:
- Stream 4 (A and C feed) goes to the stripper, not directly to the reactor. [HOV]; consistent with [SHAP]: "feeds A, B, and C help remove unreacted components" in the stripper. Its material reaches the reactor through stream 5 (stripper overhead).
- The recycle is stream 8, not stream 5. Stream 5 is the stripper overhead. (Search-result summaries first suggested "stream 5 recycle"; the [BIB], [HOV], [SSCAN] tables all say recycle = 8.)
- The reactor feed has five inputs (1, 2, 3, 5, 8), not just fresh feeds plus recycle. SINGLE-SOURCE (Texas Tech snippet).
- Stream-3 clash: [RICKER] text says "step change in D feed temperature (stream 3)" for IDV(3), while [PMIM] and [BIB] number D feed as stream 2. [RICKER] looks like a slip in numbering (UNVERIFIED which is D&V's own).

## 3. Reactions

Four reactions, all irreversible and exothermic ([DOM], [SSCAN], [HOV]):
1. A(g) + C(g) + D(g) -> G(liq), product 1.
2. A(g) + C(g) + E(g) -> H(liq), product 2.
3. A(g) + E(g) -> F(liq), byproduct.
4. 3 D(g) -> 2 F(liq), byproduct.

Notes:
- Reactions 1-4 as written are in [DOM] (explicit equations). [HOV]'s reaction 3 text was garbled ("verify").
- D&V changed components, kinetics, flowsheet and conditions for confidentiality (search snippet of Reinartz et al. 2021). That claim is UNVERIFIED here.
- Rates depend on temperature; G has higher activation energy than H [HOV].

## 4. Variable map

Units and names from [BIB] (D&V Tables), cross-checked against [HOV] (same names, same units). "kscmh" = thousand standard cubic metres per hour. Column names in the Rieth files follow `xmeas_N` and `xmv_N` (per the project skill).

### Continuous measurements (xmeas_1..22, one value per 3-minute sample; noise added)

| xmeas | D&V name | Unit | Unit / stream |
|---|---|---|---|
| 1 | A feed | kscmh | stream 1 |
| 2 | D feed | kg/h | stream 2 |
| 3 | E feed | kg/h | stream 3 |
| 4 | A and C feed | kscmh | stream 4 |
| 5 | Recycle flow | kscmh | stream 8 |
| 6 | Reactor feed rate | kscmh | stream 6 |
| 7 | Reactor pressure | kPa gauge | reactor |
| 8 | Reactor level | % | reactor |
| 9 | Reactor temperature | deg C | reactor |
| 10 | Purge rate | kscmh | stream 9 |
| 11 | Product separator temperature | deg C | separator |
| 12 | Product separator level | % | separator |
| 13 | Product separator pressure | kPa gauge | separator |
| 14 | Product separator underflow | m3/h | stream 10 |
| 15 | Stripper level | % | stripper |
| 16 | Stripper pressure | kPa gauge | stripper |
| 17 | Stripper underflow | m3/h | stream 11 |
| 18 | Stripper temperature | deg C | stripper |
| 19 | Stripper steam flow | kg/h | stripper utility |
| 20 | Compressor work | kW | compressor |
| 21 | Reactor cooling water outlet temperature | deg C | reactor utility |
| 22 | Separator cooling water outlet temperature | deg C | condenser utility (D&V wording says "separator"; the cooling is on the condenser, per XMV(11) "condenser cooling water") |

Caveat on xmeas_22: the name "Separator cooling water outlet temp" is from [BIB]; [HOV] calls it "Condenser cooling water outlet temp". The outlet temperature of the condenser coolant is the intended meaning, but mapping to the condenser rests on [HOV] and the XMV(11) name only. SINGLE-SOURCE inference.

### Sampled analyzers (mole %)

| xmeas | Component | Stream | Sampling interval | Dead time |
|---|---|---|---|---|
| 23-28 | A, B, C, D, E, F | 6 (reactor feed) | 0.1 h | 0.1 h |
| 29-36 | A, B, C, D, E, F, G, H | 9 (purge gas) | 0.1 h | 0.1 h |
| 37-41 | D, E, F, G, H | 11 (product) | 0.25 h | 0.25 h |

- Intervals and dead times: [BIB] (table headers), [HOV] (same numbers). [OKID] confirms the counts: "14 sampled every 0.1 h, and 5 sampled every 0.25 h" (6 + 8 = 14).
- Derived, not from a source: at 3 min per sample, 0.1 h = 6 min = every 2nd sample and 0.25 h = 15 min = every 5th sample. The held values should repeat between updates. [BIB] says the original simulation uses "3, 6, and 15 minute" sampling, which matches. Dead time 0.1 h and 0.25 h = 2 and 5 samples. Check this against the Rieth data before relying on it.
- Stream 11 composition has no A, B, C: only D-H are measured.

### Manipulated variables (xmv_1..11 in Rieth; D&V has a 12th)

| xmv | D&V name | Unit/stream |
|---|---|---|
| 1 | D feed flow | stream 2 |
| 2 | E feed flow | stream 3 |
| 3 | A feed flow | stream 1 |
| 4 | A and C feed flow | stream 4 |
| 5 | Compressor recycle valve | compressor |
| 6 | Purge valve | stream 9 |
| 7 | Separator pot liquid flow | stream 10 |
| 8 | Stripper liquid product flow | stream 11 |
| 9 | Stripper steam valve | stripper utility |
| 10 | Reactor cooling water flow | stream 12 |
| 11 | Condenser cooling water flow | stream 13 |
| (12) | Agitator speed | reactor, not in Rieth data |

- Source: [BIB] (marks the XMV(1)-(3) order as "Corrected Order") and [HOV]. XMV units: [HOV] gives m3/h for flows and % for valves; mixed in [HOV] and not all rows trusted. D&V units per XMV: UNVERIFIED. Rieth/Ricker data expresses XMVs as percent of range (UNVERIFIED here).
- Trap: XMV order is D, E, A, A+C (xmv_1 = D feed), while XMEAS order is A, D, E, A+C (xmeas_1 = A feed). Easy to cross-wire a diagram.
- [BIB] and [HOV] both list the agitator as XMV(12). Whether Rieth dropped it because it is constant is UNVERIFIED (not documented in [DVRS]; the record only says 52 process variables).

## 5. Faults

Types as in D&V's disturbance table, taken from [SHAP], [PMIM] Table 5, [SSCAN], [BIB], [RICKER].

| IDV | Description | Type |
|---|---|---|
| 1 | A/C feed ratio, B composition constant (stream 4) | Step |
| 2 | B composition, A/C ratio constant (stream 4) | Step |
| 3 | D feed temperature (stream 2) | Step |
| 4 | Reactor cooling water inlet temperature | Step |
| 5 | Condenser cooling water inlet temperature | Step |
| 6 | A feed loss (stream 1) | Step |
| 7 | C header pressure loss, reduced availability (stream 4) | Step |
| 8 | A, B, C feed composition (stream 4) | Random variation |
| 9 | D feed temperature (stream 2) | Random variation |
| 10 | C feed temperature (stream 4) | Random variation |
| 11 | Reactor cooling water inlet temperature | Random variation |
| 12 | Condenser cooling water inlet temperature | Random variation |
| 13 | Reaction kinetics | Slow drift |
| 14 | Reactor cooling water valve | Sticking |
| 15 | Condenser cooling water valve | Sticking |
| 16-20 | Unknown | Unknown |
| (21) | Stream 4 valve fixed at steady-state position [PMIM] | Constant position; not in Rieth |

### Grouping check

Confirmed: step 1-7, random variation 8-12, slow drift 13, sticking 14-15, unknown 16-20.
- [SHAP] states this exactly; [BIB], [PMIM] Table 5 and [SSCAN] agree.
- Caveats on 16-20. [PMIM] annotates 16, 17, 18 as "heat transfer deviation in stripper / reactor / condenser" (likely from later work, not D&V; origin UNVERIFIED). [SSCAN] extraction shows IDV(20) with a random-variation type (garbled; UNVERIFIED). The D&V label for all five is "unknown". Treat 16-20 as unknown.
- Faults 3, 9, 15 are the "hard" ones: [SSCAN] (9 and 15 undetected by all unsupervised models); a snippet of arXiv 2101.06993 says there is no observable change in mean or variance for 3, 9, 15 (snippet only).

### Rieth faultNumber vs IDV

- [DVRS] says faultNumber is "0 for fault-free, 1 to 20 for faulty". It does not mention IDV or Downs & Vogel anywhere I could read. The record points to Rieth et al. (2017) "Issues and Advances in Anomaly Detection Evaluation for Joint Human-Automated Systems".
- Indirect support for one-to-one: a paper using the Rieth data (snippet of arXiv 2101.06993) lists faultNumber 3 as the D feed temperature disturbance, consistent with IDV(3). [SSCAN] says "TEPRieth has 20 faults", while the Ricker-model dataset adds 8 more (faults 21-28), consistent with Rieth = IDV(1)-(20) with IDV(21) excluded. [FDDB] lists 20 fault states.
- Verdict: one-to-one mapping to IDV(1)-IDV(20) is LIKELY but NOT CONFIRMED from Rieth's own documentation. UNVERIFIED. Suggested check: compare Rieth faulty-training means against known fault signatures, e.g. fault 6 (A feed loss) should drive xmeas_1 to about 0 within the first hours, and fault 4 should move xmeas_21.

### Dataset-version notes (for comparability)

Rieth data was generated in MATLAB/Simulink from the Ricker/Bathelt revision lineage? UNVERIFIED (not stated in [DVRS]). Versions: original Braatz FORTRAN-based data (d00-d21) vs Rieth vs the Reinartz 2021 extended set (28 faults, six operating modes, https://data.dtu.dk/articles/dataset/Tennessee_Eastman_Reference_Data_for_Fault-Detection_and_Decision_Support_Systems/13385936). Fault 21 exists only in the original set.

## Open items for the orchestrator

1. Read D&V 1993 Fig. 1 and the stream table to confirm streams 5, 7, 12, 13 and the steam routing (all UNVERIFIED or single-source above).
2. Confirm the Rieth one-to-one mapping empirically, or via Rieth et al. 2017 (not found online).
3. Confirm XMV units, and whether xmeas_23..41 repeat held values between samples in the Rieth files (check on a normal run).
4. Confirm the xmeas_22 name wording in D&V ("separator" vs "condenser" cooling water).

## Stream routing verified from the flowsheet figure (added 2026-10-22)
Source: Bathelt, Ricker & Jelali (2015), "Revision of the Tennessee Eastman Process Model", IFAC-PapersOnLine 48(8):309-314, Fig. 3 (P&ID of the revised model), author copy: https://folk.ntnu.no/skoge/prost/proceedings/adchem2015/media/papers/0010.pdf (read from the rendered figure on p. 3).
| Stream | From | To |
|---|---|---|
| 1 A feed, 2 D feed, 3 E feed | battery limit | join the reactor feed header |
| 4 C feed (D&V: "A and C feed") | battery limit | **stripper** (not the reactor) |
| 5 stripper overhead | stripper top | joins the recycle line to the reactor feed |
| 6 reactor feed | header (1 + 2 + 3 + 5 + 8) | reactor |
| 7 reactor effluent | reactor top | condenser |
| (unnumbered) condenser outlet | condenser | vapor-liquid separator |
| 8 recycle | separator vapor via the compressor (recycle valve bypasses it) | reactor feed header |
| 9 purge | separator vapor line, after the compressor branch | battery limit |
| 10 separator underflow | separator bottom (pump) | stripper |
| 11 product | stripper bottom (pump) | battery limit |
| 12 reactor cooling water | supply | return (reactor cooling coil) |
| 13 condenser cooling water | supply | return (condenser) |
| steam / condensate | supply | stripper reboiler |
- The reactor has an agitator (speed control; D&V XMV 12, not in the Rieth files).
- These routings settle streams 5, 7, 12 and 13, which were UNVERIFIED above.
- xmeas_22 (D&V name: "separator cooling water outlet temperature"): in Fig. 3 the only cooling-water outlet temperature downstream of the reactor sits on the **condenser** return (stream 13). Treat xmeas_22 as belonging to the condenser. Keep D&V's name; note it as an inference from the figure.
- The figure is IFAC copyright: we draw our own diagram, without copying it or its instrument tags.
