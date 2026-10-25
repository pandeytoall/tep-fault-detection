# Process description: the Tennessee Eastman plant

A simulated chemical plant (Downs & Vogel, 1993). It turns gaseous reactants into two liquid products. The simulation is public and was disguised by its authors: the chemicals are labelled A to H, not real names.

## Components and reactions
- A, C, D and E are gaseous reactants. B is an inert that only passes through. F is an unwanted byproduct. G and H are the liquid products.
- Four reactions, all irreversible and heat-releasing:
  1. A + C + D gives G.
  2. A + C + E gives H.
  3. A + E gives F.
  4. Three D give two F.
- Reaction speed rises with temperature. The plant has to remove the reaction heat, which is the job of the reactor cooling water.

## The five scoring units
The assistant reports the fault location as one of these five.

1. **Feeds.** Fresh A (stream 1), D (stream 2), E (stream 3) and the "A and C" feed (stream 4, which also carries some inert B). Streams 1 to 3 join the reactor feed header. Stream 4 does not go to the reactor directly: it enters the stripper and its material reaches the reactor through the stripper overhead.
2. **Reactor.** A single vessel where the gases react to liquid products. Its feed (stream 6) is the mix of streams 1, 2, 3, 5 (stripper overhead) and 8 (recycle). Cooling water (stream 12) removes the reaction heat. An agitator exists in the original design, but it is not among our 52 readings.
3. **Condenser and separator.** The reactor effluent (stream 7) is cooled in the condenser (cooling water, stream 13) and flows to the vapor-liquid separator. There the vapor and liquid split. Liquid leaves the bottom (stream 10) towards the stripper.
4. **Compressor and recycle.** Vapor from the separator is pushed back to the reactor feed header by a centrifugal compressor (stream 8, the recycle). A recycle valve is fitted around the compressor. A small part of the vapor is purged out of the plant (stream 9) so that inert B and byproduct F do not build up.
5. **Stripper.** Separator liquid enters the stripper. Steam heats it, and feed stream 4 strips out unreacted material. Gas from the top (stream 5) returns to the reactor feed. The liquid from the bottom is the product (stream 11, mostly G and H), which leaves the plant for refining that is not simulated.

## Stream sequence in words
fresh feeds 1, 2, 3 -> reactor feed header -> reactor -> condenser -> separator.
Separator vapor -> compressor -> back to the header (recycle); some of it leaves as purge.
Separator liquid -> stripper -> product. Stripper overhead -> back to the header. Feed 4 -> stripper.

## Control, in plain words
- The plant is not run open loop. Controllers move 11 final elements (valves and flow set-points: the four feeds, compressor recycle valve, purge valve, separator liquid flow, stripper product flow, stripper steam valve, reactor cooling water, condenser cooling water). These are the xmv readings.
- Their job is to hold levels, pressures and temperatures steady and to keep the production rate and product mix near target.
- Because the controllers react, a disturbance can show up in readings other than the one it started in, and a controller can hide it for a while. The detailed loop pairing is not described in these notes.

## Measurements
41 measured readings and 11 manipulated variables are sampled every 3 minutes. See `tag-list.md`. Gas analyzers on the reactor feed, purge and product are sampled less often than the rest.

## Not covered here
Exact piping details of the compressor recycle valve, and the exact wording of some D&V names, are noted in `tag-list.md`.

## Sources
- J. J. Downs and E. F. Vogel (1993), "A plant-wide industrial process control problem", Computers & Chemical Engineering 17(3), 245-255.
- Bathelt, Ricker and Jelali (2015), "Revision of the Tennessee Eastman Process Model", IFAC-PapersOnLine 48(8), 309-314 (stream routing).
