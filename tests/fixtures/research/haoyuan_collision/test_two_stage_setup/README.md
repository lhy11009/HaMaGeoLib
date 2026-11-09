# Two-stage setup fixture

Snapshot taken on 2026-11-09 from:

```text
/mnt/lochy/ASPECT_DATA/Collision0/collision_setup39/C_PE5.00e+06_WLF2.0e-02_SA100.0_Kf3.00e-07_marine_SeaL1.50e+03_GN
```

`base_case/case.wb` and `summary.csv` are unmodified copies. `case.prm`
has tabs expanded to spaces and extra trailing blank lines removed; its
parameter values are retained.
`base_case/output/statistics` retains the source visualization rows and only
the time-step, time, and visualization-filename columns required by
`CASE_OPTIONS_TWOD`. Simulation grids, logs, and other outputs are omitted.
The tests copy the fixture to a temporary directory before invoking the real
case reader and `TwoStageRule`; they do not access the source directory.

`current.json` enables the two-stage rule with a convergence target of
0.04 m/year and earliest time of 14 million years. Its relative base-case path
is resolved to the temporary copy by the tests. Other options use rule defaults.
The rule sets `Output directory = output_1` and `Checkpointing/Resume time`
to the convergence-selected time `16207888.72701`. The earliest time remains
the cutoff for searching the summary, rather than the restart target.

The first qualifying summary row is graphical step 162:

- Time: 16207888.72701 years
- Subducting plate velocity: 0.0383175835013389 m/year
- Overriding plate velocity: -0.0022147244308143 m/year
- Convergence: 0.0405323079321532 m/year
- Trench position at 50 km depth: 2964843.75 m

`expected_velocity.json` freezes the prescribed velocity configuration derived
from these values and the default far-field distances. It includes the rule's
current numeric formatting: for example, the selected time is written as
`t0=1.62e+07` in the generated parameter file. This fixture tests configuration
generation, not execution of a second-stage ASPECT simulation.
