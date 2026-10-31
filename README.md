# HaMaGeo: A Geodynamic Research Toolkit

**Authors:** Haoyuan Li (hylli@ucdavis.edu), Magali Billen (mibillen@ucdavis.edu) 
**Version:** 0.1.0  
**License:** MIT  

hamageolib is a Python package designed to provide tools and models for geodynamic research, with a focus on simulating mantle convection, plate tectonics, and related geodynamic processes. Developed by researchers for researchers, HaMaGeo offers essential utilities, modeling capabilities, and visualization tools for geodynamics.

---

## Features

- **Core Functionality (`hamageolib_core`)**: The core module provides foundational classes and functions for geodynamic modeling and simulations, such as mantle convection and plate tectonics, designed to support a wide range of research in geodynamics.
  
- **Utility Functions (`hamageolib_utils`)**: Utility tools include parameter parsing and file management features that facilitate data processing, loading, and saving for geodynamic models. For example, the `parse_parameters_to_dict` function converts parameter files to dictionaries for easier manipulation.
  
- **Visualization Tools (`hamageolib_viz`)**: This module will include visualization utilities for interpreting simulation results, creating insightful plots of model outputs, and visualizing geological structures, assisting in the analysis and presentation of research findings.
---

## Prerequisites

Anaconda is required to manage the Python environment for this package.

To install conda environment for hamageolib

   conda env create --file environment.yml

To activate in terminal

   conda activate hmgeolib

To activate in Jupyter notebooks, select "hmgeolib" as kernel

## Installation

### Anaconda version

1. **hamageolib is currently in development. To install the latest version from anaconda**:

```bash
conda install lhy11009::hamageolib
```

2. **After this, you’ll be able to import and use hamageolib**:

```python
import hamageolib
```

### Development Version

If you’d like to work with the development version, you can clone the GitHub repository and add it to your `sys.path` in a script. This allows you to access the latest updates directly from the repository.

1. **Clone the repository**:
   
```bash
git clone https://github.com/lhy11009/HaMaGeoLib
```

2. **Add to sys.path in your script**:

In your Python script, include the following lines to add the cloned directory to your path:

```python
import sys
sys.path.insert(0, "path/to/HaMaGeoLib")  # Replace with the actual path to hamageolib
```

3. **After this, you’ll be able to import and use hamageolib as usual**:

```python
import hamageolib
```
---

4. **Sync the big test folder from cloud service**:

The `big_tests` folder contains large test files that are **not synced to GitHub** due to their size. Instead, these files are stored in a cloud-based storage solution.
These typically include files that are case outputs.

Note for myself: use the lihaoyuan81@gmail.com space for this purpose.


## Case registry

`CaseManager`, migrated from `gdmate.aspect.case_manager`, manages a CSV table
with unique `case_id` values, absolute `case_path` values, and optional metadata
columns. It requires pandas, included in `environment.yml`.

```python
from hamageolib.utils.case_manager import CaseManager

cases = CaseManager("cases.csv")  # Load an existing table or start an empty one.
index, row = cases.add_case("runs/case_a")
cases.save()
```

Existing gdmate CSV tables can be loaded directly. Relative case paths resolve
against the current working directory. Changes remain in memory until `save()`.
The original API is preserved, including `CaseManager.TabelRowError`.

## Contributing

We welcome contributions to **hamageolib**! If you’re interested in adding new features or improving existing functionality, please follow the workflow below:

1. **Implement Your Feature**:
   - Develop your Python script in the appropriate submodule (e.g., `core`, `utils`, or `visualization`).
   - Ensure your code is modular, well-documented, and follows the project’s coding standards.

2. **Create Tests**:
   - Add tests to validate your new feature. Place tests in the `tests` directory following a similar structure to the main package.
   - Use `pytest` to organize and run tests efficiently.
   - Ensure all tests pass before submitting your contribution.

3. **Document with Jupyter Notebooks**:
   - Create a Jupyter notebook under `notebooks/examples` to demonstrate how to use your new feature.
   - Create a Jupyter notebook under `notebooks/experiments` to demonstrate how to work on some developing features.
   - Create a Jupyter notebook under `notebooks/tutorials` to demonstrate how to follow a workflow used in research.
   - Include explanations, code examples, and visualizations as necessary to guide users through the functionality.
   - Keep the notebook focused and clear, aiming to showcase the feature’s usage and applications.
   - As an additional step, "Clear All Outputs" in the notebook before and PR to substantially save space

4. **Submit a Pull Request**:
   - After implementing, testing, and documenting your feature, submit a pull request to the main repository on GitHub.
   - Provide a clear summary of your changes and link to relevant issues if applicable.

Following this workflow will help maintain the quality and usability of **hamageolib**. Thank you for your contributions!


### Parse an ASPECT case

`CaseASPECT` reuses the parameter and log parsers in gdmate's `dev` branch.
Install that version of gdmate in the same Python environment before use.

```python
from hamageolib.utils.case_aspect import CaseASPECT

case = CaseASPECT("/path/to/case")
# For a different output subdirectory:
case = CaseASPECT("/path/to/case", output_dir="output-custom")
print(case.parameters)
print(case.world_builder)
print(case.timesteps)
```

Construction reads `original.prm`, `original.wb`, and `log.txt` from the
specified subdirectory of the case folder (default: `output`). Parameters
are parsed with gdmate; optional World Builder JSON is loaded as a dictionary.
If `original.wb` is absent, `world_builder` is `None`.
The attributes `timesteps`, `graphical_outputs`, and `timing_blocks` contain
gdmate log records. No files or directories are created. Missing inputs or
malformed files raise reader exceptions. The log parser requires ASPECT
major version 3.

`existing_snapshots` contains the sorted snapshot numbers currently present in
`output_dir/solution`; `n_existing_snapshots` gives their count. A snapshot
requires a readable `solution-NNNNN.pvtu` declaring at least one VTU piece,
with every referenced `.vtu` file present. Missing pieces, orphan VTU files,
and incomplete or malformed PVTU files are excluded. A missing solution
directory gives an empty list and a count of zero. This checks file presence,
not VTU contents, when the case is constructed. `graphical_outputs` continues
to describe outputs reported in the log, even if their files are absent.

### ASPECT configuration utilities

ASPECT configuration utilities previously imported from `gdmate.aspect`
are now provided by `hamageolib.utils.aspect`:

```python
from hamageolib.utils.aspect.io import parse_parameters_to_dict
from hamageolib.utils.aspect.config_engine import Rule, RuleEngine
from hamageolib.utils.aspect.builtin_rules import CasePathRule, InitialStepRule
from hamageolib.utils.aspect.prm_wb_utils import find_WB_feature_by_name
from hamageolib.utils.aspect.table import DepthAverageTable
```

The `io`, `config_engine`, `builtin_rules`, `prm_wb_utils`, and `table`
modules were migrated from gdmate's `haoyuan_main` branch at commit
`a1f19954d230b0e96de7db864af5345fdbf5e753`, preserving their APIs and
source license (included in `hamageolib/utils/aspect/LICENSE`). Package,
test, and notebook imports use these local utilities. The existing
`hamageolib.utils.dealii_param_parser` API is unchanged.

`CaseASPECT` still depends on `gdmate.aspect_tools` from gdmate's `dev`
branch; that newer API is not part of this migration.

### Plan solution-file removal at a time interval

```python
from hamageolib.utils.case_aspect import CaseASPECT
from hamageolib.utils.aspect.solution_cleanup import write_solution_cleanup_script

case = CaseASPECT("/path/to/case")
plan = write_solution_cleanup_script(
    case, time_interval_years=1e6, script_path="remove_solutions.sh"
)
print(plan["kept_snapshots"])
print(plan["removed_snapshots"])
print(plan["files_to_remove"])
```

The function rescans the solution directory and matches complete snapshots to
log-recorded times. It keeps the nearest snapshot to each target `0`, `1e6`,
`2e6`, etc., through the last available time. Ties choose the earlier time,
then the smaller snapshot number. A snapshot can serve several targets; the
last snapshot is not automatically kept. Log times in seconds are converted
using 365.25 days per year. The interval must be finite and positive.

Only a shell script is created. Review it, then run it yourself if desired:

```sh
sh remove_solutions.sh
```

Each command uses an absolute, shell-quoted filename and `rm --`, without
wildcards. The script is not executed by Python and an existing script is
not overwritten. Its parent directory must already exist. Only complete,
log-matched snapshots with an unambiguous time are eligible for removal;
incomplete snapshots, unrecorded files, orphan VTU files, and pieces shared
with preserved snapshots are left alone. Only PVTU and referenced VTU files
are listed; PVD/Visit indexes and other output files are not rewritten.
The returned plan and script describe files at generation time.

### Prescribed-velocity ParaView view

The final block of `notebooks/research/prescribe_velocity/PrescribeVelocity.ipynb`
reproduces the spherical prescribed-velocity reference view. Set
`do_prescribed_velocity_visualization = True`, select the case directory and
saved time, and run that block (the case-creation cell is not required).
`PVPYTHON` may specify a ParaView Python executable; otherwise `pvpython` is
used. The block writes a PNG and reusable PVSM under `case/img/pv_outputs`.

The workflow is `scripts/paraview_scripts/PrescribedVelocity/prescribed_velocity.py`.
It loads the solution PVD, thresholds point `plate` from 0.8 to 1.0, and shows
a green slab at opacity 0.1, white velocity arrows for the domain, and blue
arrows for the slab. Default sample limits are 5000 and 500; both glyph layers
use velocity scaling of 5e6. The camera/background reproduce the reference
`visual.pvsm` at 1024 x 782 and time zero. Spatial glyph sampling can produce
small differences in individual arrow placement.

The notebook-generated runner loads `base.py` with `runpy` and passes its
namespace to the separate workflow. It does not concatenate either file.
The source main block retains `VISUALIZATION_DIRECTORY`, `IMAGE_DIRECTORY`,
and `PLOT_TIME` placeholders for existing template workflows. No reference
state file is needed to run the new script. ParaView 6.1 was used for validation.

### Generate a spherical trench-retreat case

The final block in `notebooks/research/prescribe_velocity/PrescribeVelocity.ipynb`
exposes `trench_retreat_m` in metres (default `100e3`), a source case and a
new destination. It defaults to `_5 → _6`, following the independent domain
modification block (`_2 → _5`). Enable `do_generate_retreat_case` after choosing
paths. Existing destinations are refused. The block is independent of case
creation, plotting and domain modification, provided its source already exists.

Python API:

```python
from hamageolib.research.prescribe_vecloity.trench_retreat import generate_retreat_case

report = generate_retreat_case(
    "/path/to/source_case", "/path/to/new_case", trench_retreat_m=50e3
)
```

Command line, with HaMaGeoLib available on the Python path:

```sh
python -m hamageolib.research.prescribe_vecloity.trench_retreat     /path/to/source_case /path/to/new_case --trench-retreat-m 100000
```

By default the generator copies `case.prm` exactly and writes a retreated `case.wb`,
`geometry.npz`, and `geometry_report.json`. It creates no simulation output,
runs no solver, and refuses existing destinations. It uses NumPy/SciPy and a
Python port of WorldBuilder's Bezier sampling, without a compiled helper.
The sampler's source attribution and LGPL license texts are included beside
`_trench_bezier.py`.

Supported sources have one east-dipping, four-segment subducting plate with
bend/straight/unbend/horizontal segments, equal bend lengths, constant
thickness, no section overrides, and spherical `begin segment` coordinates.
The final segment stays anchored; two 20 km overlap collars connect the
retreated upper slab to the original deep geometry. The physical retreat is
measured along spherical arcs, with end-bearing adjustments to meet the fixed
anchor. Source material, composition and velocity models remain unchanged.
An existing retreated two-feature case cannot be used recursively as a source.

50 km and 100 km retreats were checked against direct WorldBuilder queries.
The 100 km result reproduces `_3` within 0.1 mm in the sampled construction
geometry. These checks do not establish validity for arbitrary distances:
inspect the diagnostic geometry and validate WorldBuilder fields, mesh and
ASPECT solver convergence before using another retreat in production.
The report contains geometric fit diagnostics, not solver results.

#### Longitude options for trench-retreat cases

The notebook's "Modify case domain and position" block exposes:

```python
chunk_min_longitude = None   # Inherit source, currently 0 degrees.
chunk_max_longitude = None   # Inherit source, currently 25 degrees.
longitude_shift_deg = 0.0    # No rotation by default.
```

For example, set `chunk_max_longitude = 35.0` and
`longitude_shift_deg = 5.0` to move the complete slab 5 degrees east inside
a wider domain. The same keyword arguments are supported by
`generate_retreat_case`; CLI equivalents are `--chunk-min-longitude`,
`--chunk-max-longitude`, and `--longitude-shift-deg`.

The generator reads parameters through gdmate's public
`parse_parameters_to_dict` interface. Bounds overrides update the corresponding
values under Geometry model / Chunk, then use `save_parameters_from_dict` to
regenerate the parameter file. Other parameter values are retained; comments,
spacing and original line endings are not preserved in regenerated files.
With unchanged bounds, the original parameter file is still copied byte-for-byte.
None inherits the source. Both functions come from `gdmate.aspect_tools`, which
must be available in the Python environment. A shift rotates both WorldBuilder features, their dip points,
the cross section, and all global Cartesian geometry diagnostics together.
The deep anchor moves with the rotated reference, while its relative geometry,
depth, prescribed speed, retreat distance and section shapes are preserved.
Local 2-D profile coordinates do not rotate. Longitudes are not wrapped at
the dateline. Source files and existing cases remain unchanged.

New bounds and shifts must be finite and bounds must be ordered. With changed
bounds or a shift, sampled physical slab geometry (including the deep tip)
must fit the domain; enlarge the eastern boundary when shifting east. This
is a sampled check, not a full volumetric containment proof. Latitude, radial
extent and mesh repetitions are inherited. Increasing longitude span without
increasing Longitude repetitions makes the base angular resolution coarser.

The combined generator API remains backwards compatible and is regression-
tested against frozen pre-change 100 km and 300 km outputs using the original
notebook source case. The notebook now separates domain changes and retreat
into independent steps, described below.

#### Keep the original trench without retreat

Set `trench_retreat_m=None` in the Python API or notebook block, or pass
`--trench-retreat-m none` on the CLI. This retains the source's original
single slab feature and all segment definitions. No connector fit or overlap
features are created. Longitude shifts and chunk bounds overrides remain
available. With no longitude shift, `case.wb` is copied byte-for-byte;
unchanged bounds also preserve `case.prm` byte-for-byte. Changed bounds still
use gdmate's standard writer. Numeric zero remains invalid.

The geometry report records `trench_retreat_m: null`, `actual_retreat_m: 0.0`,
and `connector_fit_performed: false`. Geometry diagnostics contain identical
old/new trench positions (or their identical shifted positions), allowing
existing diagnostic consumers to read the same array keys. Source geometry
validation and refusal to overwrite existing destinations are unchanged.

#### Separate domain modification from trench retreat

`PrescribeVelocity.ipynb` has two independent sections:

1. **Modify case domain and position:** `do_modify_case_domain`, source/destination
   paths, longitude bounds and longitude shift. Current example: `_2 → _5`,
   bounds −5° to 35°, zero shift. This never fits the slab or adds overlaps.
2. **Generate a trench-retreat case:** `do_generate_retreat_case`, separate
   source/destination paths and `trench_retreat_m`. Current example: `_5 → _6`,
   100 km, disabled until enabled. Domain and placement are inherited exactly.

Both sections may run independently. To chain them, create the domain case
first and select it as the retreat source. Choose fresh destination paths.
No simulation outputs are copied and ASPECT is never launched by these blocks.

```python
from hamageolib.research.prescribe_vecloity.case_domain import modify_case_domain
from hamageolib.research.prescribe_vecloity.trench_retreat import generate_retreat_case

modify_case_domain("/path/to/source", "/path/to/domain_case",
                   chunk_max_longitude=35, longitude_shift_deg=5)
generate_retreat_case("/path/to/domain_case", "/path/to/retreated_case",
                      trench_retreat_m=100e3)
```

`modify_case_domain` retains all existing segment and section definitions.
It supports original four-segment slabs and already-retreated cases with
matching source `geometry.npz` diagnostics. Missing diagnostics for a retreated
or sectioned source are rejected rather than refitting it. The known Cartesian
arrays are rotated; local profile coordinates remain unchanged. Geometry
reports retain the original retreat distance and accumulate longitude shifts.
`domain_report.json` records the current operation and its source-relative shift.
Its return value includes `slab_refit_performed=False`.

Without a source geometry archive, original cases are sampled geometrically
without optimization. The diagnostics must correspond to the source WorldBuilder
file; checks use sampled physical geometry, not a full volume-containment proof.
A retreated two-feature case may be moved with `modify_case_domain`, but positive
`generate_retreat_case` still requires the original-style single-feature slab.
It does not recursively retreat an already-retreated case.

The domain CLI is `python -m hamageolib.research.prescribe_vecloity.case_domain`,
with source/destination arguments and the same `--chunk-min-longitude`,
`--chunk-max-longitude` and `--longitude-shift-deg` options. The existing combined
`generate_retreat_case` options and `trench_retreat_m=None` remain available for
backwards compatibility.
