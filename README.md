## Acknowledgment
Portions of `zipGrouper.py` and this accompanying documentation were generated with assistance from an AI language models [Codex 5.4, Qwen3-Coder-Next]. Final review, validation, and responsibility for the code and documentation remain with the author.

# Dependency and License Notes
Author: Alvaro Carvajal
Last updated: April 23, 2026.

This file documents the libraries imported by `zipGrouper.py`, what each one is used
for, and where to find upstream source and license information.

## Direct Imports

| Import in `zipGrouper.py` | Package or module | Role in this project |
| --- | --- | --- |
| `logging` | Python standard library | Writes console and file trace logs. |
| `dataclasses.dataclass` | Python standard library | Defines immutable configuration objects, such as `ZoneSettings`. |
| `pathlib.Path` | Python standard library | Creates per-state output folders and builds output paths. |
| `typing.Optional`, `typing.Union` | Python standard library | Documents function arguments that can be `None` or accept multiple path types. |
| `numpy as np` | NumPy | Converts coordinates to radians, calculates distance-band indexes, and supports numeric array operations. |
| `pandas as pd` | pandas | Stores ZIP data in DataFrames, filters rows, groups zones, and writes CSV files. |
| `pgeocode` | pgeocode | Loads postal-code data through `pgeocode.Nominatim("us")`. |
| `sklearn.neighbors.BallTree` | scikit-learn | Performs fast haversine distance queries between the origin ZIP and state ZIP coordinates. |

## Library Source and License Links

| Library | Project source | License | License link | Notes |
| --- | --- | --- | --- | --- |
| Python standard library | https://github.com/python/cpython | Python Software Foundation License Version 2 | https://docs.python.org/3/license.html and https://github.com/python/cpython/blob/main/LICENSE | Used for `logging`, `dataclasses`, `pathlib`, and `typing`. |
| NumPy | https://github.com/numpy/numpy | BSD 3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt | Used for numeric coordinate conversion and zone math. |
| pandas | https://github.com/pandas-dev/pandas | BSD 3-Clause | https://github.com/pandas-dev/pandas/blob/main/LICENSE | Used for DataFrame transformations and CSV export. |
| pgeocode | https://github.com/symerio/pgeocode | BSD 3-Clause | https://github.com/symerio/pgeocode/blob/main/LICENSE | Used to access postal-code data and metadata. |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING | Only `sklearn.neighbors.BallTree` is used. |

## Postal-Code Data Source

The code calls `pgeocode.Nominatim("us")`. pgeocode documents its default data
sources as:

1. `https://download.geonames.org/export/zip/{country}.zip`
2. `https://symerio.github.io/postal-codes-data/data/geonames/{country}.txt`

For this script, `{country}` is `US`.

Relevant source and license links:

| Data source | Link | License notes |
| --- | --- | --- |
| GeoNames postal-code files | https://download.geonames.org/export/zip/ | GeoNames states the postal-code files are licensed under Creative Commons Attribution 4.0. |
| GeoNames home/about | https://www.geonames.org/about.html | GeoNames states its geographic database is available under a Creative Commons attribution license. |
| Symerio postal-codes-data mirror | https://github.com/symerio/postal-codes-data | The mirror states postal-code datasets are redistributed under the same Creative Commons Attribution 4.0 license as the original files. |

When distributing outputs derived from the GeoNames postal-code data, include
GeoNames attribution. A practical attribution line is:

> Postal-code data source: GeoNames, https://www.geonames.org/

## Runtime Traceability

`zipGrouper.py` writes outputs to a folder named after `STATE_CODE`. For
example, `STATE_CODE = "CA"` writes the trace log and CSV files under `CA/`.
By default, the detailed CSV writes ZIP cells in a spreadsheet-friendly text
format so apps like Excel display leading zeros instead of treating ZIPs as
numbers. The log records:

- the pgeocode module path and version;
- raw source columns and sample rows;
- row counts before and after cleaning;
- rows dropped for missing coordinates or invalid ZIP format;
- state filter counts;
- origin ZIP validation;
- distance query method and distance range;
- zone counts;
- CSV filenames and output row counts.

## Version Pinning

This project currently does not pin dependency versions. For reproducible runs,
add a `requirements.txt` with the versions installed in the working Python
environment, then regenerate the CSVs and keep the trace log with the output.
