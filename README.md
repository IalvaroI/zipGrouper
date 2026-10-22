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

# zipGrouperGeo.py — GeoJSON delivery-fee zones

`zipGrouperGeo.py` is a polygon-based companion to `zipGrouper.py`. Instead of
treating each ZIP as a single point, it uses real **US Census ZCTA polygons**,
so the extra granularity comes from the ZIP's actual shape.

> **Full usage guide:** [`docs/zipGrouperGeo.md`](docs/zipGrouperGeo.md) — install,
> configuration, fee-schedule editing, output reference, mapping, and library use.

## What it adds

| `zipGrouper.py` | `zipGrouperGeo.py` |
| --- | --- |
| ZIP = one pgeocode point | ZIP = real Census ZCTA polygon |
| One centroid-distance per ZIP | Nearest-edge, centroid, and farthest-edge distance per ZIP |
| Mileage bands only | Configurable **fee schedule** (flat fee + per-band minimum order + open-ended per-mile surcharge) |
| — | **Flat-fee override zones** — price a whole town (a circle around it) at one flat fee regardless of distance |
| Two CSVs | Two CSVs **plus** GeoJSON map layers |
| — | Flags ZIPs that **straddle a fee boundary** |

The default `main()` runs `example_fee_schedule()` for a placeholder store
(`Example Store`, origin ZIP `21701`, Maryland):

- 0–10 mi: free, $50 minimum
- 10.1–15 mi: $5.00, $50 minimum
- 15.1–20 mi: $10.00, $50 minimum
- 20.1–25 mi: $15.00, $75 minimum
- beyond 25 mi: $15.00 + $1/mile over 25, $75 minimum (e.g. 26 mi → $16.00)

No flat-fee override zones are set by default; `main()` includes a commented
example.

Edit the `FEE_SCHEDULE` and CONFIGURATION block in `main()` for another store or
fee structure. Set `ORIGIN_LATLON` to the store's exact coordinates to measure
from its real location instead of the ZIP centroid.

## Flat-fee override zones

Some stores price a whole town at one promotional flat fee regardless of how far
it is from the store. An `OverrideZone` is a circle — `radius_miles` around a
center point — applied **after** the mileage schedule, so it wins over the
distance bands. A ZIP joins the zone when its **centroid** is within the radius
(matching `primary_metric`). Configure them in `main()`:

```python
OVERRIDE_ZONES = (
    OverrideZone(
        name="Hagerstown",
        center_lat=town_lat,   # town_lat, town_lon = latlon_for_zip("21740")
        center_lon=town_lon,
        radius_miles=10.0,
        fee=4.99,
        min_order=50.0,
        label="Hagerstown flat $4.99",
    ),
)
```

Override ZIPs are tagged with an `override_zone` column in the CSV, dissolve into
their own GeoJSON fee zone (drawn in a distinct purple), and get their own circle
in the `*_fee_rings.geojson` overlay. When zones overlap, the first listed wins.

## How distances are computed

Polygons are reprojected to a store-centered **azimuthal-equidistant** CRS in
which the store sits at the origin `(0, 0)` and planar distance from the origin
equals true great-circle distance. Per ZIP this yields:

- `distance_nearest_miles` — closest edge ("any part of this ZIP is within X mi")
- `distance_centroid_miles` — representative middle of the ZIP (default fee basis)
- `distance_farthest_miles` — farthest corner ("the whole ZIP is within X mi")

`primary_metric` (`"centroid"` by default) chooses which distance drives the
headline fee tier; the other two are always reported, and any ZIP whose edges
fall in different tiers is marked `straddles_boundary = True`.

## Outputs

Written to the per-state folder (e.g. `MD/`), with file names prefixed by
`STORE_SLUG` (default `example_store_`):

| File | Contents |
| --- | --- |
| `*_zip_fees.csv` | per-ZIP fee table (all three distances, fee, min order, straddle flag, override zone) |
| `*_fees_by_tier.csv` | one row per fee tier with its ZIP roster |
| `*_zips.geojson` | per-ZIP polygons with fee properties + simplestyle colors |
| `*_fee_zones.geojson` | dissolved one-polygon-per-tier fee zones |
| `*_fee_rings.geojson` | concentric fee-boundary rings (reference overlay) |
| `*_zips_clipped.geojson` | ZIP polygons **clipped by the rings** — each large ZIP split into the exact slice that sits in each fee tier |

The GeoJSON files carry [simplestyle-spec](https://github.com/mapbox/simplestyle-spec)
properties (`fill`, `stroke`, `fill-opacity`), so dropping any of them onto
[geojson.io](https://geojson.io) renders the fee zones colored near-to-far
(green → red) with no extra styling.

## Dependencies and data source

`zipGrouperGeo.py` needs the geospatial extras listed under the
"GeoJSON support" section of `requirements.txt` (`geopandas`, `shapely`,
`pyproj`, `pyogrio`); `zipGrouper.py` itself does not. The ZIP polygons come
from the **US Census 2020 ZCTA cartographic boundary file**
(`cb_2020_us_zcta520_500k`, public domain), downloaded once and cached under
`~/.cache/zip_zcta/`. Point `POLYGON_SOURCE` at a local GeoJSON / shapefile /
`.zip` to use your own boundaries and skip the download. The ZIP → state / place
mapping still comes from GeoNames via pgeocode (attribution:
`Postal-code data source: GeoNames, https://www.geonames.org/`).

## Tests

`test_fees.py` covers the fee engine (band assignment, the open-ended per-mile
tier, rounding modes, validation) using only the standard library:

```
python3 test_fees.py
```
