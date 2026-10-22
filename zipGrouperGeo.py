"""
Build delivery-fee zones for a single store using real ZIP-code polygons.

This is the GeoJSON-powered companion to ``zipGrouper.py``. Where the original
script treats every ZIP code as a single point (the pgeocode centroid) and bins
it into one mileage band, this script uses the actual US Census ZIP Code
Tabulation Area (ZCTA) *polygons*. That polygon boundary is where the extra
granularity comes from:

* A ZIP code covers area, not a dot. For each ZIP this script measures three
  distances from the store: the NEAREST edge, the area CENTROID, and the
  FARTHEST edge. A large ZIP that straddles a fee boundary (say, the 25-mile
  line) is flagged instead of being silently assigned by one centroid point.
* The fee schedule is a configurable engine, not a fixed set of labels. It
  supports flat-fee bands with per-band minimum orders, plus an open-ended
  final tier that adds a per-mile surcharge (e.g. "$24.99 plus $2 per mile
  beyond 25 miles").
* Outputs are written as both CSV (for spreadsheets) and GeoJSON (for maps).
  The GeoJSON includes per-ZIP polygons, the dissolved fee zones, the concentric
  fee-boundary rings, and a ring-clipped layer that splits each ZIP polygon into
  the exact slices that fall in each fee tier.

The default configuration in ``main()`` runs an example schedule for a
placeholder store. Change the CONFIGURATION block to run it for your own store,
state, or fee schedule.

Data sources and attribution
-----------------------------
* ZIP polygons: US Census Bureau, 2020 ZCTA cartographic boundary file
  (cb_2020_us_zcta520_500k). Public domain.
* ZIP -> state / place / point mapping: GeoNames via pgeocode
  ("Postal-code data source: GeoNames, https://www.geonames.org/").

Pure helpers and the fee engine at the top of this file depend only on the
standard library, so they can be imported and unit-tested without the heavy
geospatial stack installed (see ``test_fees.py``). The data-loading and
geometry functions import pandas / geopandas / pgeocode lazily, and reuse the
trusted loading and logging helpers from ``zipGrouper.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.request import urlopen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The international mile is exactly 1609.344 metres.
METERS_PER_MILE = 1609.344

# Default ZIP-polygon source: the US Census 2020 ZCTA cartographic boundary
# file. ZCTAs are built from the decennial census, so 2020 is the current
# vintage (there are no 2021+ ZCTA cartographic files). The "500k" file is the
# generalized version, which is plenty accurate for delivery zones and far
# smaller than the full-resolution TIGER/Line file.
CENSUS_ZCTA_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip"
)

# Where downloaded ZCTA archives are cached so repeat runs do not re-download.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "zip_zcta"

# Field names that have held the 5-digit ZCTA code across Census vintages and
# common third-party exports. The first one present in the data wins.
ZCTA_FIELD_CANDIDATES = (
    "ZCTA5CE20",
    "ZCTA5CE10",
    "ZCTA5CE",
    "GEOID20",
    "GEOID10",
    "GEOID",
    "ZCTA5",
    "ZCTA",
    "ZIPCODE",
    "ZIP",
    "zip_code",
    "postal_code",
)

# CRS used only for accurate area: NAD83 / Conus Albers is equal-area, so a
# polygon's ``.area`` there converts cleanly to square miles. (The
# store-centered azimuthal-equidistant CRS used for distances is NOT equal-area,
# which is why area is computed separately.)
EQUAL_AREA_CRS = "EPSG:5070"
WGS84 = "EPSG:4326"

# Colour ramp for map styling, near (green) to far (red). Indexed by tier order
# and clamped at the ends, so any number of tiers renders sensibly.
TIER_COLORS = (
    "#1a9850",  # 0  free / nearest
    "#66bd63",
    "#a6d96a",
    "#fee08b",
    "#fdae61",
    "#f46d43",
    "#d73027",  # farthest
)

# Flat-fee override zones (e.g. a promotional zone around a town) are not part
# of the near->far distance ramp, so they get one distinct colour that stands
# out against it on a map.
OVERRIDE_COLOR = "#542788"  # purple

PathInput = Union[str, Path]

# Module logger. Shares the "zip_zones" name with zipGrouper.py so that
# zipGrouper.setup_logging() configures both scripts' output identically.
LOGGER = logging.getLogger("zip_zones")

# Tolerance (in miles) for float comparisons at band boundaries, so a distance
# of exactly 25.0 lands in the "up to 25" tier rather than the open tier.
_BOUNDARY_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Pure helpers (standard library only)
# ---------------------------------------------------------------------------


def normalize_zip(postal_code: object) -> str:
    """
    Return a five-character ZIP-code string.

    Mirrors ``zipGrouper.normalize_zip`` but is duplicated here so the fee
    engine and ZIP formatting stay importable without the heavy stack.
    """

    postal_code_text = str(postal_code).strip()
    if postal_code_text.endswith(".0"):
        postal_code_text = postal_code_text[:-2]
    return postal_code_text.zfill(5)


def normalize_state_code(state_code: object) -> str:
    """Return a two-letter uppercase state code, e.g. ``" nj "`` -> ``"NJ"``."""

    return str(state_code).strip().upper()


def format_miles(value: float) -> str:
    """
    Format mileage for a clean label.

    Whole numbers drop the decimal; values like 10.1 keep one place.
    """

    return f"{value:.1f}".rstrip("0").rstrip(".")


def format_money(value: float) -> str:
    """Format a dollar amount as ``$0.00`` for labels and logs."""

    return f"${value:,.2f}"


def format_zip_for_spreadsheet(postal_code: object) -> str:
    """
    Force spreadsheet apps to keep a ZIP code's leading zeros.

    Writing the cell as ``="01001"`` stops Excel from turning 01001 into 1001.
    """

    return f'="{normalize_zip(postal_code)}"'


# ---------------------------------------------------------------------------
# Fee schedule engine (standard library only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeeTier:
    """
    One band of the delivery-fee schedule.

    upper_miles:
        Inclusive upper bound of the band in miles. ``None`` marks the final,
        open-ended band that has no upper limit.

    fee:
        Flat delivery fee for the band, in dollars.

    min_order:
        Minimum order subtotal required to deliver into this band, in dollars.

    per_mile_over / over_threshold_miles:
        Optional per-mile surcharge for the open-ended band. When set, the fee
        becomes ``fee + per_mile_over * (distance - over_threshold_miles)`` for
        distances beyond ``over_threshold_miles``. For example, "$15 plus $1
        per mile beyond 25 miles" prices 26 miles at $16.00.

    label:
        Human-readable name shown in CSV, GeoJSON, and the map legend.
    """

    upper_miles: Optional[float]
    fee: float
    min_order: float
    per_mile_over: float = 0.0
    over_threshold_miles: Optional[float] = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.fee < 0:
            raise ValueError("fee cannot be negative")
        if self.min_order < 0:
            raise ValueError("min_order cannot be negative")
        if self.per_mile_over < 0:
            raise ValueError("per_mile_over cannot be negative")
        if self.per_mile_over and self.over_threshold_miles is None:
            raise ValueError(
                "over_threshold_miles is required when per_mile_over is set"
            )


@dataclass(frozen=True)
class Quote:
    """The result of pricing one distance against a :class:`FeeSchedule`."""

    distance_miles: float
    tier_order: int
    tier_label: str
    fee: float
    min_order: float


@dataclass
class FeeZoneResult:
    """
    Everything ``save_outputs`` needs, returned by ``build_fee_zones``.

    ``gdf`` is the priced per-ZIP GeoDataFrame (in the store-centered azimuthal
    CRS); the origin and config travel alongside it explicitly rather than
    through ``GeoDataFrame.attrs`` (which pandas does not reliably preserve
    across operations).
    """

    gdf: object
    origin_lat: float
    origin_lon: float
    settings: "GeoZoneSettings"
    schedule: FeeSchedule
    override_zones: Tuple["OverrideZone", ...] = ()


@dataclass(frozen=True)
class FeeSchedule:
    """
    An ordered set of fee tiers plus the rule for rounding per-mile surcharges.

    tiers:
        Bands ordered from nearest to farthest. Exactly one band (the last one)
        should have ``upper_miles=None`` so every distance can be priced.

    over_rounding:
        How partial miles are charged in the open-ended band:
        ``"continuous"`` (default; exact, so 25.5 mi adds $1.00 at $2/mi),
        ``"ceil"`` (round each started mile up), ``"floor"``, or ``"round"``.
    """

    tiers: Tuple[FeeTier, ...]
    over_rounding: str = "continuous"

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("FeeSchedule needs at least one tier")
        # Every finite band must precede the open-ended band, and bounds must
        # strictly increase, otherwise pricing is ambiguous.
        last_bound = -math.inf
        open_seen = False
        for tier in self.tiers:
            if open_seen:
                raise ValueError("no tier may follow the open-ended (None) tier")
            if tier.upper_miles is None:
                open_seen = True
                continue
            if tier.upper_miles <= last_bound:
                raise ValueError("tier upper_miles must strictly increase")
            last_bound = tier.upper_miles
        if self.over_rounding not in {"continuous", "ceil", "floor", "round"}:
            raise ValueError(f"unknown over_rounding: {self.over_rounding}")

    def boundaries(self) -> List[float]:
        """Return the finite band upper bounds, e.g. ``[10, 15, 20, 25]``."""

        return [t.upper_miles for t in self.tiers if t.upper_miles is not None]

    def tier_order_for(self, distance_miles: float) -> int:
        """Return the index of the tier a distance falls into."""

        for index, tier in enumerate(self.tiers):
            if tier.upper_miles is None:
                return index
            if distance_miles <= tier.upper_miles + _BOUNDARY_EPSILON:
                return index
        # Reached only if there is no open tier and distance exceeds all bounds.
        return len(self.tiers) - 1

    def _round_overage(self, overage_miles: float) -> float:
        if self.over_rounding == "continuous":
            return overage_miles
        if self.over_rounding == "ceil":
            return float(math.ceil(overage_miles))
        if self.over_rounding == "floor":
            return float(math.floor(overage_miles))
        return float(round(overage_miles))

    def quote(self, distance_miles: float) -> Quote:
        """Price one distance, returning the tier, fee, and minimum order."""

        order = self.tier_order_for(distance_miles)
        tier = self.tiers[order]
        fee = tier.fee
        if (
            tier.per_mile_over
            and tier.over_threshold_miles is not None
            and distance_miles > tier.over_threshold_miles
        ):
            overage = self._round_overage(distance_miles - tier.over_threshold_miles)
            fee += tier.per_mile_over * overage
        return Quote(
            distance_miles=distance_miles,
            tier_order=order,
            tier_label=tier.label,
            fee=round(fee, 2),
            min_order=tier.min_order,
        )


def example_fee_schedule() -> FeeSchedule:
    """
    Return an example delivery-fee schedule to copy and adapt.

    0-10 mi:      free, $50 minimum
    10.1-15 mi:   $5.00, $50 minimum
    15.1-20 mi:   $10.00, $50 minimum
    20.1-25 mi:   $15.00, $75 minimum
    beyond 25 mi: $15.00 + $1 per mile over 25, $75 minimum
                  (e.g. 26 mi -> $16.00)
    """

    return FeeSchedule(
        tiers=(
            FeeTier(upper_miles=10.0, fee=0.00, min_order=50.0, label="0-10 mi (free)"),
            FeeTier(upper_miles=15.0, fee=5.00, min_order=50.0, label="10.1-15 mi"),
            FeeTier(upper_miles=20.0, fee=10.00, min_order=50.0, label="15.1-20 mi"),
            FeeTier(upper_miles=25.0, fee=15.00, min_order=75.0, label="20.1-25 mi"),
            FeeTier(
                upper_miles=None,
                fee=15.00,
                min_order=75.0,
                per_mile_over=1.00,
                over_threshold_miles=25.0,
                label="25.1+ mi (+$1/mi over 25)",
            ),
        ),
        over_rounding="continuous",
    )


@dataclass(frozen=True)
class OverrideZone:
    """
    A flat-fee zone that overrides the mileage schedule inside a circle.

    Some stores price a whole town at one promotional flat fee regardless of its
    real distance from the store. An ``OverrideZone`` is a circle of radius
    ``radius_miles`` around (``center_lat``, ``center_lon``); any ZIP whose
    primary-metric distance from that center is within the radius is charged the
    flat ``fee`` with ``min_order``, regardless of its distance-from-store tier.
    Overrides are applied after the schedule, so they win over the distance bands.

    Resolve ``center_lat`` / ``center_lon`` for a town from its ZIP with
    :func:`latlon_for_zip`.

    name:
        Short identifier recorded in the ``override_zone`` column, e.g.
        ``"Downtown"``.

    label:
        Human-readable tier name shown in CSV, GeoJSON, and the legend. Defaults
        to ``"<name> flat $<fee>"``.
    """

    name: str
    center_lat: float
    center_lon: float
    radius_miles: float
    fee: float
    min_order: float
    label: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OverrideZone needs a name")
        if self.radius_miles <= 0:
            raise ValueError("radius_miles must be positive")
        if self.fee < 0:
            raise ValueError("fee cannot be negative")
        if self.min_order < 0:
            raise ValueError("min_order cannot be negative")

    def display_label(self) -> str:
        """Return the map/CSV label, defaulting to ``<name> flat $<fee>``."""

        return self.label or f"{self.name} flat {format_money(self.fee)}"

    def covers(self, distance_from_center_miles: float) -> bool:
        """True if a distance from this zone's center lies inside the circle."""

        return distance_from_center_miles <= self.radius_miles + _BOUNDARY_EPSILON


# ---------------------------------------------------------------------------
# Geo configuration object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeoZoneSettings:
    """
    Settings for the geometry pass.

    primary_metric:
        Which per-ZIP distance decides a ZIP's headline fee tier:
        ``"centroid"`` (default; the representative middle of the ZIP),
        ``"nearest"`` (closest edge, most generous), or ``"farthest"``
        (most conservative). All three distances are always reported; this only
        chooses which one drives the ``fee_tier``/dissolved-zone assignment.

    ring_segments:
        Number of segments per quarter-circle when drawing distance rings.
        Higher is smoother.

    emit_clipped:
        When True, also write the ring-clipped sub-ZIP layer.

    max_extent_miles:
        Optional hard cap (miles) on the outermost ring and clipped annulus.
        ``None`` uses the farthest ZIP's distance.
    """

    primary_metric: str = "centroid"
    ring_segments: int = 96
    emit_clipped: bool = True
    max_extent_miles: Optional[float] = None

    def __post_init__(self) -> None:
        if self.primary_metric not in {"nearest", "centroid", "farthest"}:
            raise ValueError(f"unknown primary_metric: {self.primary_metric}")
        if self.ring_segments < 4:
            raise ValueError("ring_segments must be at least 4")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def download_census_zcta(
    url: str = CENSUS_ZCTA_URL,
    cache_dir: PathInput = DEFAULT_CACHE_DIR,
) -> Path:
    """
    Download the Census ZCTA archive to a cache directory, once.

    Returns the local path. Re-running reuses the cached file. A partial
    download is written to a ``.part`` file and renamed only on success, so an
    interrupted run never leaves a corrupt archive behind.
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    target = cache_path / Path(url).name

    if target.exists() and target.stat().st_size > 0:
        LOGGER.info("Using cached ZCTA archive at %s (%s bytes)", target, target.stat().st_size)
        return target

    partial = target.with_suffix(target.suffix + ".part")
    LOGGER.info("Downloading ZCTA polygons from %s", url)
    with urlopen(url) as response:  # nosec - fixed, trusted Census URL by default
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        chunk_size = 1 << 20  # 1 MiB
        with open(partial, "wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    LOGGER.debug(
                        "Downloaded %.1f%% (%s / %s bytes)",
                        100.0 * downloaded / total_bytes,
                        downloaded,
                        total_bytes,
                    )
    partial.replace(target)
    LOGGER.info("Saved ZCTA archive to %s (%s bytes)", target, target.stat().st_size)
    return target


def _detect_zip_field(columns) -> str:
    """Return the first known ZIP-code column present, or raise."""

    available = set(columns)
    for candidate in ZCTA_FIELD_CANDIDATES:
        if candidate in available:
            return candidate
    raise ValueError(
        "Could not find a ZIP-code column in the polygon data. Looked for "
        f"{ZCTA_FIELD_CANDIDATES}; saw {sorted(available)}"
    )


def load_zcta_polygons(source: PathInput):
    """
    Load ZIP polygons from a local file or a zipped shapefile path.

    ``source`` may be a GeoJSON file, a shapefile, or a Census ``.zip`` archive
    (geopandas reads the shapefile inside it directly). Returns a GeoDataFrame
    with a normalized 5-digit ``postal_code`` column, repaired geometries, and a
    defined CRS.
    """

    import geopandas as gpd  # lazy: keeps the fee engine import-light

    source_path = Path(source)
    LOGGER.info("Reading ZIP polygons from %s", source_path)
    gdf = gpd.read_file(source_path)
    LOGGER.info("Loaded %s polygon rows; columns: %s", len(gdf), list(gdf.columns))

    zip_field = _detect_zip_field(gdf.columns)
    LOGGER.info("Using %r as the ZIP-code column", zip_field)
    gdf = gdf[[zip_field, gdf.geometry.name]].copy()
    gdf = gdf.rename(columns={zip_field: "postal_code"})
    gdf["postal_code"] = gdf["postal_code"].apply(normalize_zip)

    before = len(gdf)
    gdf = gdf[gdf["postal_code"].str.fullmatch(r"\d{5}")].copy()
    dropped = before - len(gdf)
    if dropped:
        LOGGER.warning(
            "Dropped %s of %s rows whose %r value was not a 5-digit ZIP",
            dropped,
            before,
            zip_field,
        )
    # A wrong column (e.g. a long GEOID that is state+county+ZCTA) shows up as a
    # large fraction of rows failing the 5-digit test. Fail loudly instead of
    # silently shipping a near-empty result.
    if before and dropped / before > 0.5:
        raise ValueError(
            f"More than half of the rows ({dropped}/{before}) failed the 5-digit "
            f"ZIP check using column {zip_field!r}. This usually means the wrong "
            "column was picked; pass a polygon source whose ZIP field is one of "
            f"{ZCTA_FIELD_CANDIDATES}."
        )

    if gdf.crs is None:
        # Census ZCTA cartographic files are NAD83 (EPSG:4269). Assume it only
        # when the file itself failed to declare a CRS.
        LOGGER.warning("Polygon source had no CRS; assuming EPSG:4269 (NAD83)")
        gdf = gdf.set_crs("EPSG:4269")

    # Repair any self-intersecting rings before doing geometry math.
    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        LOGGER.info("Repairing %s invalid polygon geometries via make_valid()", invalid)
        gdf["geometry"] = gdf.geometry.make_valid()

    duplicates = int(gdf["postal_code"].duplicated().sum())
    if duplicates:
        LOGGER.info("Dissolving %s duplicate ZIP polygon rows", duplicates)
        gdf = gdf.dissolve(by="postal_code", as_index=False)

    LOGGER.info("Prepared %s unique ZIP polygons (CRS %s)", len(gdf), gdf.crs)
    return gdf


def attach_zip_metadata(zcta_gdf):
    """
    Join state / place / point metadata onto the ZIP polygons via pgeocode.

    Census ZCTA polygons carry no state column, so the state filter and origin
    validation rely on the same trusted ZIP->state table that ``zipGrouper.py``
    already uses. Returns the polygons inner-joined to that table.
    """

    from zipGrouper import clean_zip_data, load_us_zip_data  # trusted, reused

    meta_df = clean_zip_data(load_us_zip_data())
    meta_df = meta_df[["postal_code", "place_name", "state_code", "latitude", "longitude"]]

    before = len(zcta_gdf)
    merged = zcta_gdf.merge(meta_df, on="postal_code", how="inner")
    dropped = before - len(merged)
    LOGGER.info(
        "Joined polygons to ZIP metadata: %s matched, %s polygons had no "
        "pgeocode match and were dropped",
        len(merged),
        dropped,
    )
    return merged


def filter_state(gdf, state_code: str):
    """Keep only ZIP polygons whose pgeocode state matches ``state_code``."""

    normalized = normalize_state_code(state_code)
    state_gdf = gdf[gdf["state_code"] == normalized].copy()
    if state_gdf.empty:
        raise ValueError(f"No ZIP polygons found for state {normalized}")
    LOGGER.info("State filter kept %s ZIP polygons for %s", len(state_gdf), normalized)
    return state_gdf


# ---------------------------------------------------------------------------
# Geometry: origin, projection, and distances
# ---------------------------------------------------------------------------


def latlon_for_zip(zip_code: str) -> Tuple[float, float]:
    """
    Return the (latitude, longitude) for a ZIP from pgeocode.

    Used to place an :class:`OverrideZone` on a town by its ZIP. This is the
    pgeocode point, which is accurate to well within a ZIP for a multi-mile
    radius; the store origin uses the polygon centroid via :func:`resolve_origin`
    instead.
    """

    import pgeocode  # lazy: keeps the fee engine import-light

    record = pgeocode.Nominatim("us").query_postal_code(normalize_zip(zip_code))
    lat = float(record["latitude"])
    lon = float(record["longitude"])
    if math.isnan(lat) or math.isnan(lon):
        raise ValueError(f"pgeocode has no coordinates for ZIP {zip_code}")
    return lat, lon


def resolve_origin(
    gdf,
    start_zip: str,
    state_code: str,
    origin_latlon: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Return the (latitude, longitude) the fee zones are measured from.

    By default this is the centroid of the starting ZIP's polygon, which keeps
    the origin inside that ZIP. Pass ``origin_latlon`` to measure from the
    store's exact coordinates instead. The starting ZIP is still validated to
    exist in the chosen state so a wrong-state typo is caught.
    """

    normalized_zip = normalize_zip(start_zip)
    normalized_state = normalize_state_code(state_code)
    match = gdf[gdf["postal_code"] == normalized_zip]

    # Explicit coordinates win and do not require the ZIP to have a polygon,
    # so a store in a PO-box-only origin ZIP can still be located.
    if origin_latlon is not None:
        lat, lon = float(origin_latlon[0]), float(origin_latlon[1])
        if match.empty:
            LOGGER.warning(
                "Starting ZIP %s has no polygon in the %s data; using the "
                "explicit ORIGIN_LATLON anyway",
                normalized_zip,
                normalized_state,
            )
        LOGGER.info("Using explicit origin coordinates lat=%s lon=%s", lat, lon)
        return lat, lon

    if match.empty:
        raise ValueError(
            f"Starting ZIP {normalized_zip} is not in the {normalized_state} polygon "
            "set. It may be a PO-box-only ZIP (ZCTAs cover populated areas only) or "
            "may have been dropped because pgeocode has no record of it. Pass "
            "ORIGIN_LATLON to set the store location explicitly."
        )
    origin_state = normalize_state_code(match.iloc[0]["state_code"])
    if origin_state != normalized_state:
        raise ValueError(
            f"Starting ZIP {normalized_zip} is in {origin_state}, not {normalized_state}"
        )

    # Compute the centroid in an equal-area CRS (not geographic) so geopandas
    # does not warn and the point is a faithful interior centroid, then convert
    # that single point back to lat/lon.
    import geopandas as gpd

    projected_centroid = match.to_crs(EQUAL_AREA_CRS).geometry.centroid.iloc[0]
    centroid = (
        gpd.GeoSeries([projected_centroid], crs=EQUAL_AREA_CRS).to_crs(WGS84).iloc[0]
    )
    lat, lon = float(centroid.y), float(centroid.x)
    LOGGER.info(
        "Origin = centroid of ZIP %s (%s): lat=%s lon=%s",
        normalized_zip,
        match.iloc[0]["place_name"],
        lat,
        lon,
    )
    return lat, lon


def azimuthal_equidistant_crs(lat: float, lon: float) -> str:
    """
    Return a PROJ string for an azimuthal-equidistant CRS centered on a point.

    In this projection the center maps to (0, 0) and the planar distance from
    the center to any point equals its true great-circle distance. That makes
    edge/centroid distances exact and makes buffers true distance rings.
    """

    return (
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def _iter_vertices(geometry):
    """
    Yield every (x, y) vertex of any geometry type.

    Handles Polygon, MultiPolygon, and the LineString / Point / GeometryCollection
    shapes that ``make_valid`` can emit when repairing a damaged polygon, so the
    farthest-point scan never silently falls back to an empty geometry.
    """

    geom_type = geometry.geom_type
    if geom_type == "Polygon":
        yield from geometry.exterior.coords
    elif geom_type in ("Point", "LineString", "LinearRing"):
        yield from geometry.coords
    elif geom_type in (
        "MultiPolygon",
        "MultiLineString",
        "MultiPoint",
        "GeometryCollection",
    ):
        for part in geometry.geoms:
            yield from _iter_vertices(part)


def _farthest_distance_meters(geometry, origin_point) -> float:
    """
    Distance from the origin to the farthest point of a geometry, in metres.

    The farthest point from an external point always lies on a boundary vertex,
    so scanning vertices is exact and avoids densification surprises. Returns
    ``None`` for an empty/degenerate geometry so the caller can fall back rather
    than silently report 0 (which would misprice the ZIP into the free tier).
    """

    from shapely.geometry import Point

    best = None
    for x, y in _iter_vertices(geometry):
        distance = origin_point.distance(Point(x, y))
        if best is None or distance > best:
            best = distance
    return best


def compute_distances(state_gdf, origin_lat: float, origin_lon: float):
    """
    Add nearest / centroid / farthest distance columns (in miles) and area.

    The work is done in a store-centered azimuthal-equidistant CRS so distances
    are true miles from the store. Area is computed separately in an equal-area
    CRS. The returned GeoDataFrame stays in that azimuthal CRS so rings and
    clipping can reuse it; reproject to WGS84 only when writing GeoJSON.
    """

    from shapely.geometry import Point

    aeqd = azimuthal_equidistant_crs(origin_lat, origin_lon)
    LOGGER.info("Projecting %s polygons to azimuthal-equidistant CRS", len(state_gdf))
    projected = state_gdf.to_crs(aeqd).copy()
    origin_point = Point(0.0, 0.0)  # the projection center

    geometry = projected.geometry
    nearest_m = geometry.distance(origin_point)
    projected["distance_nearest_miles"] = (nearest_m / METERS_PER_MILE).to_numpy()

    # Farthest vertex; fall back to the nearest distance if a repaired geometry
    # is degenerate, so the nearest <= centroid <= farthest invariant holds.
    nearest_values = nearest_m.to_numpy()
    farthest_m = [
        far if (far := _farthest_distance_meters(geom, origin_point)) is not None else nearest
        for geom, nearest in zip(geometry, nearest_values)
    ]
    projected["distance_farthest_miles"] = [m / METERS_PER_MILE for m in farthest_m]

    # Centroid and area both come from the equal-area CRS. The centroid of the
    # azimuthal geometry is distorted for ZIPs far from the store, so the true
    # (equal-area) centroid is reprojected into the azimuthal CRS and measured
    # from the origin there, where distance is exact.
    equal_area = state_gdf.to_crs(EQUAL_AREA_CRS)
    centroids_aeqd = equal_area.geometry.centroid.to_crs(aeqd)
    projected["distance_centroid_miles"] = (
        centroids_aeqd.distance(origin_point) / METERS_PER_MILE
    ).to_numpy()
    projected["area_sq_miles"] = (
        equal_area.geometry.area / (METERS_PER_MILE**2)
    ).to_numpy()

    LOGGER.info(
        "Distance range (nearest edge): %.2f-%.2f miles across %s ZIPs",
        projected["distance_nearest_miles"].min(),
        projected["distance_nearest_miles"].max(),
        len(projected),
    )
    return projected


# ---------------------------------------------------------------------------
# Fee assignment
# ---------------------------------------------------------------------------


def apply_fee_schedule(
    gdf,
    schedule: FeeSchedule,
    primary_metric: str,
):
    """
    Add fee columns to the ZIP GeoDataFrame.

    The headline tier/fee/min-order come from the chosen ``primary_metric``
    distance. Fees at the nearest and farthest edges are also recorded, along
    with a ``straddles_boundary`` flag and the count of fee tiers a single ZIP
    spans, so borderline ZIPs are easy to spot and handle by hand.
    """

    metric_column = f"distance_{primary_metric}_miles"
    LOGGER.info("Pricing ZIPs by %s distance", primary_metric)

    primary = gdf[metric_column].apply(schedule.quote)
    gdf = gdf.copy()
    gdf["primary_distance_miles"] = gdf[metric_column]
    gdf["fee_tier_order"] = [q.tier_order for q in primary]
    gdf["fee_tier"] = [q.tier_label for q in primary]
    gdf["delivery_fee"] = [q.fee for q in primary]
    gdf["min_order"] = [q.min_order for q in primary]

    near = gdf["distance_nearest_miles"].apply(schedule.quote)
    far = gdf["distance_farthest_miles"].apply(schedule.quote)
    near_order = [q.tier_order for q in near]
    far_order = [q.tier_order for q in far]
    gdf["fee_at_nearest"] = [q.fee for q in near]
    gdf["fee_at_farthest"] = [q.fee for q in far]
    gdf["tiers_spanned"] = [hi - lo + 1 for lo, hi in zip(near_order, far_order)]
    gdf["straddles_boundary"] = [hi != lo for lo, hi in zip(near_order, far_order)]

    straddlers = int(gdf["straddles_boundary"].sum())
    LOGGER.info(
        "%s of %s ZIPs straddle a fee boundary (their edges price differently)",
        straddlers,
        len(gdf),
    )
    return gdf.sort_values(
        ["fee_tier_order", "primary_distance_miles", "postal_code"]
    ).reset_index(drop=True)


def apply_override_zones(
    gdf,
    zones: Tuple["OverrideZone", ...],
    schedule: FeeSchedule,
    primary_metric: str,
):
    """
    Re-price ZIPs that fall inside a flat-fee override zone.

    Call after :func:`apply_fee_schedule`. For each override zone this measures
    the chosen ``primary_metric`` distance from the zone center to every ZIP and,
    for those inside the circle, replaces the fee, minimum order, tier label, and
    tier order with the zone's flat values. An ``override_zone`` column (empty
    string when none) records which zone, if any, re-priced each ZIP.

    Override tiers are numbered after the schedule's tiers so they dissolve and
    colour as their own zones. When zones overlap, the first listed wins.
    """

    gdf = gdf.copy()
    if "override_zone" not in gdf.columns:
        gdf["override_zone"] = ""

    if not zones:
        return gdf

    from shapely.geometry import Point

    origin = Point(0.0, 0.0)
    # Centroids in an equal-area CRS, matching how distance_centroid_miles and
    # the store origin centroid are computed elsewhere, so "centroid within N mi
    # of the town" stays consistent with the rest of the tool.
    equal_area_centroids = gdf.to_crs(EQUAL_AREA_CRS).geometry.centroid

    base_order = len(schedule.tiers)
    for zone_index, zone in enumerate(zones):
        zone_aeqd = azimuthal_equidistant_crs(zone.center_lat, zone.center_lon)
        if primary_metric == "nearest":
            measured = gdf.to_crs(zone_aeqd).geometry
        else:
            # "centroid" (default) and "farthest" both decide membership by the
            # centre of the ZIP here: a flat town zone is a centre-of-ZIP test.
            measured = equal_area_centroids.to_crs(zone_aeqd)
        distance_miles = measured.distance(origin) / METERS_PER_MILE

        # Only claim a ZIP for this zone if an earlier (higher-priority) zone has
        # not already taken it.
        unclaimed = gdf["override_zone"] == ""
        inside = (distance_miles <= zone.radius_miles + _BOUNDARY_EPSILON) & unclaimed
        count = int(inside.sum())
        if not count:
            LOGGER.warning(
                "Override zone %r covered no ZIPs (center=%.4f,%.4f r=%.1f mi)",
                zone.name,
                zone.center_lat,
                zone.center_lon,
                zone.radius_miles,
            )
            continue

        tier_order = base_order + zone_index
        flat_fee = round(zone.fee, 2)
        gdf.loc[inside, "override_zone"] = zone.name
        gdf.loc[inside, "fee_tier_order"] = tier_order
        gdf.loc[inside, "fee_tier"] = zone.display_label()
        gdf.loc[inside, "delivery_fee"] = flat_fee
        gdf.loc[inside, "min_order"] = zone.min_order
        # A flat zone prices the same everywhere inside it, so the edge fees and
        # the straddle flag (which describe the store rings) no longer apply.
        gdf.loc[inside, "fee_at_nearest"] = flat_fee
        gdf.loc[inside, "fee_at_farthest"] = flat_fee
        gdf.loc[inside, "tiers_spanned"] = 1
        gdf.loc[inside, "straddles_boundary"] = False
        LOGGER.info(
            "Override zone %r (flat %s, min %s) re-priced %s ZIPs",
            zone.name,
            format_money(zone.fee),
            format_money(zone.min_order),
            count,
        )

    return gdf.sort_values(
        ["fee_tier_order", "primary_distance_miles", "postal_code"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived map layers: rings, dissolved zones, clipped sub-ZIP pieces
# ---------------------------------------------------------------------------


def tier_color(tier_order: int) -> str:
    """Return a hex colour for a tier order, clamped to the ramp ends."""

    if tier_order < 0:
        tier_order = 0
    if tier_order >= len(TIER_COLORS):
        tier_order = len(TIER_COLORS) - 1
    return TIER_COLORS[tier_order]


def _add_simplestyle(gdf, fill_opacity: float = 0.45):
    """
    Add mapbox/geojson.io "simplestyle-spec" properties keyed on fee tier.

    Viewers like geojson.io and many Leaflet plugins read these so the export
    renders coloured by fee tier with no extra styling code.
    """

    gdf = gdf.copy()
    colors = gdf["fee_tier_order"].apply(tier_color)
    # Flat-fee override zones sit outside the near->far ramp; paint them in the
    # single distinct override colour instead of a distance-tier colour.
    if "override_zone" in gdf.columns:
        is_override = gdf["override_zone"].astype(str).str.len() > 0
        colors = colors.where(~is_override, OVERRIDE_COLOR)
    gdf["fill"] = colors
    gdf["stroke"] = colors
    gdf["fill-opacity"] = fill_opacity
    gdf["stroke-width"] = 1
    return gdf


def build_rings(
    schedule: FeeSchedule,
    origin_lat: float,
    origin_lon: float,
    max_extent_miles: float,
    ring_segments: int,
    override_zones: Tuple["OverrideZone", ...] = (),
):
    """
    Build concentric fee-boundary rings as a GeoDataFrame (in WGS84).

    Each finite band boundary becomes a circle outline; the open tier gets an
    outer circle at ``max_extent_miles``. Each override zone adds its own circle
    (centered on the town, not the store) in the override colour. Rings carry
    simplestyle properties with no fill so only the outlines show on a map.
    """

    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    aeqd = azimuthal_equidistant_crs(origin_lat, origin_lon)
    origin = Point(0.0, 0.0)
    quad_segs = max(1, ring_segments // 4)

    radii_miles = list(schedule.boundaries())
    if max_extent_miles > (radii_miles[-1] if radii_miles else 0):
        radii_miles.append(max_extent_miles)

    rows = []
    geometries = []
    for index, radius in enumerate(radii_miles):
        # quad_segs passed positionally: it is the 2nd positional argument in
        # both shapely 1.x and 2.x, so this is robust to either version.
        disk = origin.buffer(radius * METERS_PER_MILE, quad_segs)
        geometries.append(disk.boundary)
        rows.append(
            {
                "boundary_miles": radius,
                "label": f"{format_miles(radius)} mi ring",
                "fee_tier_order": index,
            }
        )

    rings = gpd.GeoDataFrame(rows, geometry=geometries, crs=aeqd).to_crs(WGS84)
    colors = rings["fee_tier_order"].apply(tier_color)
    rings["stroke"] = colors
    rings["stroke-width"] = 2
    rings["fill-opacity"] = 0

    # Override-zone circles are drawn around their own centre, so each is built
    # in a CRS centred on that town and then reprojected to WGS84.
    extra_rows = []
    extra_geoms = []
    for zone in override_zones:
        zone_aeqd = azimuthal_equidistant_crs(zone.center_lat, zone.center_lon)
        disk = origin.buffer(zone.radius_miles * METERS_PER_MILE, quad_segs)
        circle = gpd.GeoSeries([disk.boundary], crs=zone_aeqd).to_crs(WGS84).iloc[0]
        extra_rows.append(
            {
                "boundary_miles": zone.radius_miles,
                "label": f"{zone.name} {format_miles(zone.radius_miles)} mi zone",
                "fee_tier_order": -1,
                "stroke": OVERRIDE_COLOR,
                "stroke-width": 2,
                "fill-opacity": 0,
            }
        )
        extra_geoms.append(circle)
    if extra_rows:
        extra = gpd.GeoDataFrame(extra_rows, geometry=extra_geoms, crs=WGS84)
        rings = gpd.GeoDataFrame(
            pd.concat([rings, extra], ignore_index=True), crs=WGS84
        )

    LOGGER.info("Built %s fee-boundary rings", len(rings))
    return rings


def dissolve_fee_zones(priced_gdf):
    """
    Dissolve per-ZIP polygons into one multipolygon per fee tier (WGS84).

    Geometry is dissolved by tier; tier attributes and the ZIP roster are
    computed from a plain groupby and merged back, which is robust across
    geopandas versions.
    """

    geo = priced_gdf[["fee_tier_order", "geometry"]].dissolve(
        by="fee_tier_order", as_index=False
    )

    attrs = (
        priced_gdf.drop(columns="geometry")
        .groupby("fee_tier_order", as_index=False)
        .agg(
            fee_tier=("fee_tier", "first"),
            delivery_fee=("delivery_fee", "first"),
            min_order=("min_order", "first"),
            override_zone=("override_zone", "first"),
            zip_count=("postal_code", "size"),
            min_distance_miles=("distance_nearest_miles", "min"),
            max_distance_miles=("distance_farthest_miles", "max"),
            zip_codes=("postal_code", lambda codes: ", ".join(sorted(codes))),
        )
    )

    zones = geo.merge(attrs, on="fee_tier_order").sort_values("fee_tier_order")
    zones = zones.to_crs(WGS84)
    zones = _add_simplestyle(zones, fill_opacity=0.5)
    LOGGER.info("Dissolved ZIPs into %s fee zones", len(zones))
    return zones


def clip_zips_to_rings(
    priced_gdf,
    schedule: FeeSchedule,
    origin_lat: float,
    origin_lon: float,
    max_extent_miles: float,
    ring_segments: int,
):
    """
    Split each ZIP polygon into the slices that fall in each fee tier (WGS84).

    This is the finest-grained output: a single large ZIP that crosses the 25
    mile line is cut into one piece inside and one piece outside, each tagged
    with its own tier. The intersection is computed in the store-centered
    azimuthal CRS so the cuts are exact distance rings.
    """

    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    aeqd = azimuthal_equidistant_crs(origin_lat, origin_lon)
    origin = Point(0.0, 0.0)
    quad_segs = max(1, ring_segments // 4)

    # Flat-fee override ZIPs are priced regardless of distance from the store, so
    # they are not sliced by the store rings; they pass through whole below.
    if "override_zone" in priced_gdf.columns:
        override_mask = priced_gdf["override_zone"].astype(str).str.len() > 0
    else:
        override_mask = pd.Series(False, index=priced_gdf.index)
    ring_zips = priced_gdf[~override_mask]
    override_zips = priced_gdf[override_mask]

    boundaries = list(schedule.boundaries())
    outer = max(max_extent_miles, boundaries[-1] if boundaries else max_extent_miles)
    edges = [0.0] + boundaries + [outer]

    # Build one annulus per tier (disk difference), tagged with the tier it
    # represents, then intersect the ZIP polygons against them.
    annulus_rows = []
    annulus_geoms = []
    for tier_order in range(len(edges) - 1):
        inner_r, outer_r = edges[tier_order], edges[tier_order + 1]
        # quad_segs passed positionally for shapely 1.x / 2.x compatibility.
        outer_disk = origin.buffer(outer_r * METERS_PER_MILE, quad_segs)
        annulus = (
            outer_disk
            if inner_r <= 0
            else outer_disk.difference(origin.buffer(inner_r * METERS_PER_MILE, quad_segs))
        )
        annulus_rows.append(
            {
                "fee_tier_order": tier_order,
                "fee_tier": schedule.tiers[tier_order].label,
                "delivery_fee": schedule.tiers[tier_order].fee,
                "min_order": schedule.tiers[tier_order].min_order,
            }
        )
        annulus_geoms.append(annulus)

    annuli = gpd.GeoDataFrame(annulus_rows, geometry=annulus_geoms, crs=aeqd)

    zips_proj = ring_zips[["postal_code", "place_name", "geometry"]].to_crs(aeqd)
    pieces = gpd.overlay(zips_proj, annuli, how="intersection", keep_geom_type=True)
    pieces = pieces[~pieces.geometry.is_empty].copy()
    # Area must come from an equal-area CRS; the azimuthal CRS distorts area for
    # slices far from the store.
    pieces["piece_sq_miles"] = (
        pieces.to_crs(EQUAL_AREA_CRS).geometry.area / (METERS_PER_MILE**2)
    ).to_numpy()
    pieces = pieces.to_crs(WGS84)
    pieces["override_zone"] = ""

    # Override ZIPs are emitted whole (one piece each) with their flat-fee
    # attributes, so the finest layer still shows the override zones.
    if not override_zips.empty:
        whole = override_zips[
            [
                "postal_code",
                "place_name",
                "fee_tier_order",
                "fee_tier",
                "delivery_fee",
                "min_order",
                "override_zone",
                "geometry",
            ]
        ].copy()
        whole["piece_sq_miles"] = (
            whole.to_crs(EQUAL_AREA_CRS).geometry.area / (METERS_PER_MILE**2)
        ).to_numpy()
        whole = whole.to_crs(WGS84)
        pieces = gpd.GeoDataFrame(
            pd.concat([pieces, whole], ignore_index=True), crs=WGS84
        )

    pieces = _add_simplestyle(pieces, fill_opacity=0.55)
    LOGGER.info(
        "Clipped %s ZIPs into %s tier slices (%s flat-fee ZIPs passed through whole)",
        priced_gdf["postal_code"].nunique(),
        len(pieces),
        len(override_zips),
    )
    return pieces


# ---------------------------------------------------------------------------
# Tabular summaries
# ---------------------------------------------------------------------------


def build_fee_table(priced_gdf):
    """Return a plain (geometry-free) per-ZIP DataFrame for the detail CSV."""

    columns = [
        "postal_code",
        "place_name",
        "state_code",
        "latitude",
        "longitude",
        "distance_nearest_miles",
        "distance_centroid_miles",
        "distance_farthest_miles",
        "primary_distance_miles",
        "fee_tier_order",
        "fee_tier",
        "delivery_fee",
        "min_order",
        "override_zone",
        "fee_at_nearest",
        "fee_at_farthest",
        "straddles_boundary",
        "tiers_spanned",
        "area_sq_miles",
    ]
    present = [c for c in columns if c in priced_gdf.columns]
    return priced_gdf[present].copy()


def build_grouped_fees(priced_gdf):
    """One summary row per fee tier, mirroring zipGrouper's by-zone CSV."""

    grouped = (
        priced_gdf.groupby(["fee_tier_order", "fee_tier"], as_index=False)
        .agg(
            delivery_fee=("delivery_fee", "first"),
            min_order=("min_order", "first"),
            zip_count=("postal_code", "size"),
            min_distance_miles=("distance_nearest_miles", "min"),
            max_distance_miles=("distance_farthest_miles", "max"),
            zip_codes=("postal_code", lambda codes: ", ".join(sorted(codes))),
        )
        .sort_values("fee_tier_order")
        .reset_index(drop=True)
    )
    LOGGER.info("Built grouped fee summary with %s tiers", len(grouped))
    return grouped


# ---------------------------------------------------------------------------
# End-to-end build and output
# ---------------------------------------------------------------------------


def build_fee_zones(
    state_code: str,
    start_zip: str,
    schedule: FeeSchedule,
    settings: Optional[GeoZoneSettings] = None,
    polygon_source: Optional[PathInput] = None,
    origin_latlon: Optional[Tuple[float, float]] = None,
    override_zones: Tuple["OverrideZone", ...] = (),
):
    """
    Run the whole pipeline and return the priced per-ZIP GeoDataFrame.

    ``polygon_source`` defaults to the cached Census ZCTA download. Pass a local
    GeoJSON / shapefile / zip path to use your own boundaries instead.
    ``override_zones`` re-prices ZIPs inside flat-fee town circles after the
    mileage schedule (see :class:`OverrideZone`).
    """

    settings = settings or GeoZoneSettings()
    normalized_state = normalize_state_code(state_code)
    normalized_zip = normalize_zip(start_zip)

    LOGGER.info(
        "Building fee zones for state=%s start_zip=%s primary_metric=%s",
        normalized_state,
        normalized_zip,
        settings.primary_metric,
    )

    source = Path(polygon_source) if polygon_source else download_census_zcta()
    polygons = load_zcta_polygons(source)
    polygons = attach_zip_metadata(polygons)
    state_gdf = filter_state(polygons, normalized_state)

    origin_lat, origin_lon = resolve_origin(
        state_gdf, normalized_zip, normalized_state, origin_latlon
    )
    distance_gdf = compute_distances(state_gdf, origin_lat, origin_lon)
    priced = apply_fee_schedule(distance_gdf, schedule, settings.primary_metric)
    priced = apply_override_zones(
        priced, override_zones, schedule, settings.primary_metric
    )

    LOGGER.info("Finished build: %s priced ZIP polygons", len(priced))
    return FeeZoneResult(
        gdf=priced,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        settings=settings,
        schedule=schedule,
        override_zones=override_zones,
    )


def save_outputs(
    result: "FeeZoneResult",
    store_slug: str,
    output_dir: PathInput,
    preserve_zip_display: bool = True,
) -> dict:
    """
    Write all CSV and GeoJSON outputs and return a dict of written paths.

    Files written (``<slug>`` = ``store_slug``):
      <slug>_zip_fees.csv        per-ZIP fee table
      <slug>_fees_by_tier.csv    one row per fee tier
      <slug>_zips.geojson        per-ZIP polygons + fee properties
      <slug>_fee_zones.geojson   dissolved one-polygon-per-tier zones
      <slug>_fee_rings.geojson   concentric fee-boundary rings
      <slug>_zips_clipped.geojson  ring-clipped sub-ZIP slices (optional)
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict = {}

    priced_gdf = result.gdf
    settings = result.settings
    schedule = result.schedule
    origin_lat = result.origin_lat
    origin_lon = result.origin_lon
    override_zones = result.override_zones

    max_extent = settings.max_extent_miles or float(
        priced_gdf["distance_farthest_miles"].max()
    )

    # --- CSV detail ---
    detail_df = build_fee_table(priced_gdf)
    if preserve_zip_display:
        detail_df = detail_df.copy()
        detail_df["postal_code"] = detail_df["postal_code"].apply(
            format_zip_for_spreadsheet
        )
    detail_csv = output_path / f"{store_slug}_zip_fees.csv"
    detail_df.to_csv(detail_csv, index=False)
    written["detail_csv"] = str(detail_csv)
    LOGGER.info("Wrote %s ZIP rows to %s", len(detail_df), detail_csv)

    # --- CSV grouped ---
    grouped_df = build_grouped_fees(priced_gdf)
    grouped_csv = output_path / f"{store_slug}_fees_by_tier.csv"
    grouped_df.to_csv(grouped_csv, index=False)
    written["grouped_csv"] = str(grouped_csv)
    LOGGER.info("Wrote %s fee tiers to %s", len(grouped_df), grouped_csv)

    # --- GeoJSON per-ZIP ---
    zips_geo = _add_simplestyle(priced_gdf).to_crs(WGS84)
    zips_geo = zips_geo.drop(columns=["primary_distance_miles"], errors="ignore")
    zips_path = output_path / f"{store_slug}_zips.geojson"
    zips_geo.to_file(zips_path, driver="GeoJSON")
    written["zips_geojson"] = str(zips_path)
    LOGGER.info("Wrote %s ZIP polygons to %s", len(zips_geo), zips_path)

    # --- GeoJSON dissolved zones ---
    zones = dissolve_fee_zones(priced_gdf)
    zones_path = output_path / f"{store_slug}_fee_zones.geojson"
    zones.to_file(zones_path, driver="GeoJSON")
    written["zones_geojson"] = str(zones_path)
    LOGGER.info("Wrote %s fee zones to %s", len(zones), zones_path)

    # --- GeoJSON rings ---
    rings = build_rings(
        schedule,
        origin_lat,
        origin_lon,
        max_extent,
        settings.ring_segments,
        override_zones,
    )
    rings_path = output_path / f"{store_slug}_fee_rings.geojson"
    rings.to_file(rings_path, driver="GeoJSON")
    written["rings_geojson"] = str(rings_path)
    LOGGER.info("Wrote %s rings to %s", len(rings), rings_path)

    # --- GeoJSON clipped sub-ZIP slices (optional) ---
    if settings.emit_clipped:
        clipped = clip_zips_to_rings(
            priced_gdf, schedule, origin_lat, origin_lon, max_extent, settings.ring_segments
        )
        clipped_path = output_path / f"{store_slug}_zips_clipped.geojson"
        clipped.to_file(clipped_path, driver="GeoJSON")
        written["clipped_geojson"] = str(clipped_path)
        LOGGER.info("Wrote %s clipped slices to %s", len(clipped), clipped_path)

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Build delivery-fee zones for the store configured below.

    Change the CONFIGURATION block to run this for another store or schedule.
    """

    from zipGrouper import output_directory, setup_logging

    # CONFIGURATION ---------------------------------------------------------
    STORE_NAME = "Example Store"
    STORE_SLUG = "example_store"
    STATE_CODE = "MD"
    START_ZIP = "21701"

    # Set to the store's real (lat, lon) to measure from its exact location.
    # None measures from the centroid of START_ZIP's polygon.
    ORIGIN_LATLON: Optional[Tuple[float, float]] = None

    # None downloads (and caches) the Census ZCTA file. Point this at a local
    # GeoJSON / shapefile / .zip to use your own boundaries instead.
    POLYGON_SOURCE: Optional[PathInput] = None

    FEE_SCHEDULE = example_fee_schedule()

    # Flat-fee override zones, priced regardless of distance from the store.
    # Each one prices every ZIP whose centroid is within radius_miles of the
    # center (matching GEO_SETTINGS.primary_metric) at one flat fee. Example:
    #
    #     town_lat, town_lon = latlon_for_zip("21740")
    #     OVERRIDE_ZONES = (
    #         OverrideZone(
    #             name="Hagerstown",
    #             center_lat=town_lat,
    #             center_lon=town_lon,
    #             radius_miles=10.0,
    #             fee=4.99,
    #             min_order=50.0,
    #         ),
    #     )
    OVERRIDE_ZONES: Tuple[OverrideZone, ...] = ()

    GEO_SETTINGS = GeoZoneSettings(
        primary_metric="centroid",
        emit_clipped=True,
        max_extent_miles=None,
    )
    PRESERVE_ZIP_DISPLAY = True
    CONSOLE_LOG_LEVEL = "INFO"
    FILE_LOG_LEVEL = "DEBUG"
    # -----------------------------------------------------------------------

    output_dir = output_directory(STATE_CODE)
    log_file = output_dir / f"{STORE_SLUG}_geo_trace.log"
    setup_logging(
        console_level=CONSOLE_LOG_LEVEL,
        log_file=log_file,
        file_level=FILE_LOG_LEVEL,
    )

    LOGGER.info("Starting GeoJSON fee-zone build for %s", STORE_NAME)
    LOGGER.info("Postal-code metadata source: GeoNames, https://www.geonames.org/")
    LOGGER.info("ZIP polygon source: US Census 2020 ZCTA cartographic boundaries")

    result = build_fee_zones(
        state_code=STATE_CODE,
        start_zip=START_ZIP,
        schedule=FEE_SCHEDULE,
        settings=GEO_SETTINGS,
        polygon_source=POLYGON_SOURCE,
        origin_latlon=ORIGIN_LATLON,
        override_zones=OVERRIDE_ZONES,
    )
    written = save_outputs(
        result,
        store_slug=STORE_SLUG,
        output_dir=output_dir,
        preserve_zip_display=PRESERVE_ZIP_DISPLAY,
    )

    grouped = build_grouped_fees(result.gdf)
    print(f"\n{STORE_NAME} delivery-fee tiers:\n")
    print(grouped.to_string(index=False))
    print(f"\n{len(result.gdf)} ZIPs priced. Files written:")
    for key, path in written.items():
        print(f"  {key:16} {path}")
    LOGGER.info("Finished GeoJSON fee-zone build for %s", STORE_NAME)


if __name__ == "__main__":
    main()
