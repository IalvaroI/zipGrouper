"""
Build ZIP-code distance zones for any US state.

The script uses pgeocode's built-in US ZIP-code dataset, calculates each ZIP's
distance from a chosen starting ZIP, assigns every ZIP to a mileage band, and
writes two CSV files to a folder named for the state code:

1. A detailed ZIP-level CSV with distance and zone columns.
2. A grouped zone CSV with one row per mileage band.

It also writes a trace log showing what data was loaded, how many rows were
kept or removed at each filter, and what files were written. The trace log is
stored in the same state-code folder as the CSV files.

Change the settings in the CONFIGURATION section at the bottom of this file to
run the same workflow for another state.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pgeocode
from sklearn.neighbors import BallTree

# Funny Enough, had to learn this here https://en.wikipedia.org/wiki/Earth_radius
EARTH_RADIUS_MILES = 3958
ZIP_COLUMNS = ["postal_code", "place_name", "state_code", "latitude", "longitude"]
LOGGER = logging.getLogger("zip_zones")
PathInput = Union[str, Path]


@dataclass(frozen=True)
class ZoneSettings:
    """
    Configuration for the distance-band labels.

    first_zone_miles:
        The upper bound for the first band. With the default value, ZIPs from
        0 through 10 miles are labeled "0-10 Miles".

    zone_step_miles:
        The width of every band after the first one. With the default value,
        the next labels are "10.1-15 Miles", "15.1-20 Miles", etc.

    label_gap_miles:
        The display-only gap between labels. A value of 0.1 creates labels like
        "10.1-15 Miles". Distances greater than 10 still go into the next band;
        the gap is only for matching the requested wording.
    """

    first_zone_miles: float = 10.0
    zone_step_miles: float = 5.0
    label_gap_miles: float = 0.1

    def __post_init__(self) -> None:
        if self.first_zone_miles <= 0:
            raise ValueError("first_zone_miles must be greater than 0")
        if self.zone_step_miles <= 0:
            raise ValueError("zone_step_miles must be greater than 0")
        if self.label_gap_miles < 0:
            raise ValueError("label_gap_miles cannot be negative")
        if self.label_gap_miles >= self.zone_step_miles:
            raise ValueError("label_gap_miles must be smaller than zone_step_miles")


def setup_logging(
    console_level: str = "INFO",
    log_file: Optional[PathInput] = "zip_zone_trace.log",
    file_level: str = "DEBUG",
) -> None:
    """
    Configure trace logging for the whole script.

    console_level controls what appears in the terminal. log_file receives the
    more detailed trace, including sample rows and column lists. Set log_file to
    None if you only want terminal logging. Unknown level names raise ValueError.
    """

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level.strip().upper())
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(file_level.strip().upper())
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)

        LOGGER.info("Trace log will be written to %s", log_path)


def log_dataframe_overview(
    stage_name: str,
    df: pd.DataFrame,
    sample_rows: int = 5,
) -> None:
    """
    Write shape, columns, and a small sample for a DataFrame.

    Sample rows are logged at DEBUG level so the file log shows what the script
    is pulling and transforming without flooding the terminal.
    """

    LOGGER.debug("%s row count: %s", stage_name, len(df))
    LOGGER.debug("%s columns: %s", stage_name, list(df.columns))

    if df.empty:
        LOGGER.debug("%s sample: DataFrame is empty", stage_name)
        return

    LOGGER.debug(
        "%s first %s rows:\n%s",
        stage_name,
        min(sample_rows, len(df)),
        df.head(sample_rows).to_string(index=False),
    )


def normalize_zip(postal_code: str) -> str:
    """Return a five-character ZIP string, tolerating values like 7060.0."""

    postal_code_text = str(postal_code).strip()
    if postal_code_text.endswith(".0"):
        postal_code_text = postal_code_text[:-2]
    return postal_code_text.zfill(5)


def normalize_state_code(state_code: str) -> str:
    """Return a two-letter uppercase state code, e.g. " nj " -> "NJ"."""

    return str(state_code).strip().upper()


def format_miles(value: float) -> str:
    """Format mileage without trailing zeros: 10.0 -> "10", 10.1 -> "10.1"."""

    return f"{value:.1f}".rstrip("0").rstrip(".")


def load_us_zip_data() -> pd.DataFrame:
    """Load raw US ZIP-code data from pgeocode."""

    LOGGER.info("Loading US ZIP data from pgeocode.Nominatim('us')")
    LOGGER.debug("pgeocode module path: %s", getattr(pgeocode, "__file__", "unknown"))
    LOGGER.debug("pgeocode version: %s", getattr(pgeocode, "__version__", "unknown"))

    raw_df = pgeocode.Nominatim("us")._data.copy()

    LOGGER.info("Loaded %s raw ZIP rows from pgeocode", len(raw_df))
    log_dataframe_overview("Raw pgeocode data", raw_df)
    return raw_df


def clean_zip_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and deduplicate ZIP-code data.

    The BallTree distance calculation needs one latitude/longitude pair per
    ZIP. Some datasets can contain duplicate ZIP rows, so this function averages
    coordinates for duplicates and keeps the first place/state label.
    """

    LOGGER.info("Cleaning raw ZIP data")
    required_columns = {"postal_code", "latitude", "longitude", "state_code"}
    missing_columns = sorted(required_columns.difference(raw_df.columns))
    if missing_columns:
        raise ValueError(f"Raw ZIP data is missing required columns: {missing_columns}")

    starting_rows = len(raw_df)
    df = raw_df.dropna(subset=["postal_code", "latitude", "longitude"]).copy()
    LOGGER.info(
        "Dropped %s rows missing postal_code, latitude, or longitude",
        starting_rows - len(df),
    )

    df["postal_code"] = df["postal_code"].apply(normalize_zip)
    valid_zip_mask = df["postal_code"].str.fullmatch(r"\d{5}")
    LOGGER.info(
        "Dropped %s rows whose postal_code was not a 5-digit ZIP",
        len(df) - int(valid_zip_mask.sum()),
    )
    df = df[valid_zip_mask].copy()
    df["state_code"] = df["state_code"].apply(normalize_state_code)

    available_columns = [column for column in ZIP_COLUMNS if column in df.columns]
    LOGGER.info("Keeping columns for downstream processing: %s", available_columns)
    df = df[available_columns]

    LOGGER.info(
        "Found %s duplicate ZIP rows before deduplication",
        int(df["postal_code"].duplicated().sum()),
    )

    aggregation = {"latitude": "mean", "longitude": "mean"}
    if "place_name" in df.columns:
        aggregation["place_name"] = "first"
    aggregation["state_code"] = "first"

    cleaned_df = df.groupby("postal_code", as_index=False).agg(aggregation)
    LOGGER.info(
        "Cleaned ZIP data from %s raw rows to %s unique ZIP rows",
        starting_rows,
        len(cleaned_df),
    )
    log_dataframe_overview("Cleaned ZIP data", cleaned_df)
    return cleaned_df


def get_origin_row(zip_df: pd.DataFrame, start_zip: str, state_code: str) -> pd.Series:
    """
    Find and validate the starting ZIP row.

    The distance bands only make sense if the origin ZIP exists and belongs to
    the state being exported. This validation catches typos like using a New
    Jersey ZIP while exporting Pennsylvania.
    """

    start_zip = normalize_zip(start_zip)
    state_code = normalize_state_code(state_code)
    LOGGER.info("Validating origin ZIP %s for state %s", start_zip, state_code)

    origin_matches = zip_df.loc[zip_df["postal_code"] == start_zip]
    if origin_matches.empty:
        LOGGER.error("Starting ZIP %s was not found in cleaned data", start_zip)
        raise ValueError(f"Starting ZIP {start_zip} was not found")

    origin = origin_matches.iloc[0]
    origin_state = normalize_state_code(origin.get("state_code", ""))
    if origin_state != state_code:
        LOGGER.error(
            "Starting ZIP %s belongs to %s, not %s", start_zip, origin_state, state_code
        )
        raise ValueError(f"Starting ZIP {start_zip} is in {origin_state}, not {state_code}")

    LOGGER.info(
        "Origin ZIP accepted: %s, %s, %s at lat=%s lon=%s",
        origin["postal_code"],
        origin.get("place_name", ""),
        origin_state,
        origin["latitude"],
        origin["longitude"],
    )
    return origin


def filter_state_zips(zip_df: pd.DataFrame, state_code: str) -> pd.DataFrame:
    """Keep the ZIP codes for one state from the nationwide dataset."""

    state_code = normalize_state_code(state_code)
    LOGGER.info(
        "Filtering %s cleaned ZIP rows to state_code == %s", len(zip_df), state_code
    )
    state_df = zip_df.loc[zip_df["state_code"] == state_code].copy()
    if state_df.empty:
        LOGGER.error("State filter produced no rows for %s", state_code)
        raise ValueError(f"No ZIP codes found for state {state_code}")

    LOGGER.info(
        "State filter kept %s ZIP rows for %s and removed %s rows",
        len(state_df),
        state_code,
        len(zip_df) - len(state_df),
    )
    log_dataframe_overview(f"{state_code} ZIP data", state_df)
    return state_df


def query_distances_from_origin(
    zip_df: pd.DataFrame,
    origin: pd.Series,
    radius_miles: Optional[float] = None,
) -> pd.DataFrame:
    """
    Add distance_miles to ZIP rows, sorted from nearest to farthest.

    radius_miles:
        None means return every ZIP in zip_df. A number means return only ZIPs
        within that many miles of the starting ZIP.
    """

    if zip_df.empty:
        LOGGER.error("Distance query received an empty ZIP DataFrame")
        raise ValueError("Cannot calculate distances for an empty ZIP DataFrame")

    LOGGER.info(
        "Calculating distances from origin ZIP %s using radius_miles=%s",
        origin["postal_code"],
        radius_miles,
    )
    LOGGER.debug(
        "Origin coordinates used for distance query: lat=%s lon=%s",
        origin["latitude"],
        origin["longitude"],
    )

    # The haversine metric expects [latitude, longitude] in radians.
    LOGGER.info("Building BallTree for %s ZIP coordinate pairs", len(zip_df))
    tree = BallTree(
        np.radians(zip_df[["latitude", "longitude"]].to_numpy()), metric="haversine"
    )
    origin_radians = np.radians([[origin["latitude"], origin["longitude"]]])

    if radius_miles is None:
        LOGGER.info("No radius limit set; querying every ZIP in the filtered state")
        distances, indices = tree.query(origin_radians, k=len(zip_df))
    else:
        radius_radians = radius_miles / EARTH_RADIUS_MILES
        LOGGER.info(
            "Radius limit set to %s miles (%s radians)", radius_miles, radius_radians
        )
        indices, distances = tree.query_radius(
            origin_radians,
            r=radius_radians,
            return_distance=True,
            sort_results=True,
        )

    result = zip_df.iloc[indices[0]].reset_index(drop=True)
    result["distance_miles"] = distances[0] * EARTH_RADIUS_MILES
    LOGGER.info(
        "Distance query returned %s ZIP rows with distance range %.2f-%.2f miles",
        len(result),
        result["distance_miles"].min(),
        result["distance_miles"].max(),
    )
    log_dataframe_overview("ZIP data with calculated distances", result)
    return result


def build_zone_label(distance_miles: float, settings: ZoneSettings) -> tuple[int, str]:
    """
    Convert one distance value into a zone order and label.

    With default settings:
        0.0 through 10.0 miles -> (0, "0-10 Miles")
        10.0001 through 15.0 miles -> (1, "10.1-15 Miles")
        15.0001 through 20.0 miles -> (2, "15.1-20 Miles")
    """

    first = settings.first_zone_miles
    step = settings.zone_step_miles

    if distance_miles <= first:
        return 0, f"0-{format_miles(first)} Miles"

    zone_order = int(np.ceil((distance_miles - first) / step))
    lower_miles = first + (zone_order - 1) * step + settings.label_gap_miles
    upper_miles = first + zone_order * step
    return zone_order, f"{format_miles(lower_miles)}-{format_miles(upper_miles)} Miles"


def assign_distance_zones(
    zip_df: pd.DataFrame,
    settings: ZoneSettings,
) -> pd.DataFrame:
    """
    Add zone_order and zone_name columns based on distance_miles.

    zone_order is numeric so CSV rows can be sorted correctly. zone_name is the
    human-readable label requested for reporting.
    """

    LOGGER.info(
        "Assigning distance zones using first_zone_miles=%s, "
        "zone_step_miles=%s, label_gap_miles=%s",
        settings.first_zone_miles,
        settings.zone_step_miles,
        settings.label_gap_miles,
    )
    result = zip_df.copy()
    zones = result["distance_miles"].apply(lambda miles: build_zone_label(miles, settings))
    result["zone_order"] = zones.str[0]
    result["zone_name"] = zones.str[1]

    result = result.sort_values(
        ["zone_order", "distance_miles", "postal_code"]
    ).reset_index(drop=True)

    zone_counts = (
        result.groupby(["zone_order", "zone_name"], as_index=False)
        .agg(zip_count=("postal_code", "size"))
        .sort_values("zone_order")
    )
    LOGGER.info(
        "Assigned %s ZIP rows into %s distance zones", len(result), len(zone_counts)
    )
    LOGGER.debug("Distance zone counts:\n%s", zone_counts.to_string(index=False))
    log_dataframe_overview("ZIP data with assigned zones", result)
    return result


def build_grouped_zones(zip_df: pd.DataFrame) -> pd.DataFrame:
    """Build one summary row per distance zone, listing every ZIP in it."""

    grouped = (
        zip_df.groupby(["zone_order", "zone_name"], as_index=False)
        .agg(
            zip_count=("postal_code", "size"),
            zip_codes=("postal_code", ", ".join),
        )
        .sort_values("zone_order")
        .reset_index(drop=True)
    )
    LOGGER.info("Built grouped zone summary with %s rows", len(grouped))
    log_dataframe_overview("Grouped zone summary", grouped)
    return grouped


def build_zip_zones(
    state_code: str,
    start_zip: str,
    radius_miles: Optional[float] = None,
    zone_settings: Optional[ZoneSettings] = None,
) -> pd.DataFrame:
    """
    End-to-end ZIP-zone builder.

    This is the main reusable function for other code. Pass a state abbreviation
    and a starting ZIP from that state, and it returns the detailed ZIP-level
    DataFrame with distance and zone columns.
    """

    state_code = normalize_state_code(state_code)
    start_zip = normalize_zip(start_zip)
    LOGGER.info(
        "Starting ZIP-zone build for state=%s start_zip=%s radius_miles=%s",
        state_code,
        start_zip,
        radius_miles,
    )
    clean_df = clean_zip_data(load_us_zip_data())
    origin = get_origin_row(clean_df, start_zip, state_code)
    state_df = filter_state_zips(clean_df, state_code)
    distance_df = query_distances_from_origin(state_df, origin, radius_miles)
    result = assign_distance_zones(distance_df, zone_settings or ZoneSettings())
    LOGGER.info("Finished ZIP-zone build for %s: %s ZIP rows", state_code, len(result))
    return result


def output_directory(state_code: str, base_dir: PathInput = ".") -> Path:
    """Create and return the output folder for a state: CA -> ./CA."""

    output_path = Path(base_dir) / normalize_state_code(state_code)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_zone_csvs(
    detail_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    state_code: str,
    output_dir: Optional[PathInput] = None,
    preserve_zip_display: bool = True,
) -> tuple[str, str]:
    """
    Write the detailed and grouped DataFrames to CSV files.

    With preserve_zip_display, detailed ZIP cells are written as ="01001" so
    spreadsheet apps like Excel keep the leading zero instead of guessing the
    column is numeric. Returns the two filenames.
    """

    prefix = f"{normalize_state_code(state_code).lower()}_zip_zones"
    output_path = Path(output_dir or ".")
    output_path.mkdir(parents=True, exist_ok=True)

    detail_filename = output_path / f"{prefix}.csv"
    grouped_filename = output_path / f"{prefix}_by_zone.csv"

    export_df = detail_df.copy()
    if preserve_zip_display:
        export_df["postal_code"] = export_df["postal_code"].apply(
            lambda postal_code: f'="{normalize_zip(postal_code)}"'
        )

    LOGGER.info("Writing %s detailed ZIP rows to %s", len(detail_df), detail_filename)
    export_df.to_csv(detail_filename, index=False)
    LOGGER.info("Writing %s grouped zone rows to %s", len(grouped_df), grouped_filename)
    grouped_df.to_csv(grouped_filename, index=False)

    LOGGER.info("CSV export complete")
    return str(detail_filename), str(grouped_filename)


def main() -> None:
    """
    Run the CSV export using the settings below.

    To use this for another state, change STATE_CODE and START_ZIP. START_ZIP
    must be a ZIP code located in that state because all mileage bands are
    measured from that ZIP.
    """

    # CONFIGURATION
    STATE_CODE = "MD"
    START_ZIP = "21701"
    CONSOLE_LOG_LEVEL = "INFO"
    FILE_LOG_LEVEL = "DEBUG"

    # True writes ZIPs as ="01001" so Excel keeps leading zeros (useful for
    # states like NJ). False writes a raw CSV.
    PRESERVE_ZIP_DISPLAY = False

    # Use None for every ZIP in the state. Use a number, such as 50, to limit
    # output to ZIPs within that many miles from START_ZIP.
    RADIUS_MILES = None

    # These settings produce labels like:
    # 0-10 Miles, 10.1-15 Miles, 15.1-20 Miles, 20.1-25 Miles, etc.
    ZONE_SETTINGS = ZoneSettings(
        first_zone_miles=10.0,
        zone_step_miles=5.0,
        label_gap_miles=0.1,
    )

    OUTPUT_DIR = output_directory(STATE_CODE)
    LOG_FILE = OUTPUT_DIR / f"zip_zone_trace_{normalize_state_code(STATE_CODE)}.log"

    setup_logging(
        console_level=CONSOLE_LOG_LEVEL,
        log_file=LOG_FILE,
        file_level=FILE_LOG_LEVEL,
    )
    LOGGER.info("Starting ZIP-zone CSV export")
    LOGGER.info(
        "Configuration: STATE_CODE=%s, START_ZIP=%s, RADIUS_MILES=%s, "
        "PRESERVE_ZIP_DISPLAY=%s",
        STATE_CODE,
        START_ZIP,
        RADIUS_MILES,
        PRESERVE_ZIP_DISPLAY,
    )
    LOGGER.info("Zone settings: %s", ZONE_SETTINGS)

    details = build_zip_zones(
        state_code=STATE_CODE,
        start_zip=START_ZIP,
        radius_miles=RADIUS_MILES,
        zone_settings=ZONE_SETTINGS,
    )
    grouped = build_grouped_zones(details)
    detail_filename, grouped_filename = save_zone_csvs(
        detail_df=details,
        grouped_df=grouped,
        state_code=STATE_CODE,
        output_dir=OUTPUT_DIR,
        preserve_zip_display=PRESERVE_ZIP_DISPLAY,
    )

    print(details.head(25))
    print(f"\nSaved {len(details)} ZIP rows to {detail_filename}")

    print("\nZIPs grouped by zone:\n")
    print(grouped)
    print(f"\nSaved {len(grouped)} grouped zones to {grouped_filename}")
    LOGGER.info("Finished ZIP-zone CSV export")


if __name__ == "__main__":
    main()
