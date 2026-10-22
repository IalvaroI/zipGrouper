"""
Unit tests for the delivery-fee engine in ``zipGrouperGeo.py``.

These cover the pure, standard-library-only pieces: fee-band assignment, the
open-ended per-mile tier, rounding modes, schedule/tier validation, and the
flat-fee :class:`OverrideZone`. They intentionally do NOT import the geospatial
stack (geopandas / shapely / pgeocode), which ``zipGrouperGeo`` only imports
lazily inside its data and geometry functions.

Run with::

    python3 test_fees.py
"""

import unittest

from zipGrouperGeo import (
    FeeSchedule,
    FeeTier,
    OverrideZone,
    example_fee_schedule,
    format_miles,
    format_money,
    normalize_zip,
)


class TestExampleSchedule(unittest.TestCase):
    """The example schedule must price each documented example exactly."""

    def setUp(self):
        self.schedule = example_fee_schedule()

    def test_band_boundaries_are_listed(self):
        self.assertEqual(self.schedule.boundaries(), [10.0, 15.0, 20.0, 25.0])

    def test_free_band(self):
        for miles in (0.0, 5.0, 10.0):
            quote = self.schedule.quote(miles)
            self.assertEqual(quote.fee, 0.0)
            self.assertEqual(quote.min_order, 50.0)
            self.assertEqual(quote.tier_order, 0)

    def test_just_past_ten_is_next_band(self):
        quote = self.schedule.quote(10.1)
        self.assertEqual(quote.fee, 5.0)
        self.assertEqual(quote.min_order, 50.0)
        self.assertEqual(quote.tier_order, 1)

    def test_fifteen_to_twenty_band(self):
        quote = self.schedule.quote(17.0)
        self.assertEqual(quote.fee, 10.0)
        self.assertEqual(quote.min_order, 50.0)

    def test_twenty_to_twentyfive_band_raises_minimum(self):
        quote = self.schedule.quote(22.5)
        self.assertEqual(quote.fee, 15.0)
        self.assertEqual(quote.min_order, 75.0)

    def test_exactly_twentyfive_stays_in_closed_band(self):
        # The boundary epsilon keeps 25.0 out of the open per-mile tier.
        quote = self.schedule.quote(25.0)
        self.assertEqual(quote.fee, 15.0)
        self.assertEqual(quote.tier_order, 3)

    def test_open_tier_adds_one_dollar_per_mile(self):
        # The documented example: 26 miles -> $16.00.
        self.assertEqual(self.schedule.quote(26.0).fee, 16.0)
        # And it keeps scaling: 30 miles -> $15 + 5 * $1 = $20.00.
        self.assertEqual(self.schedule.quote(30.0).fee, 20.0)
        self.assertEqual(self.schedule.quote(30.0).min_order, 75.0)

    def test_open_tier_continuous_partial_mile(self):
        # Continuous rounding charges exact partial miles: 25.5 -> +$0.50.
        self.assertEqual(self.schedule.quote(25.5).fee, 15.5)


class TestRoundingModes(unittest.TestCase):
    """The open-tier surcharge honours the configured rounding mode."""

    def _schedule(self, mode):
        return FeeSchedule(
            tiers=(
                FeeTier(upper_miles=25.0, fee=24.99, min_order=75.0, label="<=25"),
                FeeTier(
                    upper_miles=None,
                    fee=24.99,
                    min_order=75.0,
                    per_mile_over=2.0,
                    over_threshold_miles=25.0,
                    label="open",
                ),
            ),
            over_rounding=mode,
        )

    def test_ceil_rounds_started_mile_up(self):
        self.assertEqual(self._schedule("ceil").quote(25.1).fee, 26.99)

    def test_floor_drops_partial_mile(self):
        self.assertEqual(self._schedule("floor").quote(25.9).fee, 24.99)

    def test_round_to_nearest_mile(self):
        self.assertEqual(self._schedule("round").quote(26.4).fee, 26.99)
        self.assertEqual(self._schedule("round").quote(26.6).fee, 28.99)

    def test_continuous_is_exact(self):
        self.assertEqual(self._schedule("continuous").quote(26.5).fee, 27.99)


class TestScheduleValidation(unittest.TestCase):
    def test_empty_schedule_rejected(self):
        with self.assertRaises(ValueError):
            FeeSchedule(tiers=())

    def test_non_increasing_bounds_rejected(self):
        with self.assertRaises(ValueError):
            FeeSchedule(
                tiers=(
                    FeeTier(upper_miles=15.0, fee=0.0, min_order=0.0),
                    FeeTier(upper_miles=10.0, fee=0.0, min_order=0.0),
                    FeeTier(upper_miles=None, fee=0.0, min_order=0.0),
                )
            )

    def test_tier_after_open_tier_rejected(self):
        with self.assertRaises(ValueError):
            FeeSchedule(
                tiers=(
                    FeeTier(upper_miles=None, fee=0.0, min_order=0.0),
                    FeeTier(upper_miles=10.0, fee=0.0, min_order=0.0),
                )
            )

    def test_unknown_rounding_mode_rejected(self):
        with self.assertRaises(ValueError):
            FeeSchedule(
                tiers=(FeeTier(upper_miles=None, fee=0.0, min_order=0.0),),
                over_rounding="bankers",
            )

    def test_per_mile_requires_threshold(self):
        with self.assertRaises(ValueError):
            FeeTier(upper_miles=None, fee=0.0, min_order=0.0, per_mile_over=2.0)

    def test_negative_values_rejected(self):
        with self.assertRaises(ValueError):
            FeeTier(upper_miles=10.0, fee=-1.0, min_order=0.0)
        with self.assertRaises(ValueError):
            FeeTier(upper_miles=10.0, fee=0.0, min_order=-1.0)


class TestOverrideZone(unittest.TestCase):
    def _zone(self, **kw):
        defaults = dict(
            name="Downtown",
            center_lat=39.4143,
            center_lon=-77.4105,
            radius_miles=10.0,
            fee=4.99,
            min_order=45.0,
        )
        defaults.update(kw)
        return OverrideZone(**defaults)

    def test_covers_inside_and_on_boundary(self):
        zone = self._zone()
        self.assertTrue(zone.covers(0.0))
        self.assertTrue(zone.covers(9.9))
        self.assertTrue(zone.covers(10.0))  # boundary is inclusive

    def test_does_not_cover_outside(self):
        self.assertFalse(self._zone().covers(10.5))

    def test_default_label(self):
        self.assertEqual(self._zone(label="").display_label(), "Downtown flat $4.99")

    def test_custom_label_is_kept(self):
        self.assertEqual(self._zone(label="Downtown promo").display_label(), "Downtown promo")

    def test_validation(self):
        with self.assertRaises(ValueError):
            self._zone(name="")
        with self.assertRaises(ValueError):
            self._zone(radius_miles=0.0)
        with self.assertRaises(ValueError):
            self._zone(fee=-1.0)
        with self.assertRaises(ValueError):
            self._zone(min_order=-1.0)


class TestFormatters(unittest.TestCase):
    def test_normalize_zip_pads_and_trims(self):
        self.assertEqual(normalize_zip("1001"), "01001")
        self.assertEqual(normalize_zip("01001.0"), "01001")
        self.assertEqual(normalize_zip(" 2134 "), "02134")

    def test_format_miles_drops_trailing_zeros(self):
        self.assertEqual(format_miles(10.0), "10")
        self.assertEqual(format_miles(10.1), "10.1")

    def test_format_money(self):
        self.assertEqual(format_money(4.99), "$4.99")
        self.assertEqual(format_money(1234.5), "$1,234.50")


if __name__ == "__main__":
    unittest.main(verbosity=2)
