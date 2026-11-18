import numpy as np
import pandas as pd

from spendkit.parsing import parse_dates, parse_number, parse_vat


def test_numbers():
    assert parse_number("5,266.63") == 5266.63
    assert parse_number("1.234,56") == 1234.56
    assert parse_number("(120)") == -120
    assert parse_number("120-") == -120
    assert parse_number("-45.5") == -45.5
    assert parse_number("1,234") == 1234
    assert np.isnan(parse_number(""))
    assert np.isnan(parse_number(None))


def test_vat_formats():
    assert parse_vat("15") == 0.15
    assert parse_vat("0.15") == 0.15
    assert parse_vat("15%") == 0.15
    assert parse_vat("0") == 0.0          # real zero-rated, NOT blank
    assert np.isnan(parse_vat(""))
    assert parse_vat(None, 0.15) == 0.15


def test_dates_junk_years_removed():
    s = pd.Series(["2024-03-29", "27-Mar-24", "2018-07-23", "garbage", None])
    d = parse_dates(s, valid_years=(2023, 2025))
    assert d.iloc[0] == pd.Timestamp("2024-03-29")
    assert d.iloc[1] == pd.Timestamp("2024-03-27")
    assert d.iloc[2:].isna().all()