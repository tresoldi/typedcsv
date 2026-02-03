import io
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest

_PKG_INIT = Path(__file__).resolve().parents[1] / "typedcsv" / "__init__.py"
_SPEC = importlib.util.spec_from_file_location("typedcsv_pkg", _PKG_INIT)
assert _SPEC and _SPEC.loader
typedcsv = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = typedcsv
_SPEC.loader.exec_module(typedcsv)


def read_rows(text, **kwargs):
    f = io.StringIO(text)
    return list(typedcsv.reader(f, **kwargs))


def read_dicts(text, **kwargs):
    f = io.StringIO(text)
    return list(typedcsv.DictReader(f, **kwargs))


def test_header_parsing_conflict_sigil_and_explicit():
    text = "age#:int\n1\n"
    with pytest.raises(typedcsv.TypedCSVError) as exc:
        read_rows(text)
    assert "both explicit type and sigil" in str(exc.value)


def test_header_parsing_unknown_explicit_type_is_error():
    text = "age:integer\n1\n"
    with pytest.raises(typedcsv.TypedCSVError) as exc:
        read_rows(text)
    assert "Unknown explicit type" in str(exc.value)


def test_header_parsing_duplicate_logical_names():
    text = "age#,age:int\n1,2\n"
    with pytest.raises(typedcsv.TypedCSVError) as exc:
        read_rows(text)
    assert "Duplicate logical column name" in str(exc.value)


def test_header_parsing_malformed_validators_clause():
    text = "age# [min=0\n1\n"
    with pytest.raises(typedcsv.TypedCSVError) as exc:
        read_rows(text)
    assert "Malformed validators clause" in str(exc.value)


def test_validators_enforcement_int_min_max_in():
    text = "age# [min=0 max=10 in=1|2|3]\n2\n4\n"
    f = io.StringIO(text)
    r = typedcsv.reader(f)
    assert next(r)[0] == 2
    with pytest.raises(typedcsv.TypedCSVError):
        next(r)


def test_validators_unknown_key_error():
    text = "age# [nope=1]\n1\n"
    with pytest.raises(typedcsv.TypedCSVError) as exc:
        read_rows(text)
    assert "not allowed" in str(exc.value)


def test_missing_values_behavior():
    text = "name$,age#\n,\n"
    rows = read_rows(text)
    assert rows[0][0] == ""
    assert rows[0][1] is None


def test_missing_values_skip_validation():
    text = "age# [min=10]\n\n"
    rows = read_rows(text)
    assert rows[0][0] is None


def test_bool_parsing_variants():
    text = "b?\ntrue\nFALSE\nT\nf\nYes\nno\n1\n0\n"
    rows = read_rows(text)
    assert [r[0] for r in rows] == [True, False, True, False, True, False, True, False]


def test_datetime_parsing_and_min_max():
    text = "ts@ [min=2020-01-01T00:00:00 max=2020-12-31T23:59:59]\n2020-06-01T12:00:00\n2019-01-01T00:00:00\n"
    f = io.StringIO(text)
    r = typedcsv.reader(f)
    assert isinstance(next(r)[0], datetime)
    with pytest.raises(typedcsv.TypedCSVError):
        next(r)


def test_re_fullmatch_for_str():
    text = "code$ [re=^[A-Z]{3}\\d{2}$]\nABC12\nAB12\n"
    f = io.StringIO(text)
    r = typedcsv.reader(f)
    assert next(r)[0] == "ABC12"
    with pytest.raises(typedcsv.TypedCSVError):
        next(r)


def test_error_context_fields_for_parse_error():
    text = "age#\nnope\n"
    with pytest.raises(typedcsv.TypedCSVError) as exc:
        read_rows(text)
    err = exc.value
    assert err.row == 2
    assert err.col == 0
    assert err.column == "age"
    assert err.header == "age#"
    assert err.value == "nope"
    assert "Parse failed" in err.reason


def test_writer_canonicalization_and_float_repr():
    f = io.StringIO()
    w = typedcsv.DictWriter(f, fieldnames=["i#", "b?", "f%", "t@", "s$"])
    w.writeheader()
    w.writerow({
        "i": 1,
        "b": True,
        "f": 1.5,
        "t": datetime(2021, 5, 1, 12, 30),
        "s": "x",
    })
    out = f.getvalue().splitlines()
    assert out[0] == "i#,b?,f%,t@,s$"
    assert out[1].startswith("1,true,1.5,2021-05-01T12:30:00,x")


def test_round_trip_dictwriter_to_dictreader():
    f = io.StringIO()
    w = typedcsv.DictWriter(f, fieldnames=["id#", "name$", "active?", "created@"])
    w.writeheader()
    w.writerow({"id": 1, "name": "Alice", "active": True, "created": datetime(2021, 5, 1, 12, 30)})
    w.writerow({"id": 2, "name": "Bob", "active": False, "created": None})

    f.seek(0)
    rows = list(typedcsv.DictReader(f))
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["active"] is True
    assert rows[0]["created"] == datetime(2021, 5, 1, 12, 30)
    assert rows[1]["created"] is None


def test_infer_types_untyped_only_without_validators():
    text = "a,b [minlen=1]\n1,x\n2,y\n"
    rows = read_rows(text, infer_types=True)
    assert rows[0][0] == 1
    assert rows[0][1] == "x"
