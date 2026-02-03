"""
typedcsv.py — “Typed CSVs” with header-embedded types + validators (stdlib-only).

Contract (v0, locked):
- Typing + validators live ONLY in the header row.
- Suffix sigils OR explicit ":type" (not both).
- Suffix-only sigils:
    # -> int, % -> float, ? -> bool, @ -> datetime, $ -> str
  Explicit types: :int, :float, :bool, :datetime, :str
- Untyped defaults to str.
- Validators: optional bracket clause after the type marker:
    col# [min=0 max=10]
    col:str [in=A|B|C]
    col$ [re=^[A-Z]{3}\d{2}$]
  Parsed as space-separated key=value pairs inside [ ... ].
  Values with spaces must be double-quoted.
  "re=" uses re.fullmatch.
- Missing values: empty string cell "" is missing:
    str columns -> "" (kept)
    non-str columns -> None
  Missing values skip validation.
- Errors: raise immediately with TypedCSVError including row/col/context.
- Writing: None -> "", bool -> true/false, datetime -> isoformat(), float -> repr(f),
  header preserved exactly as provided to DictWriter(fieldnames).

API (csv-like, but typed readers consume header):
- reader(f, ...) -> iterator of typed list rows
- DictReader(f, ...) -> iterator of typed dict rows
- writer(f, ...) -> basic writer wrapper for lists (no schema enforcement)
- DictWriter(f, fieldnames, ...) -> typed-aware dict writer; preserve typed fieldnames

Python: 3.10+
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


# ----------------------------
# Exceptions
# ----------------------------

class TypedCSVError(ValueError):
    """Raised on header parse, cell parse, or validation failures with context."""

    def __init__(
        self,
        *,
        row: int,
        col: int,
        column: str,
        header: str,
        value: str,
        reason: str,
    ) -> None:
        super().__init__(f"TypedCSVError(row={row}, col={col}, column={column!r}): {reason}")
        self.row = row          # 1-based row index in CSV (header row is 1)
        self.col = col          # 0-based column index
        self.column = column    # logical column name (stripped)
        self.header = header    # raw header cell text
        self.value = value      # raw cell text
        self.reason = reason


# ----------------------------
# Type dialect
# ----------------------------

TypeName = str  # "int" | "float" | "str" | "bool" | "datetime"

@dataclass(frozen=True)
class TypeDialect:
    sigils: Mapping[str, TypeName] = field(
        default_factory=lambda: {
            "#": "int",
            "%": "float",
            "?": "bool",
            "@": "datetime",
            "$": "str",
        }
    )
    explicit_sep: str = ":"
    validators_open: str = "["
    validators_close: str = "]"
    in_sep: str = "|"
    # bool parsing (case-insensitive)
    bool_true: Tuple[str, ...] = ("true", "t", "yes", "y", "1")
    bool_false: Tuple[str, ...] = ("false", "f", "no", "n", "0")
    datetime_parser: Callable[[str], datetime] = staticmethod(datetime.fromisoformat)
    datetime_formatter: Callable[[datetime], str] = staticmethod(lambda dt: dt.isoformat())


DEFAULT_TYPE_DIALECT = TypeDialect()

_EXPLICIT_TYPES: Tuple[TypeName, ...] = ("int", "float", "str", "bool", "datetime")


# ----------------------------
# Schema / column spec
# ----------------------------

@dataclass(frozen=True)
class ColumnSpec:
    name: str                  # logical name (sigil/type stripped)
    type_name: TypeName        # one of _EXPLICIT_TYPES
    raw_header: str            # raw header cell (including markers/validators)
    validators: Dict[str, str] # raw validator values (strings) from header
    # Parsers/formatters depend on type dialect + type_name
    parser: Callable[[str], Any]
    formatter: Callable[[Any], str]


@dataclass(frozen=True)
class Schema:
    columns: List[ColumnSpec]

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.columns]


# ----------------------------
# Header parsing helpers
# ----------------------------

def _split_validators(cell: str, td: TypeDialect) -> Tuple[str, Optional[str]]:
    """
    Split "prefix [k=v ...]" into ("prefix", "[k=v ...]") or ("prefix", None).
    Validator clause must be the last bracketed block (no nesting in v0).
    """
    s = cell.strip()
    if not s:
        return s, None
    # Find a trailing " [ ... ]" block
    # We allow whitespace before '['.
    idx = s.rfind(td.validators_open)
    if idx == -1:
        return s, None
    # If there is a '[', it must close at the end.
    if not s.endswith(td.validators_close):
        return s, None  # treat as no validators; later header parsing may reject if junk remains
    prefix = s[:idx].rstrip()
    validators = s[idx:].strip()
    return prefix, validators


def parse_validators(text: str, td: TypeDialect, *, row: int, col: int, column: str, header: str) -> Dict[str, str]:
    """
    Parse validators clause like: [min=0 max=10 re=^a+$ in=A|B|C]
    - Space-separated key=value pairs.
    - Values may be double-quoted if they contain spaces.
    """
    t = text.strip()
    if not (t.startswith(td.validators_open) and t.endswith(td.validators_close)):
        raise TypedCSVError(
            row=row, col=col, column=column, header=header, value="",
            reason=f"Malformed validators clause: {text!r}"
        )

    inner = t[1:-1].strip()
    if not inner:
        return {}

    # Tokenize into key=value "words", respecting quoted values
    # Simple state machine: read key= then read value (quoted or unquoted).
    i, n = 0, len(inner)
    out: Dict[str, str] = {}

    def skip_ws(j: int) -> int:
        while j < n and inner[j].isspace():
            j += 1
        return j

    i = skip_ws(i)
    while i < n:
        # read key up to '='
        eq = inner.find("=", i)
        if eq == -1:
            raise TypedCSVError(
                row=row, col=col, column=column, header=header, value="",
                reason=f"Invalid validator token (missing '=') near: {inner[i:]!r}"
            )
        key = inner[i:eq].strip()
        if not key:
            raise TypedCSVError(
                row=row, col=col, column=column, header=header, value="",
                reason=f"Invalid validator key near: {inner[i:]!r}"
            )
        j = eq + 1
        if j >= n:
            raise TypedCSVError(
                row=row, col=col, column=column, header=header, value="",
                reason=f"Validator {key!r} missing value"
            )

        # read value
        if inner[j] == '"':
            # quoted value
            j += 1
            start = j
            while j < n and inner[j] != '"':
                j += 1
            if j >= n:
                raise TypedCSVError(
                    row=row, col=col, column=column, header=header, value="",
                    reason=f"Unterminated quote in validator {key!r}"
                )
            value = inner[start:j]
            j += 1  # consume closing quote
        else:
            # unquoted value up to whitespace
            start = j
            while j < n and not inner[j].isspace():
                j += 1
            value = inner[start:j]

        if key in out:
            raise TypedCSVError(
                row=row, col=col, column=column, header=header, value="",
                reason=f"Duplicate validator key: {key!r}"
            )
        out[key] = value

        i = skip_ws(j)

    return out


def parse_header_cell(cell: str, td: TypeDialect, *, row: int, col: int) -> ColumnSpec:
    """
    Parse one header cell into ColumnSpec.
    Raises TypedCSVError on invalid syntax.
    """
    raw_header = cell
    prefix, vtext = _split_validators(cell, td)

    # Determine type marker: either explicit ":type" or trailing sigil
    type_name: TypeName = "str"
    base = prefix.strip()

    if not base:
        raise TypedCSVError(
            row=row, col=col, column="", header=raw_header, value="",
            reason="Empty header cell"
        )

    used_explicit = False

    # Explicit :type
    if td.explicit_sep in base:
        # Only allow a single explicit separator at the end: name:type
        # Names may contain ':' in theory, but we keep v0 strict and simple:
        # split on last ':' and interpret suffix as type token.
        name_part, type_part = base.rsplit(td.explicit_sep, 1)
        if type_part in _EXPLICIT_TYPES:
            used_explicit = True
            type_name = type_part
            base_name = name_part.strip()
        else:
            base_name = base  # treat as no explicit type; later may still have sigil
    else:
        base_name = base

    # Sigil suffix (only if explicit not used)
    if not used_explicit:
        if base_name and base_name[-1] in td.sigils:
            sig = base_name[-1]
            type_name = td.sigils[sig]
            base_name = base_name[:-1].rstrip()

    # Reject both explicit and sigil in same cell
    if used_explicit and (base.endswith(tuple(td.sigils.keys())) or (base_name and base_name[-1:] in td.sigils)):
        # More robust: detect if original base (without validators) ends in sigil while explicit parsed.
        if prefix.strip().endswith(tuple(td.sigils.keys())):
            raise TypedCSVError(
                row=row, col=col, column="", header=raw_header, value="",
                reason="Header uses both explicit type and sigil (not allowed)"
            )

    name = base_name.strip()
    if not name:
        raise TypedCSVError(
            row=row, col=col, column="", header=raw_header, value="",
            reason="Header name is empty after stripping type marker"
        )

    # Parse validators (if present)
    validators: Dict[str, str] = {}
    if vtext is not None:
        validators = parse_validators(vtext, td, row=row, col=col, column=name, header=raw_header)

    # Build parser/formatter
    parser, formatter = _get_type_codec(type_name, td)

    # Validate validators are allowed for the type (unknown keys error)
    _validate_validator_keys(type_name, validators, td, row=row, col=col, column=name, header=raw_header)

    return ColumnSpec(
        name=name,
        type_name=type_name,
        raw_header=raw_header,
        validators=validators,
        parser=parser,
        formatter=formatter,
    )


def parse_header_row(headers: Sequence[str], td: TypeDialect) -> Schema:
    cols: List[ColumnSpec] = []
    # header row is row=1 in error context
    for i, cell in enumerate(headers):
        cols.append(parse_header_cell(cell, td, row=1, col=i))
    # Enforce unique logical names (like csv.DictReader does)
    names = [c.name for c in cols]
    if len(set(names)) != len(names):
        # Find first duplicate for nicer error
        seen = set()
        for idx, n in enumerate(names):
            if n in seen:
                raise TypedCSVError(
                    row=1, col=idx, column=n, header=headers[idx], value="",
                    reason=f"Duplicate logical column name: {n!r}"
                )
            seen.add(n)
    return Schema(columns=cols)


# ----------------------------
# Type codecs (parse/format)
# ----------------------------

def _parse_bool(raw: str, td: TypeDialect) -> bool:
    s = raw.strip().lower()
    if s in td.bool_true:
        return True
    if s in td.bool_false:
        return False
    raise ValueError(f"Invalid bool literal: {raw!r}")


def _format_bool(v: Any) -> str:
    return "true" if bool(v) else "false"


def _get_type_codec(type_name: TypeName, td: TypeDialect) -> Tuple[Callable[[str], Any], Callable[[Any], str]]:
    if type_name == "str":
        return (lambda s: s), (lambda v: "" if v is None else str(v))
    if type_name == "int":
        return int, (lambda v: "" if v is None else str(int(v)))
    if type_name == "float":
        return float, (lambda v: "" if v is None else repr(float(v)))
    if type_name == "bool":
        return (lambda s: _parse_bool(s, td)), (lambda v: "" if v is None else _format_bool(v))
    if type_name == "datetime":
        return td.datetime_parser, (lambda v: "" if v is None else td.datetime_formatter(v))
    raise AssertionError(f"Unsupported type: {type_name!r}")


# ----------------------------
# Validators
# ----------------------------

_ALLOWED_VALIDATORS: Dict[TypeName, Tuple[str, ...]] = {
    "int": ("min", "max", "in"),
    "float": ("min", "max", "in"),
    "str": ("minlen", "maxlen", "in", "re"),
    "datetime": ("min", "max"),
    "bool": tuple(),
}

def _validate_validator_keys(
    type_name: TypeName,
    validators: Mapping[str, str],
    td: TypeDialect,
    *,
    row: int,
    col: int,
    column: str,
    header: str,
) -> None:
    allowed = set(_ALLOWED_VALIDATORS.get(type_name, tuple()))
    for k in validators:
        if k not in allowed:
            raise TypedCSVError(
                row=row, col=col, column=column, header=header, value="",
                reason=f"Validator {k!r} not allowed for type {type_name!r}"
            )


def _parse_in_set(type_name: TypeName, raw: str, td: TypeDialect, spec: ColumnSpec) -> set:
    parts = raw.split(td.in_sep) if raw != "" else []
    if type_name == "str":
        return set(parts)
    # numeric
    if type_name == "int":
        return {int(p) for p in parts}
    if type_name == "float":
        return {float(p) for p in parts}
    raise AssertionError("in= not supported for this type in v0")


def _validate_value(spec: ColumnSpec, value: Any, *, row: int, col: int) -> None:
    # Missing values skip validation (nullable by default)
    if value is None:
        return

    v = spec.validators
    t = spec.type_name

    try:
        if t in ("int", "float"):
            if "min" in v and value < spec.parser(v["min"]):
                raise ValueError(f"value {value!r} < min {v['min']!r}")
            if "max" in v and value > spec.parser(v["max"]):
                raise ValueError(f"value {value!r} > max {v['max']!r}")
            if "in" in v:
                allowed = _parse_in_set(t, v["in"], DEFAULT_TYPE_DIALECT, spec)
                if value not in allowed:
                    raise ValueError(f"value {value!r} not in {sorted(allowed)!r}")

        elif t == "str":
            s = value
            if not isinstance(s, str):
                s = str(s)
            if "minlen" in v and len(s) < int(v["minlen"]):
                raise ValueError(f"len {len(s)} < minlen {v['minlen']!r}")
            if "maxlen" in v and len(s) > int(v["maxlen"]):
                raise ValueError(f"len {len(s)} > maxlen {v['maxlen']!r}")
            if "in" in v:
                allowed = set(v["in"].split(DEFAULT_TYPE_DIALECT.in_sep)) if v["in"] != "" else set()
                if s not in allowed:
                    raise ValueError(f"value {s!r} not in {sorted(allowed)!r}")
            if "re" in v:
                pattern = v["re"]
                if re.fullmatch(pattern, s) is None:
                    raise ValueError(f"value {s!r} does not fullmatch /{pattern}/")

        elif t == "datetime":
            if "min" in v:
                mn = DEFAULT_TYPE_DIALECT.datetime_parser(v["min"])
                if value < mn:
                    raise ValueError(f"value {value!r} < min {v['min']!r}")
            if "max" in v:
                mx = DEFAULT_TYPE_DIALECT.datetime_parser(v["max"])
                if value > mx:
                    raise ValueError(f"value {value!r} > max {v['max']!r}")

        elif t == "bool":
            # no validators in v0
            return

    except TypedCSVError:
        raise
    except Exception as e:
        raise TypedCSVError(
            row=row, col=col, column=spec.name, header=spec.raw_header, value=str(value),
            reason=f"Validation failed: {e}"
        ) from e


# ----------------------------
# Cell parsing (incl. missing handling)
# ----------------------------

def _parse_cell(spec: ColumnSpec, raw: str, *, row: int, col: int) -> Any:
    # Missing handling
    if raw == "":
        if spec.type_name == "str":
            return ""
        return None

    try:
        value = spec.parser(raw)
    except Exception as e:
        raise TypedCSVError(
            row=row, col=col, column=spec.name, header=spec.raw_header, value=raw,
            reason=f"Parse failed for type {spec.type_name!r}: {e}"
        ) from e

    _validate_value(spec, value, row=row, col=col)
    return value


# ----------------------------
# Optional inference (v0: conservative, no datetime inference by default)
# ----------------------------

def infer_type(values: Iterable[str], td: TypeDialect = DEFAULT_TYPE_DIALECT) -> TypeName:
    """
    Infer a type for an *untyped* column from sample values.
    v0: conservative: int -> float -> bool (no datetime inference by default).
    Empty strings are ignored.
    """
    samples = [v for v in values if v != ""]
    if not samples:
        return "str"

    def can_parse_all(parse_fn: Callable[[str], Any]) -> bool:
        try:
            for s in samples:
                parse_fn(s)
            return True
        except Exception:
            return False

    if can_parse_all(int):
        return "int"
    if can_parse_all(float):
        return "float"
    if can_parse_all(lambda s: _parse_bool(s, td)):
        return "bool"
    return "str"


# ----------------------------
# Readers
# ----------------------------

class TypedReader:
    """
    Typed equivalent of csv.reader that consumes the header row to build schema.
    Iteration yields typed list rows.
    """
    def __init__(
        self,
        f,
        dialect: Union[str, csv.Dialect] = "excel",
        *,
        type_dialect: TypeDialect = DEFAULT_TYPE_DIALECT,
        infer_types: bool = False,
        infer_rows: int = 50,
        **fmtparams,
    ) -> None:
        self._csv = csv.reader(f, dialect=dialect, **fmtparams)
        self._td = type_dialect

        try:
            raw_headers = next(self._csv)
        except StopIteration:
            self.schema = Schema(columns=[])
            return

        schema = parse_header_row(raw_headers, type_dialect)

        # Optional inference for columns that are default str and have no explicit str marker.
        # In v0 we infer only for columns *without* any typing marker.
        if infer_types and schema.columns:
            # To detect "untyped": original header cell must have no sigil and no ":type" token.
            # We'll re-check using a simple predicate.
            cols = list(schema.columns)
            # Peek up to infer_rows rows to sample strings per column.
            peek: List[List[str]] = []
            for _ in range(infer_rows):
                try:
                    r = next(self._csv)
                except StopIteration:
                    break
                peek.append(r)

            # Build per-col samples
            for j, spec in enumerate(cols):
                raw = spec.raw_header.strip()
                # untyped if:
                # - no validators stripping doesn't matter here; raw_header includes validators; use prefix split
                prefix, _ = _split_validators(raw, type_dialect)
                p = prefix.strip()
                has_explicit = (type_dialect.explicit_sep in p and p.rsplit(type_dialect.explicit_sep, 1)[1] in _EXPLICIT_TYPES)
                has_sigil = (len(p) > 0 and p[-1] in type_dialect.sigils)
                if has_explicit or has_sigil:
                    continue  # keep declared type
                # Only infer if currently str
                if spec.type_name != "str":
                    continue
                # Sample values at col j
                col_vals = [row[j] if j < len(row) else "" for row in peek]
                inferred = infer_type(col_vals, type_dialect)
                if inferred != "str":
                    parser, formatter = _get_type_codec(inferred, type_dialect)
                    cols[j] = ColumnSpec(
                        name=spec.name,
                        type_name=inferred,
                        raw_header=spec.raw_header,   # header stays untyped; that's fine—typing is for runtime
                        validators=spec.validators,   # note: validators for str might not make sense after inference; v0 keeps them
                        parser=parser,
                        formatter=formatter,
                    )

            self.schema = Schema(columns=cols)

            # Re-yield peeked rows during iteration
            self._buffer = peek
        else:
            self.schema = schema
            self._buffer = []

        self._row_index = 1  # header already consumed; next data row is 2

    def __iter__(self) -> Iterator[List[Any]]:
        # Yield buffered peek rows first (if any)
        for r in getattr(self, "_buffer", []):
            self._row_index += 1
            yield self._parse_row(r)
        # Then stream remaining
        for r in self._csv:
            self._row_index += 1
            yield self._parse_row(r)

    def _parse_row(self, row: List[str]) -> List[Any]:
        out: List[Any] = []
        for j, spec in enumerate(self.schema.columns):
            raw = row[j] if j < len(row) else ""
            out.append(_parse_cell(spec, raw, row=self._row_index, col=j))
        return out


class TypedDictReader:
    """
    Typed equivalent of csv.DictReader that consumes header and yields typed dicts.
    """
    def __init__(
        self,
        f,
        dialect: Union[str, csv.Dialect] = "excel",
        *,
        type_dialect: TypeDialect = DEFAULT_TYPE_DIALECT,
        infer_types: bool = False,
        infer_rows: int = 50,
        **fmtparams,
    ) -> None:
        self._tr = TypedReader(
            f,
            dialect=dialect,
            type_dialect=type_dialect,
            infer_types=infer_types,
            infer_rows=infer_rows,
            **fmtparams,
        )
        self.fieldnames = self._tr.schema.names

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self._tr:
            yield dict(zip(self.fieldnames, row))


def reader(
    f,
    dialect: Union[str, csv.Dialect] = "excel",
    *,
    type_dialect: TypeDialect = DEFAULT_TYPE_DIALECT,
    infer_types: bool = False,
    infer_rows: int = 50,
    **fmtparams,
) -> TypedReader:
    return TypedReader(
        f,
        dialect=dialect,
        type_dialect=type_dialect,
        infer_types=infer_types,
        infer_rows=infer_rows,
        **fmtparams,
    )


def DictReader(
    f,
    dialect: Union[str, csv.Dialect] = "excel",
    *,
    type_dialect: TypeDialect = DEFAULT_TYPE_DIALECT,
    infer_types: bool = False,
    infer_rows: int = 50,
    **fmtparams,
) -> TypedDictReader:
    return TypedDictReader(
        f,
        dialect=dialect,
        type_dialect=type_dialect,
        infer_types=infer_types,
        infer_rows=infer_rows,
        **fmtparams,
    )


# ----------------------------
# Writers
# ----------------------------

class TypedWriter:
    """
    Minimal wrapper around csv.writer. No schema enforcement by default.
    Provided mainly for API symmetry. Use DictWriter for typed formatting.
    """
    def __init__(self, f, dialect: Union[str, csv.Dialect] = "excel", **fmtparams) -> None:
        self._csv = csv.writer(f, dialect=dialect, **fmtparams)

    def writerow(self, row: Sequence[Any]) -> int:
        return self._csv.writerow(["" if v is None else str(v) for v in row])

    def writerows(self, rows: Iterable[Sequence[Any]]) -> None:
        for r in rows:
            self.writerow(r)


class TypedDictWriter:
    """
    Typed-aware dict writer. `fieldnames` are header cells (may include typing/validators).
    - Preserves header exactly as provided.
    - Formats values canonically based on parsed types.
    """
    def __init__(
        self,
        f,
        fieldnames: Sequence[str],
        dialect: Union[str, csv.Dialect] = "excel",
        *,
        type_dialect: TypeDialect = DEFAULT_TYPE_DIALECT,
        **fmtparams,
    ) -> None:
        self._csv = csv.writer(f, dialect=dialect, **fmtparams)
        self._td = type_dialect
        self.raw_fieldnames = list(fieldnames)

        # Build schema from provided header cells
        self.schema = parse_header_row(self.raw_fieldnames, type_dialect)
        self.fieldnames = self.schema.names  # logical keys

    def writeheader(self) -> int:
        return self._csv.writerow(self.raw_fieldnames)

    def writerow(self, rowdict: Mapping[str, Any]) -> int:
        out: List[str] = []
        for spec in self.schema.columns:
            v = rowdict.get(spec.name, None)
            # Canonical formatting:
            if v is None:
                out.append("")
            else:
                out.append(spec.formatter(v))
        return self._csv.writerow(out)

    def writerows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self.writerow(r)


def writer(f, dialect: Union[str, csv.Dialect] = "excel", **fmtparams) -> TypedWriter:
    return TypedWriter(f, dialect=dialect, **fmtparams)


def DictWriter(
    f,
    fieldnames: Sequence[str],
    dialect: Union[str, csv.Dialect] = "excel",
    *,
    type_dialect: TypeDialect = DEFAULT_TYPE_DIALECT,
    **fmtparams,
) -> TypedDictWriter:
    return TypedDictWriter(f, fieldnames, dialect=dialect, type_dialect=type_dialect, **fmtparams)


# ----------------------------
# Quick examples (doctest-like)
# ----------------------------
if __name__ == "__main__":
    import io

    data = """id#,name$,active?,created@ [min=2020-01-01T00:00:00]
1,Alice,true,2021-05-01T12:30:00
2,Bob,false,
"""
    f = io.StringIO(data)
    r = DictReader(f)
    for row in r:
        print(row)
    # {'id': 1, 'name': 'Alice', 'active': True, 'created': datetime(...)}
    # {'id': 2, 'name': 'Bob', 'active': False, 'created': None}
