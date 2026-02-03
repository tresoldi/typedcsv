"""
typedcsv — “Typed CSVs” with header-embedded types + validators (stdlib-only).

Contract (v0):
- Typing + validators live ONLY in the header row.
- Suffix sigils OR explicit ":type" (not both).
- Suffix-only sigils:
    # -> int, % -> float, ? -> bool, @ -> datetime, $ -> str
  Constraint sigils (suffix, after type sigil): ! -> required, + -> unique
  Explicit types: :int, :float, :bool, :datetime, :str
- Untyped defaults to str.
- Validators: optional bracket clause after the type marker:
    col# [min=0 max=10]
    col:str [in=A|B|C]
    col$ [re=^[A-Z]{3}\\d{2}$]
    col:str [required unique]
  Flag-only validators: required, unique.
  Parsed as space-separated key=value pairs inside [ ... ].
  Values with spaces must be double-quoted.
  "re=" uses re.fullmatch.
- Missing values: empty string cell "" is missing:
    str columns -> "" (kept)
    non-str columns -> None
  Missing values skip validation.
  required rejects missing values; unique ignores missing values.
- Errors: raise immediately with TypedCSVError including row/col/context.
- Writing: None -> "", bool -> true/false, datetime -> isoformat(), float -> repr(f),
  header preserved exactly as provided to DictWriter(fieldnames).

API (csv-like, but typed readers consume header):
- reader(f, ...) -> iterator of typed list rows
- DictReader(f, ...) -> iterator of typed dict rows
- writer(f, ...) -> basic csv.writer wrapper for lists (no schema enforcement)
- DictWriter(f, fieldnames, ...) -> typed-aware dict writer; preserve typed fieldnames

Python: 3.10+
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


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
        msg = (
            "TypedCSVError(" +
            f"row={row}, col={col}, column={column!r}, "
            f"header={header!r}, value={value!r}): {reason}"
        )
        super().__init__(msg)
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


DEFAULT = TypeDialect()
DEFAULT_TYPE_DIALECT = DEFAULT

_EXPLICIT_TYPES: Tuple[TypeName, ...] = ("int", "float", "str", "bool", "datetime")
_MODIFIER_SIGILS = {"!", "+"}
_FLAG_VALIDATORS = {"required", "unique"}
__version__ = "0.2.0"


# ----------------------------
# Schema / column spec
# ----------------------------

@dataclass(frozen=True)
class ColumnSpec:
    name: str                  # logical name (sigil/type stripped)
    type_name: TypeName        # one of _EXPLICIT_TYPES
    raw_header: str            # raw header cell (including markers/validators)
    validators: Dict[str, str] # raw validator values (strings) from header
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

def _split_validators(cell: str, td: TypeDialect, *, row: int, col: int) -> Tuple[str, Optional[str]]:
    """
    Split "prefix [k=v ...]" into ("prefix", "[k=v ...]") or ("prefix", None).
    If '[' appears, require a trailing validators clause; otherwise raise.
    """
    s = cell.strip()
    if not s:
        return s, None

    idx = s.find(td.validators_open)
    if idx == -1:
        return s, None

    if not s.endswith(td.validators_close):
        raise TypedCSVError(
            row=row, col=col, column="", header=cell, value="",
            reason="Malformed validators clause (missing closing ']')"
        )

    prefix = s[:idx].rstrip()
    validators = s[idx:].strip()
    return prefix, validators


def parse_validators(
    text: str,
    td: TypeDialect,
    *,
    row: int,
    col: int,
    column: str,
    header: str,
) -> Dict[str, str]:
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

    i, n = 0, len(inner)
    out: Dict[str, str] = {}

    def skip_ws(j: int) -> int:
        while j < n and inner[j].isspace():
            j += 1
        return j

    i = skip_ws(i)
    while i < n:
        start = i
        while i < n and not inner[i].isspace() and inner[i] != "=":
            i += 1
        key = inner[start:i].strip()
        if not key:
            raise TypedCSVError(
                row=row, col=col, column=column, header=header, value="",
                reason=f"Invalid validator key near: {inner[start:]!r}"
            )

        value = ""
        if i < n and inner[i] == "=":
            i += 1
            if i >= n:
                raise TypedCSVError(
                    row=row, col=col, column=column, header=header, value="",
                    reason=f"Validator {key!r} missing value"
                )

            if inner[i] == '"':
                i += 1
                start = i
                while i < n and inner[i] != '"':
                    i += 1
                if i >= n:
                    raise TypedCSVError(
                        row=row, col=col, column=column, header=header, value="",
                        reason=f"Unterminated quote in validator {key!r}"
                    )
                value = inner[start:i]
                i += 1
            else:
                start = i
                while i < n and not inner[i].isspace():
                    i += 1
                value = inner[start:i]
        else:
            if key not in _FLAG_VALIDATORS:
                raise TypedCSVError(
                    row=row, col=col, column=column, header=header, value="",
                    reason=f"Invalid validator token (missing '=') near: {inner[start:]!r}"
                )

        if key in _FLAG_VALIDATORS and value != "":
            raise TypedCSVError(
                row=row, col=col, column=column, header=header, value="",
                reason=f"Validator {key!r} must be flag-only"
            )

        if key in out:
            raise TypedCSVError(
                row=row, col=col, column=column, header=header, value="",
                reason=f"Duplicate validator key: {key!r}"
            )
        out[key] = value

        i = skip_ws(i)

    return out


def parse_header_cell(cell: str, td: TypeDialect, *, row: int, col: int) -> ColumnSpec:
    """Parse one header cell into ColumnSpec. Raises TypedCSVError on invalid syntax."""
    raw_header = cell
    prefix, vtext = _split_validators(cell, td, row=row, col=col)

    base = prefix.strip()
    if not base:
        raise TypedCSVError(
            row=row, col=col, column="", header=raw_header, value="",
            reason="Empty header cell"
        )

    used_explicit = False
    used_sigil = False
    type_name: TypeName = "str"
    modifier_unique = False
    modifier_required = False

    if td.explicit_sep in base:
        name_part, type_part = base.rsplit(td.explicit_sep, 1)
        if any(c in _MODIFIER_SIGILS for c in type_part):
            stripped = type_part.rstrip("".join(_MODIFIER_SIGILS))
            if stripped in _EXPLICIT_TYPES and all(c in _MODIFIER_SIGILS for c in type_part[len(stripped):]):
                raise TypedCSVError(
                    row=row, col=col, column="", header=raw_header, value="",
                    reason="Header uses modifiers with explicit type (not allowed)"
                )
        if type_part not in _EXPLICIT_TYPES:
            raise TypedCSVError(
                row=row, col=col, column="", header=raw_header, value="",
                reason=f"Unknown explicit type: {type_part!r}"
            )
        used_explicit = True
        type_name = type_part
        base_name = name_part.strip()
    else:
        base_name = base

    if not used_explicit and base_name:
        allowed_tail = set(td.sigils) | _MODIFIER_SIGILS
        i = len(base_name)
        while i > 0 and base_name[i - 1] in allowed_tail:
            i -= 1
        tail = base_name[i:]
        if tail:
            type_sigils = [c for c in tail if c in td.sigils]
            if len(type_sigils) > 1:
                raise TypedCSVError(
                    row=row, col=col, column="", header=raw_header, value="",
                    reason="Header uses multiple type sigils (not allowed)"
                )
            if len(type_sigils) == 1:
                used_sigil = True
                type_name = td.sigils[type_sigils[0]]
                modifier_unique = "+" in tail
                modifier_required = "!" in tail
                base_name = base_name[:i].rstrip()
            else:
                raise TypedCSVError(
                    row=row, col=col, column="", header=raw_header, value="",
                    reason="Header uses modifiers without a type sigil"
                )

    if used_explicit and base_name and base_name[-1] in td.sigils:
        raise TypedCSVError(
            row=row, col=col, column="", header=raw_header, value="",
            reason="Header uses both explicit type and sigil (not allowed)"
        )

    if used_explicit and used_sigil:
        raise TypedCSVError(
            row=row, col=col, column="", header=raw_header, value="",
            reason="Header uses both explicit type and sigil (not allowed)"
        )
    if used_explicit and base_name and base_name[-1] in _MODIFIER_SIGILS:
        raise TypedCSVError(
            row=row, col=col, column="", header=raw_header, value="",
            reason="Header uses modifiers with explicit type (not allowed)"
        )

    name = base_name.strip()
    if not name:
        raise TypedCSVError(
            row=row, col=col, column="", header=raw_header, value="",
            reason="Header name is empty after stripping type marker"
        )

    validators: Dict[str, str] = {}
    if vtext is not None:
        validators = parse_validators(vtext, td, row=row, col=col, column=name, header=raw_header)
    if modifier_unique:
        if "unique" in validators:
            raise TypedCSVError(
                row=row, col=col, column=name, header=raw_header, value="",
                reason="Duplicate validator key: 'unique'"
            )
        validators["unique"] = ""
    if modifier_required:
        if "required" in validators:
            raise TypedCSVError(
                row=row, col=col, column=name, header=raw_header, value="",
                reason="Duplicate validator key: 'required'"
            )
        validators["required"] = ""

    parser, formatter = _get_type_codec(type_name, td)
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
    for i, cell in enumerate(headers):
        cols.append(parse_header_cell(cell, td, row=1, col=i))

    names = [c.name for c in cols]
    if len(set(names)) != len(names):
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
    "int": ("min", "max", "in", "required", "unique"),
    "float": ("min", "max", "in", "required", "unique"),
    "str": ("minlen", "maxlen", "in", "re", "required", "unique"),
    "datetime": ("min", "max", "required", "unique"),
    "bool": ("required", "unique"),
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


def _parse_in_set(type_name: TypeName, raw: str, td: TypeDialect) -> set:
    parts = raw.split(td.in_sep) if raw != "" else []
    if type_name == "str":
        return set(parts)
    if type_name == "int":
        return {int(p) for p in parts}
    if type_name == "float":
        return {float(p) for p in parts}
    raise AssertionError("in= not supported for this type in v0")


def _validate_value(spec: ColumnSpec, value: Any, *, row: int, col: int, td: TypeDialect) -> None:
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
                allowed = _parse_in_set(t, v["in"], td)
                if value not in allowed:
                    raise ValueError(f"value {value!r} not in {sorted(allowed)!r}")

        elif t == "str":
            s = value if isinstance(value, str) else str(value)
            if "minlen" in v and len(s) < int(v["minlen"]):
                raise ValueError(f"len {len(s)} < minlen {v['minlen']!r}")
            if "maxlen" in v and len(s) > int(v["maxlen"]):
                raise ValueError(f"len {len(s)} > maxlen {v['maxlen']!r}")
            if "in" in v:
                allowed = set(v["in"].split(td.in_sep)) if v["in"] != "" else set()
                if s not in allowed:
                    raise ValueError(f"value {s!r} not in {sorted(allowed)!r}")
            if "re" in v:
                pattern = v["re"]
                if re.fullmatch(pattern, s) is None:
                    raise ValueError(f"value {s!r} does not fullmatch /{pattern}/")

        elif t == "datetime":
            if "min" in v:
                mn = td.datetime_parser(v["min"])
                if value < mn:
                    raise ValueError(f"value {value!r} < min {v['min']!r}")
            if "max" in v:
                mx = td.datetime_parser(v["max"])
                if value > mx:
                    raise ValueError(f"value {value!r} > max {v['max']!r}")

        elif t == "bool":
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

def _parse_cell(spec: ColumnSpec, raw: str, *, row: int, col: int, td: TypeDialect) -> Any:
    if raw == "":
        if "required" in spec.validators:
            raise TypedCSVError(
                row=row, col=col, column=spec.name, header=spec.raw_header, value=raw,
                reason="Missing value not allowed"
            )
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

    _validate_value(spec, value, row=row, col=col, td=td)
    return value


# ----------------------------
# Optional inference (conservative)
# ----------------------------

def infer_type(values: Iterable[str], td: TypeDialect = DEFAULT) -> TypeName:
    """
    Infer a type for an *untyped* column from sample values.
    Conservative: int -> float -> bool, else str. Empty strings ignored.
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
    """Typed equivalent of csv.reader that consumes the header row to build schema."""

    def __init__(
        self,
        f: Any,
        dialect: Union[str, csv.Dialect] = "excel",
        *,
        type_dialect: TypeDialect = DEFAULT,
        infer_types: bool = False,
        infer_rows: int = 50,
        **fmtparams: Any,
    ) -> None:
        self._csv = csv.reader(f, dialect=dialect, **fmtparams)
        self._td = type_dialect

        try:
            raw_headers = next(self._csv)
        except StopIteration:
            self.schema = Schema(columns=[])
            self._buffer = []
            self._row_index = 1
            return

        schema = parse_header_row(raw_headers, type_dialect)

        if infer_types and schema.columns:
            cols = list(schema.columns)
            peek: List[List[str]] = []
            for _ in range(infer_rows):
                try:
                    r = next(self._csv)
                except StopIteration:
                    break
                peek.append(r)

            for j, spec in enumerate(cols):
                raw = spec.raw_header.strip()
                prefix, _ = _split_validators(raw, type_dialect, row=1, col=j)
                p = prefix.strip()
                has_explicit = (type_dialect.explicit_sep in p)
                has_sigil = (len(p) > 0 and p[-1] in type_dialect.sigils)
                has_validators = _ is not None
                if has_explicit or has_sigil or has_validators:
                    continue
                if spec.type_name != "str":
                    continue
                col_vals = [row[j] if j < len(row) else "" for row in peek]
                inferred = infer_type(col_vals, type_dialect)
                if inferred != "str":
                    parser, formatter = _get_type_codec(inferred, type_dialect)
                    cols[j] = ColumnSpec(
                        name=spec.name,
                        type_name=inferred,
                        raw_header=spec.raw_header,
                        validators=spec.validators,
                        parser=parser,
                        formatter=formatter,
                    )

            self.schema = Schema(columns=cols)
            self._buffer = peek
        else:
            self.schema = schema
            self._buffer = []

        self._row_index = 1
        self._unique_cols = {
            idx for idx, spec in enumerate(self.schema.columns)
            if "unique" in spec.validators
        }
        self._unique_sets: Dict[int, set] = {idx: set() for idx in self._unique_cols}

    def __iter__(self) -> "TypedReader":
        return self

    def __next__(self) -> List[Any]:
        if not hasattr(self, "_iter"):
            self._iter = self._iter_rows()
        return next(self._iter)

    def _iter_rows(self) -> Iterator[List[Any]]:
        for r in self._buffer:
            self._row_index += 1
            yield self._parse_row(r)
        for r in self._csv:
            self._row_index += 1
            yield self._parse_row(r)

    def _parse_row(self, row: List[str]) -> List[Any]:
        out: List[Any] = []
        for j, spec in enumerate(self.schema.columns):
            raw = row[j] if j < len(row) else ""
            value = _parse_cell(spec, raw, row=self._row_index, col=j, td=self._td)
            if j in self._unique_cols and raw != "":
                seen = self._unique_sets[j]
                if value in seen:
                    raise TypedCSVError(
                        row=self._row_index, col=j, column=spec.name, header=spec.raw_header, value=raw,
                        reason="Duplicate value in unique column"
                    )
                seen.add(value)
            out.append(value)
        return out


class TypedDictReader:
    """Typed equivalent of csv.DictReader that consumes header and yields typed dicts."""

    def __init__(
        self,
        f: Any,
        dialect: Union[str, csv.Dialect] = "excel",
        *,
        type_dialect: TypeDialect = DEFAULT,
        infer_types: bool = False,
        infer_rows: int = 50,
        **fmtparams: Any,
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
    f: Any,
    dialect: Union[str, csv.Dialect] = "excel",
    *,
    type_dialect: TypeDialect = DEFAULT,
    infer_types: bool = False,
    infer_rows: int = 50,
    **fmtparams: Any,
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
    f: Any,
    dialect: Union[str, csv.Dialect] = "excel",
    *,
    type_dialect: TypeDialect = DEFAULT,
    infer_types: bool = False,
    infer_rows: int = 50,
    **fmtparams: Any,
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

class TypedDictWriter:
    """
    Typed-aware dict writer. `fieldnames` are header cells (may include typing/validators).
    - Preserves header exactly as provided.
    - Formats values canonically based on parsed types.
    """

    def __init__(
        self,
        f: Any,
        fieldnames: Sequence[str],
        dialect: Union[str, csv.Dialect] = "excel",
        *,
        type_dialect: TypeDialect = DEFAULT,
        **fmtparams: Any,
    ) -> None:
        self._csv = csv.writer(f, dialect=dialect, **fmtparams)
        self._td = type_dialect
        self.raw_fieldnames = list(fieldnames)

        self.schema = parse_header_row(self.raw_fieldnames, type_dialect)
        self.fieldnames = self.schema.names

    def writeheader(self) -> int:
        return int(self._csv.writerow(self.raw_fieldnames))

    def writerow(self, rowdict: Mapping[str, Any]) -> int:
        out: List[str] = []
        for spec in self.schema.columns:
            v = rowdict.get(spec.name, None)
            if v is None:
                out.append("")
            else:
                out.append(spec.formatter(v))
        return int(self._csv.writerow(out))

    def writerows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self.writerow(r)


def writer(
    f: Any,
    dialect: Union[str, csv.Dialect] = "excel",
    **fmtparams: Any,
) -> Any:
    return csv.writer(f, dialect=dialect, **fmtparams)


def DictWriter(
    f: Any,
    fieldnames: Sequence[str],
    dialect: Union[str, csv.Dialect] = "excel",
    *,
    type_dialect: TypeDialect = DEFAULT,
    **fmtparams: Any,
) -> TypedDictWriter:
    return TypedDictWriter(f, fieldnames, dialect=dialect, type_dialect=type_dialect, **fmtparams)


__all__ = [
    "TypedCSVError",
    "TypeDialect",
    "DEFAULT",
    "DEFAULT_TYPE_DIALECT",
    "__version__",
    "DictReader",
    "DictWriter",
    "reader",
    "writer",
    "infer_type",
]
