# typedcsv

[![PyPI version](https://img.shields.io/pypi/v/typedcsv-lib.svg)](https://pypi.org/project/typedcsv-lib/)
[![Python versions](https://img.shields.io/pypi/pyversions/typedcsv-lib.svg)](https://pypi.org/project/typedcsv-lib/)
[![License](https://img.shields.io/pypi/l/typedcsv-lib.svg)](https://pypi.org/project/typedcsv-lib/)

Typed CSVs via **header-embedded types** (sigils or `:type`) plus optional **header-embedded validation** — **stdlib-only**, Python **3.10+**.

The core lives in a single module and is copy-pasteable into projects.

```bash
pip install typedcsv-lib
```

---

## Header typing

Declare types using either **suffix sigils** or **explicit `:type`** (not both on the same column).

| Type       | Sigil (suffix) | Explicit     |
|------------|-----------------|--------------|
| `int`      | `#`             | `:int`       |
| `float`    | `%`             | `:float`     |
| `bool`     | `?`             | `:bool`      |
| `datetime` | `@`             | `:datetime`  |
| `str`      | `$`             | `:str`       |

Untyped columns default to `str`.

**Logical column names** are the header names with the type marker removed:

- `age#` becomes key `"age"`
- `created:datetime` becomes key `"created"`

---

## Validators

Add an optional validator clause after the type marker:

```text
age# [min=0 max=120]
ratio% [min=0 max=1]
status$ [in=OPEN|CLOSED|PENDING]
code$ [re=^[A-Z]{3}\d{2}$]
created@ [min=2020-01-01T00:00:00 max=2030-12-31T23:59:59]
```

Notes:

- Validators are space-separated `key=value` pairs inside `[ ... ]`.
- `re=` uses Python `re.fullmatch`.
- `in=` uses `|` as separator.
- Unknown validator keys raise an error.

---

## Missing values (nullable by default)

- An empty cell (`""`) is missing.
- For `str` columns, missing stays `""`.
- For non-`str` columns, missing becomes `None`.
- Missing values skip validation.

---

## Reading

```python
import typedcsv

with open("data.csv", newline="") as f:
    for row in typedcsv.DictReader(f):
        print(row)
```

Example CSV:

```csv
id#,name$,active?,created@
1,Alice,true,2021-05-01T12:30:00
2,Bob,false,
```

Produces:

```python
{'id': 1, 'name': 'Alice', 'active': True, 'created': datetime(2021, 5, 1, 12, 30)}
{'id': 2, 'name': 'Bob', 'active': False, 'created': None}
```

---

## Writing (canonical formatting)

- `None` → empty cell
- `bool` → `true` / `false`
- `datetime` → `isoformat()`
- `float` → `repr(f)`
- Header preserved exactly as provided to `DictWriter(fieldnames=...)`

```python
import typedcsv
from datetime import datetime

rows = [
    {"id": 1, "name": "Alice", "active": True, "created": datetime(2021, 5, 1, 12, 30)},
    {"id": 2, "name": "Bob", "active": False, "created": None},
]

with open("out.csv", "w", newline="") as f:
    w = typedcsv.DictWriter(f, fieldnames=["id#", "name$", "active?", "created@"])
    w.writeheader()
    w.writerows(rows)
```

---

## More examples

### Validators with quoted values

```csv
name$ [in="Alice Smith"|Bob]
```

### Regex validation (fullmatch)

```csv
code$ [re=^[A-Z]{3}\d{2}$]
ABC12
```

### Type inference for untyped columns

```python
import typedcsv
import io

data = "a,b\n1,true\n2,false\n"
rows = list(typedcsv.DictReader(io.StringIO(data), infer_types=True))
```

---

## Errors

Parsing/validation failures raise `TypedCSVError` with context:

- `row` (1-based; header row is 1)
- `col` (0-based)
- `column` (logical name)
- `header` (raw header cell)
- `value` (raw cell)
- `reason` (short message)

---

## API reference (csv-compatible)

typedcsv mirrors Python's `csv` module API and is designed to be a drop-in replacement where you want typed rows.

- `typedcsv.reader(f, ...)` → yields typed list rows (header consumed)
- `typedcsv.DictReader(f, ...)` → yields typed dict rows keyed by logical names (header consumed)
- `typedcsv.writer(f, ...)` → returns a standard `csv.writer`
- `typedcsv.DictWriter(f, fieldnames, ...)` → writes typed dict rows with canonical formatting

---
