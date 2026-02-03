# Changelog

## 0.2.0 - 2026-02-03

- Add required (`!`) and unique (`+`) constraint sigils for suffix-typed headers.
- Add flag-only validators `required` and `unique` for `:type` headers.
- Enforce required/unique constraints during parsing; unique ignores missing values.
- Document `.tcsv` and `.ttsv` default filename conventions.

## 0.1.0 - 2026-02-03

- Initial release.
- Header-embedded typing and validators.
- Typed readers and writers compatible with `csv` API.
- Pytest tests, ruff, and mypy configs.
