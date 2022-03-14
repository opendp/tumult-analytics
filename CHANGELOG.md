# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Removed
- Multi-query evaluate support is entirely removed.
- Columns that are neither floats nor doubles will no longer be checked for NaN values.

### Changed
- *Backwards-incompatible*: Renamed `query_exprs` parameter in `Session.evaluate` to `query_expr`.
- *Backwards-incompatible*: `QueryBuilder.join_public` and the `JoinPublic` query expression can now accept public tables specified as Spark dataframes. The existing behavior using public source IDs is still supported, but the `public_id` parameter/property is now called `public_table`.
- Installation on Python 3.7.1 through 3.7.3 is now allowed.
- KeySets now do type coercion on creation, matching the type coercion that Sessions do for private sources.


### Fixed
- Joining with a public table that contains no NaNs, but has a column where NaNs are allowed, previously caused an error when compiling queries. This is now handled correctly.

## 0.1.1 - 2022-02-28
### Added
- Added a `KeySet` class, which will eventually be used for all GroupBy queries.
- Added `QueryBuilder.groupby()`, a new group-by based on `KeySet`s.

### Changed
- The Analytics library now uses `KeySet` and `QueryBuilder.groupby()` for all
  GroupBy queries.
- The various `Session` methods for loading in data from CSV no longer support loading the data's schema from a file.
- Made Session return a more user-friendly error message when the user provides  a privacy budget of 0.
- Removed all instances of the old name of this library, and replaced them with "Analytics"

### Deprecated
- `QueryBuilder.groupby_domains()` and `QueryBuilder.groupby_public_source()` are now deprecated in favor of using `QueryBuilder.groupby()` with `KeySet`s.
  They will be removed in a future version.

## 0.1.0 - 2022-02-15
### Added
- Initial release
