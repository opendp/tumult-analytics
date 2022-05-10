# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased
### Added
- Added `QueryBuilder.bin_column` and an associated `BinningSpec` type.
- Dates may now be used in `KeySet`s.

### Fixed
- `KeySet`s now explicitly check for and disallow the use of floats and timestamps as keys.
  This has always been the intended behavior, but it was previously not checked for and could work or cause non-obvious errors depending on the situation.
- `KeySet.dataframe()` now always returns a dataframe where all rows are distinct.
- Under certain circumstances, evaluating a `GroupByCountDistinct` query expression used to modify the input `QueryExpr`.
  This no longer occurs.
- It is now possible to partition on a column created by a grouping flat map (used to raise exception from Core)

## 0.2.1 - 2022-04-14
### Added
- Added support for basic operations (filter, map, etc.) on Spark date and timestamp columns. `ColumnType` has two new variants, `DATE` and `TIMESTAMP`, to support these.
- Future documentation will include any exceptions defined in Analytics.

### Changed
- Switch session to use Persist/Unpersist instead of Cache.

## 0.2.0 - 2022-03-28
### Removed
- Multi-query evaluate support is entirely removed.
- Columns that are neither floats nor doubles will no longer be checked for NaN values.
- The `BIT` variant of the `ColumnType` enum was removed, as it was not supported elsewhere in Analytics.

### Changed
- *Backwards-incompatible*: Renamed `query_exprs` parameter in `Session.evaluate` to `query_expr`.
- *Backwards-incompatible*: `QueryBuilder.join_public` and the `JoinPublic` query expression can now accept public tables specified as Spark dataframes. The existing behavior using public source IDs is still supported, but the `public_id` parameter/property is now called `public_table`.
- Installation on Python 3.7.1 through 3.7.3 is now allowed.
- KeySets now do type coercion on creation, matching the type coercion that Sessions do for private sources.
- Sessions created by `partition_and_create` must be used in the order they were created, and using the parent session will forcibly close all child sessions. Sessions can be manually closed with `session.stop()`.


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
