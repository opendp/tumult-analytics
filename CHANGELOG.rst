..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2025

.. _changelog:

Changelog
=========

🚨 Important Update: the Tumult Labs Team is Joining LinkedIn 🚨
----------------------------------------------------------------

The `Tumult Labs team has joined LinkedIn <https://www.linkedin.com/pulse/whats-next-us-tumult-labs-gerome-miklau-zmpye>`__! 🎉 As part of this transition, we are exploring options for the future of Tumult Analytics, including finding a new home for the project. 🏡
We greatly appreciate the community’s support and contributions. If your organization is interested in maintaining or adopting Tumult Analytics, please reach out! 📩
For now, the repository remains available, and we encourage users to continue engaging with the project. We’ll provide updates as soon as we have more to share.

— The Tumult Labs Team 💙

.. _v0.20.1:

0.20.1 - 2025-04-02
-------------------

Added
~~~~~
- Add LinkedIn announcement to CHANGELOG.rst.

Changed
~~~~~~~
- Updated tutorials and guides to pull example data from gitlab rather than s3.

.. _v0.20.0:

0.20.0 - 2025-02-25
-------------------

This release introduces a major overhaul of KeySets, including several new operations and tweaks to existing ones. 
It also features reorganized documentation. 
For Pro users, we've used the new KeySet capabilities to slightly simplify the Synthetics API and introduced some behind-the-scenes performance and accuracy improvements.

Added
~~~~~

- The :class:`.KeySet` class now supports additional mechanisms for combining KeySets: :meth:`.KeySet.join` and :meth:`.KeySet.__sub__`.

Changed
~~~~~~~

- The documentation is now split in three distinct product-level documentation pages, for :ref:`Tumult Analytics<analytics>`, :ref:`Tumult Synthetics<synthetics>`, and :ref:`Tumult Tune<tune>`.
- The API reference is now organized into semantically-grouped subpages and no longer follows the internal module structure.
- All Python objects should now be imported directly from ``tmlt.analytics``, ``tmlt.synthetics``, and ``tmlt.tune``. The documentation and demos have been updated accordingly. Existing import paths still work but will be removed in a future release, so users should switch to the new import statements.
- :meth:`.KeySet.__eq__` now ignores column order when comparing KeySets.
- :meth:`.KeySet.__getitem__` can no longer be used to drop all of the columns in KeySet, for example with ``keyset[[]]``.
- :meth:`.KeySet.filter` no longer allows using the KeySet's dataframe to be used when specifying columns (e.g. writing expressions like ``ks.filter(ks.dataframe().A > 5)``).
  Use ``pyspark.sql.functions.col`` to identify the columns instead.
  Passing filter conditions as strings is unaffected.
- :meth:`.KeySet.schema` reports column nullability more accurately than it did previously, for example identifying that passing a column with no ``None`` values to :meth:`.KeySet.from_dict` will cause that column to not contain nulls.
- Numpy arrays can now be used when initializing a :class:`~tmlt.analytics.binning_spec.BinningSpec`.
- |PRO| Renamed ``CountReleasedRows`` to :class:`~tmlt.tune.CountDPRows` and ``HighRelativeErrorFraction`` to :class:`~tmlt.tune.HighRelativeErrorRate`.
- |PRO| *Backwards-incompatible*: :func:`~.synthetics.generate_synthetic_data` no longer accepts a separate ``count_structural_zeroes`` argument.
  Instead, build these structural zeroes into the ``keyset`` argument using :meth:`.KeySet.__sub__`.
- |PRO| Significant behind-the-scenes refactor of the synthetic data generation code.
  - This addresses an OOM crash when generating synthetic data where the number of rows is large relative to the model size.
  - The way numeric columns are generated based on sums has been changed to be significantly more accurate and reliable.

Fixed
~~~~~
- |PRO| Fixed a crash when using :class:`.AutomaticBounds` with :func:`~.synthetics.generate_synthetic_data`.
- |PRO| Fixed a crash when using dates or timestamps when there are empty groups for ``split_columns`` in the generated data.
- |PRO| Fixed a bug in tuners run with :meth:`.SessionProgramTuner.Builder.with_cache`; the option should now give a much more substantial performance improvement.

.. _v0.19.0:

0.19.0 - 2024-11-21
-------------------

This release includes a number of improvements and bug fixes to the Tumult Synthetics API.

Changed
~~~~~~~
- |PRO| :meth:`.ErrorReport.dataframes` and :meth:`.MultiErrorReport.dataframes` now return an empty dict if there are no metrics.
- |PRO| Pandas DataFrames produced by :meth:`.ErrorReport.dataframe`, :meth:`.ErrorReport.dataframes`, and equivalent :class:`~tmlt.tune.MultiErrorReport` methods now have column types that allow null values.
- |PRO| The ``BinningStrategy`` class has been removed.
  Instead, use :class:`.BinningSpec` directly for binning in :func:`~.synthetics.generate_synthetic_data`.
  The behavior is the same as if you had used ``BinningStrategy`` with ``generate_nulls=True``.

Fixed
~~~~~
- |PRO| Using :class:`.AutomaticBounds` when generating synthetic data no longer produces an error.

.. _v0.18.0:

0.18.0 - 2024-11-19
-------------------

This release adds an initial version of Tumult Synthetics, Tumult Labs' new differentially private synthetic data generator.
The :ref:`API for this generator<synthetics-api>` is still under development and may undergo significant changes.

Additionally, this release increases the minimum supported Python version to 3.9, and the minimum supported PySpark version to 3.3.1.

Added
~~~~~
- |PRO| Added :meth:`.ErrorReport.dataframe`, which combines the metric results in this error in a single DataFrame, if possible.
- |PRO| Added :class:`~tmlt.tune.NamedValue`, which allows users to pass a name along with a parameter value when using a :class:`.SessionProgramTuner`. The name is then used when printing error reports or converting them to DataFrames.

Changed
~~~~~~~
- |PRO| *Backwards-incompatible*: The ``with_cache_enabled`` method on Tuner builders has been renamed to :meth:`.SessionProgramTuner.Builder.with_cache`.
- |PRO| :meth:`.ErrorReport.dataframes` and :meth:`.MultiErrorReport.dataframes` now return an empty dict if there are no metrics.
- |PRO| Pandas DataFrames produced by :meth:`.ErrorReport.dataframe`, :meth:`.ErrorReport.dataframes`, and equivalent :class:`~tmlt.tune.MultiErrorReport` methods now have column types that allow null values.

Fixed
~~~~~
- The ``columns`` argument to :meth:`.KeySet.from_tuples` is no longer required to be a tuple, any sequence type (e.g. a list) is now acceptable.
- |PRO| Fixes a bug where some metric values would be matched to incorrect parameters when calling :meth:`.MultiErrorReport.dataframe`.

.. _v0.17.0:

0.17.0 - 2024-11-04
-------------------

This release provides a number of quality of life improvements, including a new :meth:`.KeySet.from_tuples` method and support for basic arithmetic on privacy budgets.

For Pro users, it also introduces query caching in :class:`.SessionProgramTuner`\ s, which can be enabled by using ``with_cache_enabled`` when initializing the tuner.
This stores query results so that they can be reused in subsequent runs that evaluate the same queries.
It also includes some significant changes to metrics and error reports, some new metrics, and minor changes to :class:`.SessionProgram`.

.. note::

   Tumult Analytics 0.17 will be the last minor version to support Python 3.8 and PySpark versions below 3.3.1.
   If you are using Python 3.8 or one of these versions of PySpark, you will need to upgrade them in order to use Tumult Analytics 0.18.0.

Changed
~~~~~~~
- The :meth:`~tmlt.analytics.QueryBuilder.map`, :meth:`~tmlt.analytics.QueryBuilder.flat_map`, and :meth:`~tmlt.analytics.QueryBuilder.flat_map_by_id` transformations now more strictly check their outputs against the provided new column types.
  This may cause some existing programs to produce errors if they relied on the previous, less-strict behavior.
- |PRO| Users are now allowed to define abstract subclasses of :class:`~tmlt.analytics.SessionProgram`, and non-concrete subclasses of :class:`~tmlt.tune.SessionProgramTuner` (without an associated :class:`~tmlt.analytics.SessionProgram`).
- |PRO| :class:`~tmlt.analytics.SessionProgram` outputs may now be optional.
  A :class:`~tmlt.analytics.SessionProgram` can be constructed without specifying parameters that are of type ``Optional``.
- |PRO| *Backwards-incompatible*: :class:`~tmlt.tune.MetricResult` no longer contains the ``metric`` field.
  In its place, some key information from the metric is now included in the :class:`~tmlt.tune.MetricResult`, and some :class:`~tmlt.tune.Metric` classes have their own :class:`~tmlt.tune.MetricResult` subclasses that add further information (e.g. :class:`~tmlt.tune.JoinedOutputMetricResult` for :class:`~tmlt.tune.JoinedOutputMetric`).
- |PRO| *Backwards-incompatible*: ``SessionProgram.outputs`` has been replaced with :meth:`.SessionProgram.run`.
- |PRO| *Backwards-incompatible*: ``SessionProgramTuner.outputs`` has been replaced with :meth:`.SessionProgramTuner.run`.
  The output of :meth:`.SessionProgramTuner.run` can be passed as input to :meth:`.SessionProgramTuner.error_report`, to compute views and metrics without re-computing DP and baseline outputs.
- |PRO| *Backwards-incompatible*: ``program_parameters`` was renamed to ``parameters`` across :class:`~tmlt.tune.SessionProgramTuner` metrics, baselines, and views.
- |PRO| *Backwards-incompatible*: ``ErrorReport.result_dataframes()`` and ``MultiErrorReport.result_dataframes()`` (deprecated in 0.16.0) were removed in favor of :meth:`.ErrorReport.dataframes` and :meth:`.MultiErrorReport.dataframes`.
- Log messages are now emitted via Python's built-in ``logging`` module.
- The supported version of typeguard has been updated to 4.*.

Added
~~~~~
- Privacy budgets now support division, multiplication, addition and subtraction.
- KeySets can now be initialized directly from a collection of Python tuples using :meth:`.KeySet.from_tuples`.
- |PRO| Added new metrics: :class:`~tmlt.tune.SpuriousCount`, :class:`~tmlt.tune.SuppressionCount`, :class:`~tmlt.tune.HighRelativeErrorCount`, ``CountReleasedRows``, and :class:`~tmlt.tune.CountBaselineRows`.
- |PRO| Added a :meth:`~tmlt.analytics.SessionProgram.output_types` method that returns the types of a program's outputs.
- |PRO| Added a new tuner builder method, ``with_cache_enabled``, for storing query results in cache so that they can be reused in subsequent runs for same queries.

Fixed
~~~~~
- |PRO| Fixed a crash in :meth:`.MultiErrorReport.dataframe` when using non-hashable parameters.

.. _v0.16.1:

0.16.1 - 2024-09-04
-------------------

This release fixes a bug where :class:`~tmlt.analytics.no_privacy_session.NoPrivacySession` would crash when evaluating queries while a view was defined. This affected any users of ``NoPrivacySession``, and any attempt to use a :class:`~tmlt.tune.SessionProgramTuner` on a program that calls :meth:`tmlt.analytics.Session.create_view`.

.. _v0.16.0:

0.16.0 - 2024-08-21
-------------------
This release adds a new :meth:`QueryBuilder.flat_map_by_id <tmlt.analytics.QueryBuilder.flat_map_by_id>` transformation, improved constraint support when using :meth:`~tmlt.analytics.Session.partition_and_create`, and performance improvements.
It also makes minor (but potentially breaking) changes to metrics and error reports.

Added
~~~~~
- Added a new transformation, :meth:`QueryBuilder.flat_map_by_id <tmlt.analytics.QueryBuilder.flat_map_by_id>`, which allows user-defined transformations to be applied to groups of rows sharing an ID on tables with the :class:`~tmlt.analytics.AddRowsWithID` protected change.
- |PRO| Metrics can now return booleans or strings.

Deprecated
~~~~~~~~~~
- |PRO| Deprecated ``ErrorReport.result_dataframes()`` and ``MultiErrorReport.result_dataframes()`` in favor of new :meth:`tmlt.tune.ErrorReport.dataframes()` and :meth:`tmlt.tune.MultiErrorReport.dataframes()` methods.

Fixed
~~~~~
- Significantly improved the performance of coercing Session input dataframe columns to supported types.
- |PRO| Fixed a crash in :meth:`~tmlt.tune.MultiErrorReport.dataframe()` when using list parameters and grouped metrics.

Changed
~~~~~~~
- |PRO| :meth:`~tmlt.tune.ErrorReport.show()` now shows which columns each metric was grouped by.
- |PRO| *Backwards-incompatible*: Metric functions, view functions, and baseline functions are no longer allowed to have a ``self`` parameter. They should instead be annotated with ``@staticmethod``.
- |PRO| :class:`~tmlt.tune.SpuriousRate` and :class:`~tmlt.tune.SuppressionRate` no longer require the user to specify an output if only one exists.
- :meth:`~tmlt.analytics.Session.partition_and_create` can now be used on a table with an :class:`~tmlt.analytics.AddRowsWithID` protected change if a :class:`~tmlt.analytics.MaxRowsPerID` constraint is present, converting the table being partitioned into one with an :class:`~tmlt.analytics.AddMaxRows` protected change.
  The behavior when using :meth:`~tmlt.analytics.Session.partition_and_create` on such a table with a :class:`~tmlt.analytics.MaxGroupsPerID` constraint has not changed.
  If both :class:`~tmlt.analytics.MaxRowsPerID` and :class:`~tmlt.analytics.MaxGroupsPerID` constraints are present, the :class:`~tmlt.analytics.MaxRowsPerID` constraint is ignored and only the :class:`~tmlt.analytics.MaxGroupsPerID` constraint gets applied.

.. _v0.15.0:

0.15.0 - 2024-08-12
-------------------
This release extends the :meth:`~tmlt.analytics.GroupedQueryBuilder.get_bounds` method so it can get upper and lower bounds for each group in a dataframe.
In addition, it changes the object used to represent queries to the new :class:`~tmlt.analytics.Query` class, and updates the format in which table schemas are returned.

It also changes the way custom metrics are specified, with new decorators, a new behavior for the :func:`@metric<tmlt.tune.metric>` decorator, and the old custom metric classes replaced with updated base metric classes.

Added
~~~~~
- Added a dependency on the library ``tabulate`` to improve table displays from :meth:`~tmlt.analytics.Session.describe`.
- |PRO| Ability to specify views on output tables in a list using :data:`tmlt.tune.SessionProgramTuner.views` class variable.
- |PRO| Output validation for custom views/baselines/metrics that cause `RuntimeError` if the output is not valid.
- Added the ability to :meth:`~tmlt.analytics.GroupedQueryBuilder.get_bounds` after calling :meth:`~tmlt.analytics.QueryBuilder.groupby`, for determining upper and lower bounds for a column per group in a differentially private way.

Changed
~~~~~~~
- *Backwards-incompatible*: The :meth:`~tmlt.analytics.QueryBuilder.get_bounds` query now returns a dataframe when evaluated instead of a tuple.
- *Backwards-incompatible*: The :meth:`Session.get_schema() <tmlt.analytics.Session.get_schema>` and :meth:`KeySet.schema() <tmlt.analytics.KeySet.schema>` methods now return a normal dictionary of column names to :class:`~tmlt.analytics.ColumnDescriptor`\ s, rather than a specialized ``Schema`` type.
  This brings them more in line with the rest of the Tumult Analytics API, but could impact code that used some functionality available through the ``Schema`` type.
  Uses of these methods where the result is treated as a dictionary should not be impacted.
- :class:`~tmlt.analytics.QueryBuilder` now returns a :class:`~tmlt.analytics.Query` object instead of a ``QueryExpr`` or ``AggregatedQueryBuilder`` when a query is created.
  This should not affect code using :class:`~tmlt.analytics.QueryBuilder` unless it directly inspects these objects.
- GroupbyCount queries now return :class:`~tmlt.analytics.GroupbyCountQuery`, a subclass of :class:`~tmlt.analytics.Query` that has the :meth:`~tmlt.analytics.GroupbyCountQuery.suppress` post-process method.
- :meth:`~tmlt.analytics.Session.evaluate` now accepts :class:`~tmlt.analytics.Query` objects instead of ``QueryExpr`` objects.
- Replaced asserts with custom exceptions in cases where internal errors are detected.
  Internal errors are now raised as :class:`~tmlt.analytics.AnalyticsInternalError`.
- |PRO| *Backwards-incompatible*: :class:`~tmlt.tune.Metric` and its subclasses only return a single result.
  As a consequence, most metrics now only work on a single baseline, rather than being applied separately to each one.
- |PRO| *Backwards-incompatible*: ``SingleBaselineMetric`` has been renamed to :class:`~tmlt.tune.SingleOutputMetric`.
- |PRO| :class:`~tmlt.tune.Metric`, :class:`~tmlt.tune.SingleOutputMetric`, and :class:`~tmlt.tune.JoinedOutputMetric` now support grouping columns, measure columns, and empty values.
  Accordingly, ``GroupedMetric`` and ``MeasureColumnMetric`` have been removed.
- |PRO| :class:`~tmlt.tune.Metric`, :class:`~tmlt.tune.SingleOutputMetric`, and :class:`~tmlt.tune.JoinedOutputMetric` now support calculating the metric based on a user-supplied function (replacing ``CustomMultiBaselineMetric``, ``CustomSingleOutputMetric``, and ``CustomGroupedMetric``).
- |PRO| :class:`~tmlt.tune.SpuriousRate` and :class:`~tmlt.tune.SuppressionRate` now support calculating error for each group in an output table.
- Updated to Tumult Core 0.16.1.

Removed
~~~~~~~
- QueryExprs (previously in ``tmlt.analytics.query_expr``) have been removed from the Tumult Analytics public API.
  Queries should be created using :class:`~tmlt.analytics.QueryBuilder`, which returns a new :class:`~tmlt.analytics.Query` when a query is created.
- Removed the ``query_expr`` attribute from the :class:`~tmlt.analytics.QueryBuilder` class.
- |PRO| Removed scalar metrics: ``AbsoluteError`` and ``RelativeError``. We recommend using :class:`~tmlt.tune.MedianAbsoluteError` and :class:`~tmlt.tune.MedianRelativeError` instead.
- Removed support for Pandas 1.2 and 1.3 due to a known bug in Pandas versions below 1.4.

.. _v0.14.0:

0.14.0 - 2024-07-18
-------------------

Tumult Analytics 0.14.0 introduces experimental support for Python 3.12.
Full support for Python 3.12 and Pandas 2 will not be available until the release of PySpark 4.0.
In addition, Python 3.7 is no longer supported.

In addition, this release deprecates the ``tmlt.analytics.query_expr`` module.
Use of ``QueryExpr`` and its subtypes to create queries has been discouraged for a long time, and these types will be removed from the Tumult Analytics API in an upcoming release.
Other types from this module have been moved into the ``tmlt.analytics.query_builder`` module, though they may be imported from either until the ``query_expr`` module is removed.

Added
~~~~~
- Tumult Analytics now has experimental support for Python 3.12 using Pandas 2.
- Added a progress bar to :meth:`SessionProgramTuner.multi_error_report <tmlt.tune.SessionProgramTuner.multi_error_report>`.

Changed
~~~~~~~
- Mechanism enums (e.g. :class:`~tmlt.analytics.CountMechanism`) should now be imported from ``tmlt.analytics.query_builder``.
  The current query expression module (``tmlt.analytics.query_expr``) will be removed from the public API in an upcoming release.
- |PRO| Made the return type of ``ErrorReport.result_dataframes`` consistent with ``MultiErrorReport.result_dataframes``.

Removed
~~~~~~~
- Removed support for Python 3.7.

Deprecated
~~~~~~~~~~
- QueryExprs (previously in ``tmlt.analytics.query_expr``) will be removed from the Tumult Analytics public API in an upcoming release.
  Queries should be created using :class:`~tmlt.analytics.QueryBuilder` instead.

.. _v0.13.0:

0.13.0 - 2024-07-03
-------------------
This release makes some supporting classes immutable.
For Pro users, it also adds the ability to calculate metrics for each group in the output. Initially this is available for relative, absolute, and custom error metrics.

Added
~~~~~
- |PRO| Added the ability to calculate metric results for each output group, rather than over the entire dataset. Absolute and relative error metrics support grouping.
- |PRO| Added custom grouped metrics via the ``CustomGroupedMetric`` class.

Changed
~~~~~~~
- Made :class:`~tmlt.analytics.BinningSpec` immutable.
- |PRO| the :func:`@metric<tmlt.tune.metric>` decorator now creates a grouped metric.

.. _v0.12.0:

0.12.0 - 2024-06-18
-------------------

This release adds support for left public joins.
It also includes a new way to specify license file locations.

Added
~~~~~
- |PRO| The Analytics Pro license file path can now be configured programmatically via the :data:`tmlt.cfg.analytics.license_file_path_override` variable instead of using an environment variable.
- Added support for left public joins to :meth:`~.join_public`, previously only inner joins were supported.

Changed
~~~~~~~
- |PRO| Renamed `tmlt.tune.MetricOutput` to :class:`tmlt.tune.MetricResult`.

Fixed
~~~~~
- |PRO| Unpersist cache on ``SessionProgramTuner.outputs``.

.. _v0.11.0:

0.11.0 - 2024-06-05
-------------------

This release introduces support in the query language for suppressing aggregates below a certain threshold, providing an easier and clearer way to express queries where small values must be dropped due to potentially-high noise.

For macOS users, it also introduces native support for Apple silicon, allowing Tumult Analytics to be used on ARM-based Macs without the need for Rosetta.
Take a look at the updated :ref:`installation guide <installation>` for more information about this.
If you have an existing installation that uses Rosetta, ensure that you are using a supported native Python installation when switching over.
Users with Intel-based Macs should not be affected.

Added
~~~~~
- Added a ``tmlt.analytics.query_expr.SuppressAggregates`` query type, for suppressing aggregates less than a certain threshold.
  This is currently only supported for post-processing ``tmlt.analytics.query_expr.GroupByCount`` queries.
  These can be built using the :class:`~tmlt.analytics.QueryBuilder` by calling ``AggregatedQueryBuilder.suppress`` after building a GroupByCount query.
  As part of this change, query builders now return an ``tmlt.analytics.AggregatedQueryBuilder`` instead of a ``tmlt.analytics.query_expr.QueryExpr`` when aggregating;
  the ``tmlt.analytics.AggregatedQueryBuilder`` can be passed to :meth:`Session.evaluate <tmlt.analytics.Session.evaluate>` so most existing code should not need to be migrated.
- :class:`~tmlt.analytics.no_privacy_session.NoPrivacySession` now has an option for whether to enforce suppression (:meth:`NoPrivacySession.Options.enforce_suppression <tmlt.tune.NoPrivacySession.Options.enforce_suppression>`).
- Added :meth:`~tmlt.analytics.KeySet.cache` and :meth:`~tmlt.analytics.KeySet.uncache` methods to :class:`~tmlt.analytics.KeySet` for caching and uncaching the underlying Spark dataframe.
  These methods can be used to improve performance because KeySets follow Spark's lazy evaluation model.

Changed
~~~~~~~
- :class:`~tmlt.analytics.PureDPBudget`, :class:`~tmlt.analytics.ApproxDPBudget`, and :class:`~tmlt.analytics.RhoZCDPBudget` are now immutable classes.
- :class:`~tmlt.analytics.PureDPBudget` and :class:`~tmlt.analytics.ApproxDPBudget` are no longer considered equal if they have the same epsilon and the :class:`~tmlt.analytics.ApproxDPBudget` has a delta of zero.

.. _v0.10.2:

0.10.2 - 2024-05-31
-------------------

Changed
~~~~~~~
- Column order is now preserved when selecting columns from a :class:`~tmlt.analytics.KeySet`.

.. _v0.10.1:

0.10.1 - 2024-05-28
-------------------

This release includes a number of bug fixes.

Changed
~~~~~~~
- |PRO| Error reports now always specify the baseline name for each metric, even if only a single baseline is used.

Fixed
~~~~~
- |PRO| Accessing a program's unprotected inputs and parameters when creating a view on an output table now works as expected.
- |PRO| :meth:`NoPrivacySession.evaluate <tmlt.analytics.no_privacy_session.NoPrivacySession.evaluate>` no longer performs an unnecessary DP evaluation, improving its performance considerably.

.. _v0.10.0:

0.10.0 - 2024-05-17
-------------------

This release adds a new :meth:`~tmlt.analytics.QueryBuilder.get_bounds` aggregation.
It also includes performance improvements for :class:`~tmlt.analytics.KeySet`\ s, and other quality-of-life improvements.
For Pro users, it includes an easier way to define custom metrics, a way for tuners to define views over outputs, and further quality-of-life improvements.

Added
~~~~~
- Added the :meth:`QueryBuilder.get_bounds <tmlt.analytics.QueryBuilder.get_bounds>` function, for determining upper and lower bounds for a column in a differentially private way.

Changed
~~~~~~~
- |PRO| Metric values are now printed in scientific notation if their absolute value
  is greater than 10,000 or less than 1/100.
- If a :class:`~tmlt.analytics.Session.Builder` has only one
  private dataframe *and* that dataframe uses the
  :class:`~tmlt.analytics.AddRowsWithID` protected change,
  the relevant ID space will automatically be added to the Builder when
  :meth:`~tmlt.analytics.Session.Builder.build` is called.
- |PRO| The same is true for :class:`SessionProgram.Builder<tmlt.analytics.SessionProgram.Builder>`
  and :class:`SessionProgramTuner.Builder<tmlt.tune.SessionProgramTuner.Builder>`.
- |PRO| Custom metrics can be defined using the :func:`@metric<tmlt.tune.metric>` decorator.
- :class:`~tmlt.analytics.KeySet` is now an abstract class, in order to
  make some KeySet operations (column selection after cross-products) more
  efficient.
  Behavior is unchanged for users of the :meth:`~tmlt.analytics.KeySet.from_dict`
  and :meth:`~tmlt.analytics.KeySet.from_dataframe` constructors.
- |PRO| Allow views on output tables before applying metrics by using the :func:`@view<tmlt.tune.view>` decorator.
  The views are persisted by default and unpersisted before the tuner is destroyed.

Fixed
~~~~~
- Stopped trying to set extra options for Java 11 and removed error when options are not set. Removed ``get_java_11_config()``.
- Updated minimum supported Spark version to 3.1.1 to prevent Java 11 error.

.. _v0.9.0:

0.9.0 - 2024-04-16
------------------

This release introduces a number of proprietary features for parameterizing and tuning differentially private programs.
It also contains bug fixes and documentation improvements.

Note that the 0.9.x release series will be the last to support Python 3.7, which has not been receiving security updates for several months.
If this is a problem, please `reach out to us <mailto:info@tmlt.io>`_.

Added
~~~~~
- |PRO| Added :class:`~.SessionProgram` for defining structured DP programs using the :class:`~.Session` API.
- |PRO| Added :class:`~.SessionProgramTuner` and a collection of :ref:`metrics<metrics>` for tuning :class:`~.SessionProgram`\ s.
- |PRO| Added :class:`~.no_privacy_session.NoPrivacySession`, which allows non-private query execution with the same interface as :class:`~.Session`.

Changed
~~~~~~~
- :class:`~tmlt.analytics.KeySet` equality is now performed without converting the underlying dataframe to Pandas.
- :meth:`~tmlt.analytics.Session.partition_and_create`: the ``column`` and ``splits`` arguments are now annotated as required.
- The minimum supported version of Tumult Core is now 0.13.0.
- The :meth:`QueryBuilder.variance <tmlt.analytics.QueryBuilder.variance>`, :meth:`QueryBuilder.stdev <tmlt.analytics.QueryBuilder.stdev>`, :meth:`GroupedQueryBuilder.variance <tmlt.analytics.GroupedQueryBuilder.variance>`, and :meth:`GroupedQueryBuilder.stdev <tmlt.analytics.GroupedQueryBuilder.stdev>` methods now calculate the sample variance or standard deviation, rather than the population variance or standard deviation.

Removed
~~~~~~~
- *Backwards-incompatible*: The ``stability`` and ``grouping_column`` parameters to :meth:`Session.from_dataframe <tmlt.analytics.Session.from_dataframe>` and :meth:`Session.Builder.with_private_dataframe <tmlt.analytics.Session.Builder.with_private_dataframe>` have been removed (deprecated since :ref:`0.7.0 <v0.7.0>`).
  As a result, the ``protected_change`` parameter to those methods is now required.

Fixed
~~~~~
- The error message when attempting to overspend an :class:`~tmlt.analytics.ApproxDPBudget` now more clearly indicates which component of the budget was insufficient to evaluate the query.
- :meth:`QueryBuilder.get_groups <tmlt.analytics.QueryBuilder.get_groups>` now automatically excludes ID columns if no columns are specified.
- Flat maps now correctly ignore ``max_rows`` when it does not apply.
  Previously they would raise a warning saying that ``max_rows`` was ignored, but would still use it to limit the number of rows in the output.

.. _v0.8.3:

0.8.3 - 2024-02-27
------------------

This is a maintenance release that adds support for newer versions of Tumult Core. It contains no API changes.

.. _v0.8.2:

0.8.2 - 2023-11-29
------------------

This release addresses a serious security vulnerability in PyArrow: `CVE-2023-47248 <https://nvd.nist.gov/vuln/detail/CVE-2023-47248>`__.
It is **strongly recommended** that all users update to this version of Analytics or apply one of the mitigations described in the `GitHub Advisory <https://github.com/advisories/GHSA-5wvp-7f3h-6wmm>`__.

Changed
~~~~~~~
- Increased minimum supported version of Tumult Core to 0.11.5.
  As a result:

  - Increased the minimum supported version of PyArrow to 14.0.1 for Python 3.8 and above.
  - Added dependency on ``pyarrow-hotfix`` on Python 3.7.
    Note that if you are using Python 3.7, the hotfix must be imported before using PySpark in order to be effective.
    Analytics imports the hotfix, so importing Analytics before using Spark will also work.

.. _v0.8.1:

0.8.1 - 2023-10-30
------------------

This release adds support for Python 3.11, as well as compatibility with newer versions of various dependencies, including PySpark.
It also includes documentation improvements, but no API changes.

.. _v0.8.0:

0.8.0 - 2023-08-15
------------------

This is a maintenance release that addresses a performance regression for complex queries and improves naming consistency in some areas of the Tumult Analytics API.

Added
~~~~~
- Added the :meth:`QueryBuilder.get_groups <tmlt.analytics.QueryBuilder.get_groups>` function, for determining groupby keys for a table in a differentially private way.

Changed
~~~~~~~
- *Backwards-incompatible*: Renamed ``DropExcess.max_records`` to :attr:`~tmlt.analytics.TruncationStrategy.DropExcess.max_rows`.
- *Backwards-incompatible*: Renamed ``FlatMap.max_num_rows`` to ``FlatMap.max_rows``.
- Changed the name of an argument for :meth:`QueryBuilder.flat_map()<tmlt.analytics.QueryBuilder.flat_map>` from ``max_num_rows`` to ``max_rows``. The old ``max_num_rows`` argument is deprecated and will be removed in a future release.

Fixed
~~~~~
- Upgrades to version 0.11 of Tumult Core.
  This addresses a performance issue introduced in Tumult Analytics 0.7.0 where some complex queries compiled much more slowly than they had previously.

.. _v0.7.3:

0.7.3 - 2023-07-13
------------------

Fixed
~~~~~
- Fixed a crash in public and private joins.

.. _v0.7.2:

0.7.2 - 2023-06-15
------------------

This release adds support for running Tumult Analytics on Python 3.10.
It also enables adding continuous Gaussian noise to query results, and addresses a number of bugs and API inconsistencies.

Added
~~~~~
- Tumult Analytics now supports Python 3.10 in addition to the previously-supported versions.
- Queries evaluated with zCDP budgets can now use continuous Gaussian noise, allowing the use of Gaussian noise for queries with non-integer results.

Changed
~~~~~~~
- The :meth:`QueryBuilder.replace_null_and_nan()<tmlt.analytics.QueryBuilder.replace_null_and_nan>` and :meth:`QueryBuilder.drop_null_and_nan()<tmlt.analytics.QueryBuilder.drop_null_and_nan>` methods now accept empty column specifications on tables with an :class:`~tmlt.analytics.AddRowsWithID` protected change.
  Replacing/dropping nulls on ID columns is still not allowed, but the ID column will now automatically be excluded in this case rather than raising an exception.
- :meth:`BinningSpec.bins()<tmlt.analytics.BinningSpec.bins>` used to only include the NaN bin if the provided bin edges were floats.
  However, float-valued columns can be binned with integer bin edges, which resulted in a confusing situation where a :class:`~tmlt.analytics.BinningSpec` could indicate that it would not use a NaN bin but still place values in the NaN bin.
  To avoid this, :meth:`BinningSpec.bins()<tmlt.analytics.BinningSpec.bins>` now always includes the NaN bin if one was specified, regardless of whether the bin edge type can represent NaN values.
- The automatically-generated bin names in :class:`~tmlt.analytics.BinningSpec` now quote strings when they are used as bin edges.
  For example, the bin generated by ``BinningSpec(["0", "1"])`` is now ``['0', '1']`` where it was previously ``[0, 1]``.
  Bins with edges of other types are not affected.

Fixed
~~~~~
- Creating a :class:`~tmlt.analytics.Session` with multiple tables in an ID space used to fail if some of those tables' ID columns allowed nulls and others did not.
  This no longer occurs, and in such cases all of the tables' ID columns are made nullable.

.. _v0.7.1:

0.7.1 - 2023-05-23
------------------

This is a maintenance release that mainly contains documentation updates.
It also fixes a bug where installing Tumult Analytics using pip 23 and above could fail due to a dependency mismatch.

.. _v0.7.0:

0.7.0 - 2023-04-27
------------------

This release adds support for *privacy identifiers*:
Tumult Analytics can now protect input tables in which the differential privacy guarantee needs to hide the presence of arbitrarily many rows sharing the same value in a particular column.
For example, this may be used to protect each user of a service when every row in a table is associated with a user ID.

Privacy identifiers are set up using the new :class:`~tmlt.analytics.AddRowsWithID` protected change.
A number of features have been added to the API to support this, including alternative behaviors for various query transformations when working with IDs and the new concept of :ref:`constraints`.
To get started with these features, take a look at the new :ref:`Working with privacy IDs <privacy-id-basics>` and :ref:`Doing more with privacy IDs <advanced-privacy-ids>` tutorials.

Added
~~~~~
- A new :class:`~tmlt.analytics.AddRowsWithID` protected change has been added, which protects the addition or removal of all rows with the same value in a specified column.
  See the documentation for :class:`~tmlt.analytics.AddRowsWithID` and the :ref:`Doing more with privacy IDs <advanced-privacy-ids>` tutorial for more information.

  - When creating a Session with :class:`~tmlt.analytics.AddRowsWithID` using a :class:`Session.Builder<tmlt.analytics.Session.Builder>`, you must use the new :meth:`~tmlt.analytics.Session.Builder.with_id_space` method to specify the identifier space(s) of tables using this protected change.
  - When creating a Session with :meth:`Session.from_dataframe()<tmlt.analytics.Session.from_dataframe>`, specifying an ID space is not necessary.

- :class:`~tmlt.analytics.QueryBuilder` has a new method, :meth:`~tmlt.analytics.QueryBuilder.enforce`, for enforcing :ref:`constraints` on a table.
- A new method, :meth:`Session.describe()<tmlt.analytics.Session.describe>`, has been added to provide a summary of the tables in a :class:`~tmlt.analytics.Session`, or of a single table or the output of a query.

Changed
~~~~~~~
- :meth:`QueryBuilder.join_private()<tmlt.analytics.QueryBuilder.join_private>` now accepts the name of a private table as ``right_operand``.
  For example, ``QueryBuilder("table").join_private("foo")`` is equivalent to ``QueryBuilder("table").join_private(QueryBuilder("foo"))``.
- The ``max_num_rows`` parameter to :meth:`QueryBuilder.flat_map()<tmlt.analytics.QueryBuilder.flat_map>` is now optional when applied to tables with an :class:`~tmlt.analytics.AddRowsWithID` protected change.
- *Backwards-incompatible*: The parameters to :meth:`QueryBuilder.flat_map()<tmlt.analytics.QueryBuilder.flat_map>` have been reordered, moving ``max_num_rows`` to be the last parameter.
- *Backwards-incompatible*: The lower and upper bounds for quantile, sum, average, variance, and standard deviation queries can no longer be equal to one another.
  The lower bound must now be strictly less than the upper bound.
- *Backwards-incompatible*: Renamed :meth:`QueryBuilder.filter()<tmlt.analytics.QueryBuilder.filter>` ``predicate`` argument to ``condition``.
- *Backwards-incompatible*: Renamed ``tmlt.analytics.query_expr.Filter`` query expression ``predicate`` property to ``condition``.
- *Backwards-incompatible*: Renamed :meth:`KeySet.filter()<tmlt.analytics.KeySet.filter>` ``expr`` argument to ``condition``.

Deprecated
~~~~~~~~~~
- The ``stability`` and ``grouping_column`` parameters to :class:`Session.from_dataframe()<tmlt.analytics.Session.from_dataframe>` and :class:`Session.Builder.with_private_dataframe()<tmlt.analytics.Session.Builder.with_private_dataframe>` are deprecated, and will be removed in a future release.
  The ``protected_change`` parameter should be used instead, and will become required.

Removed
~~~~~~~
- The ``attr_name`` parameter to :class:`Session.partition_and_create()<tmlt.analytics.Session.partition_and_create>`, which was deprecated in version 0.5.0, has been removed.

Fixed
~~~~~
- :meth:`Session.add_public_datafame()<tmlt.analytics.Session.add_public_dataframe>` used to allow creation of a public table with the same name as an existing public table, which was neither intended nor fully supported by some :class:`~tmlt.analytics.Session` methods.
  It now raises a ``ValueError`` in this case.
- Some query patterns on tables containing nulls could cause grouped aggregations to produce the wrong set of group keys in their output.
  This no longer happens.
- In certain unusual cases, join transformations could erroneously drop rows containing nulls in columns that were not being joined on.
  These rows are no longer dropped.

.. _v0.6.1:

0.6.1 - 2022-12-07
------------------

This is a maintenance release which introduces a number of documentation improvements, but has no publicly-visible API changes.

.. _v0.6.0:

0.6.0 - 2022-12-06
------------------

.. _changelog#protected-change:

This release introduces a new way to specify what unit of data is protected by the privacy guarantee of a :class:`~tmlt.analytics.Session`.
A new ``protected_change`` parameter is available when creating a :class:`~tmlt.analytics.Session`, taking an instance of the new :class:`~tmlt.analytics.ProtectedChange` class which describes the largest unit of data in the resulting table on which the differential privacy guarantee will hold.
See the :ref:`API documentation<privacy-guarantees>` for more information about the available protected changes and how to use them.

The ``stability`` and ``grouping_column`` parameters which were used to specify this information are still accepted, and work as before, but they will be deprecated and eventually removed in future releases.
The default behavior of assuming ``stability=1`` if no other information is given will also be deprecated and removed, on a similar timeline to ``stability`` and ``grouping_column``; instead, explicitly specify ``protected_change=AddOneRow()``.
These changes should make the privacy guarantees provided by the :class:`~tmlt.analytics.Session` interface easier to understand and harder to misuse, and allow for future support for other units of protection that were not representable with the existing API.

Added
~~~~~
- As described above, :meth:`Session.Builder.with_private_dataframe <tmlt.analytics.Session.Builder.with_private_dataframe>` and :meth:`Session.from_dataframe <tmlt.analytics.Session.from_dataframe>` now have a new parameter, ``protected_change``.
  This parameter takes an instance of one of the classes subclassing :class:`~tmlt.analytics.ProtectedChange` module, specifying the unit of data in the corresponding table to be protected.

0.5.1 - 2022-11-16
------------------

Changed
~~~~~~~

-  Updated to Tumult Core 0.6.0.

.. _v0.5.0:

0.5.0 - 2022-10-17
------------------

Added
~~~~~

-  Added a diagram to the API reference page.
-  Analytics now does an additional Spark configuration check for users running Java 11+ at the time of Analytics Session initialization. If the user is running Java 11 or higher with an incorrect Spark configuration, Analytics raises an informative exception.
-  Added a method to check that basic Analytics functionality works (``tmlt.analytics.utils.check_installation``).

Changed
~~~~~~~

-  *Backwards-incompatible*: Changed argument names for ``QueryBuilder.count_distinct`` and ``KeySet.__getitem__`` from ``cols`` to ``columns``, for consistency. The old argument has been deprecated, but is still available.
-  *Backwards-incompatible*: Changed the argument name for ``Session.partition_and_create`` from ``attr_name`` to ``column``. The old argument has been deprecated, but is still available.
-  Improved the error message shown when a filter expression is invalid.
-  Updated to Tumult Core 0.5.0.
   As a result, ``python-flint`` is no longer a transitive dependency, simplifying the Analytics installation process.

Deprecated
~~~~~~~~~~

-  The contents of the ``cleanup`` module have been moved to the ``utils`` module. The ``cleanup`` module will be removed in a future version.

.. _v0.4.2:

0.4.2 - 2022-09-06
------------------

Fixed
~~~~~

-  Switched to Core version 0.4.3 to avoid warnings when evaluating some queries.

.. _v0.4.1:

0.4.1 - 2022-08-25
------------------

Added
~~~~~

-  Added ``QueryBuilder.histogram`` function, which provides a shorthand for generating binned data counts.
-  Analytics now checks to see if the user is running Java 11 or higher. If they are, Analytics either sets the appropriate Spark options (if Spark is not yet running) or raises an informative exception (if Spark is running and configured incorrectly).

Changed
~~~~~~~

-  Improved documentation for ``QueryBuilder.map`` and ``QueryBuilder.flat_map``.

Fixed
~~~~~

-  Switched to Core version 0.4.2, which contains a fix for an issue that sometimes caused queries to fail to be compiled.

.. _v0.4.0:

0.4.0 - 2022-07-22
------------------

Added
~~~~~

-  ``Session.from_dataframe`` and ``Session.Builder.with_private_dataframe`` now have a ``grouping_column`` option and support non-integer stabilities.
   This allows setting up grouping columns like those that result from grouping flatmaps when loading data.
   This is an advanced feature, and should be used carefully.

.. _v0.3.0:

0.3.0 - 2022-06-23
------------------

Added
~~~~~

-  Added ``QueryBuilder.bin_column`` and an associated ``BinningSpec`` type.
-  Dates may now be used in ``KeySet``\ s.
-  Added support for DataFrames containing NaN and null values. Columns created by Map and FlatMap are now marked as potentially containing NaN and null values.
-  Added ``QueryBuilder.replace_null_and_nan`` function, which replaces null and NaN values with specified defaults.
-  Added ``QueryBuilder.replace_infinite`` function, which replaces positive and negative infinity values with specified defaults.
-  Added ``QueryBuilder.drop_null_and_nan`` function, which drops null and NaN values for specified columns.
-  Added ``QueryBuilder.drop_infinite`` function, which drops infinite values for specified columns.
-  Aggregations (sum, quantile, average, variance, and standard deviation) now silently drop null and NaN values before being performed.
-  Aggregations (sum, quantile, average, variance, and standard deviation) now silently clamp infinite values (+infinity and -infinity) to the query’s lower and upper bounds.
-  Added a ``cleanup`` module with two functions: a ``cleanup`` function to remove the current temporary table (which should be called before ``spark.stop()``), and a ``remove_all_temp_tables`` function that removes all temporary tables ever created by Analytics.
-  Added a topic guide in the documentation for Tumult Analytics’ treatment of null, NaN, and infinite values.

Changed
~~~~~~~

-  *Backwards-incompatible*: Sessions no longer allow DataFrames to contain a column named ``""`` (the empty string).
-  *Backwards-incompatible*: You can no longer call ``Session.Builder.with_privacy_budget`` multiple times on the same builder.
-  *Backwards-incompatible*: You can no longer call ``Session.add_private_data`` multiple times with the same source id.
-  *Backwards-incompatible*: Sessions now use the DataFrame’s schema to determine which columns are nullable.

Removed
~~~~~~~

-  *Backwards-incompatible*: Removed ``groupby_public_source`` and ``groupby_domains`` from ``QueryBuilder``.
-  *Backwards-incompatible*: ``Session.from_csv`` and CSV-related methods on ``Session.Builder`` have been removed.
   Instead, use ``spark.read.csv`` along with ``Session.from_dataframe`` and other dataframe-based methods.
-  *Backwards-incompatible*: Removed ``validate`` option from ``Session.from_dataframe``, ``Session.add_public_dataframe``, ``Session.Builder.with_private_dataframe``, ``Session.Builder.with_public_dataframe``.
-  *Backwards-incompatible*: Removed ``KeySet.contains_nan_or_null``.

Fixed
~~~~~

-  *Backwards-incompatible*: ``KeySet``\ s now explicitly check for and disallow the use of floats and timestamps as keys.
   This has always been the intended behavior, but it was previously not checked for and could work or cause non-obvious errors depending on the situation.
-  ``KeySet.dataframe()`` now always returns a dataframe where all rows are distinct.
-  Under certain circumstances, evaluating a ``GroupByCountDistinct`` query expression used to modify the input ``QueryExpr``.
   This no longer occurs.
-  It is now possible to partition on a column created by a grouping flat map, which used to raise exception from Core.

.. _v0.2.1:

0.2.1 - 2022-04-14 (internal release)
-------------------------------------

Added
~~~~~

-  Added support for basic operations (filter, map, etc.) on Spark date and timestamp columns.
   ``ColumnType`` has two new variants, ``DATE`` and ``TIMESTAMP``, to support these.
-  Future documentation will now include any exceptions defined in Analytics.

Changed
~~~~~~~

-  Switch session to use Persist/Unpersist instead of Cache.

.. _v0.2.0:

0.2.0 - 2022-03-28 (internal release)
-------------------------------------

Removed
~~~~~~~

-  Multi-query evaluate support is entirely removed.
-  Columns that are neither floats nor doubles will no longer be checked for NaN values.
-  The ``BIT`` variant of the ``ColumnType`` enum was removed, as it was not supported elsewhere in Analytics.

Changed
~~~~~~~

-  *Backwards-incompatible*: Renamed ``query_exprs`` parameter in ``Session.evaluate`` to ``query_expr``.
-  *Backwards-incompatible*: ``QueryBuilder.join_public`` and the ``JoinPublic`` query expression can now accept public tables specified as Spark dataframes. The existing behavior using public source IDs is still supported, but the ``public_id`` parameter/property is now called ``public_table``.
-  Installation on Python 3.7.1 through 3.7.3 is now allowed.
-  KeySets now do type coercion on creation, matching the type coercion that Sessions do for private sources.
-  Sessions created by ``partition_and_create`` must be used in the order they were created, and using the parent session will forcibly close all child sessions.
   Sessions can be manually closed with ``session.stop()``.

Fixed
~~~~~

-  Joining with a public table that contains no NaNs, but has a column where NaNs are allowed, previously caused an error when compiling queries. This is now handled correctly.

.. _v0.1.1:

0.1.1 - 2022-02-28 (internal release)
-------------------------------------

Added
~~~~~

-  Added a ``KeySet`` class, which will eventually be used for all GroupBy queries.
-  Added ``QueryBuilder.groupby()``, a new group-by based on ``KeySet``\ s.

Changed
~~~~~~~

-  The Analytics library now uses ``KeySet`` and ``QueryBuilder.groupby()`` for all
   GroupBy queries.
-  The various ``Session`` methods for loading in data from CSV no longer support loading the data’s schema from a file.
-  Made Session return a more user-friendly error message when the user provides a privacy budget of 0.
-  Removed all instances of the old name of this library, and replaced them with “Analytics”

Deprecated
~~~~~~~~~~

-  ``QueryBuilder.groupby_domains()`` and ``QueryBuilder.groupby_public_source()`` are now deprecated in favor of using ``QueryBuilder.groupby()`` with ``KeySet``\ s.
   They will be removed in a future version.

.. _v0.1.0:

0.1.0 - 2022-02-15 (internal release)
-------------------------------------

Added
~~~~~

-  Initial release.
