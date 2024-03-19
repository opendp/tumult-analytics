.. _Tutorials:

Tutorials
=========

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2024

This tutorial series introduces the key concepts of the Tumult Analytics API, and help you design and implement your first differentially private data publication. The series is split in three sections.

#. **Interface basics and first queries** introduces the fundamentals of Tumult Analytics, and explains how to compute simple differentially private aggregations. Go through these first; this section is a prerequisite to the other two.

   .. toctree::
      :maxdepth: 1

      first-steps
      privacy-budget-basics
      clamping-bounds
      groupby-queries

   |
#. **Advanced queries and privacy notions** demonstrates how to perform common data transformations on Tumult Analytics, and how to use it in contexts where a single person can contribute to multiple rows in the input data. You can skip this section if you want to directly get started with error measurement and tuning.

   .. toctree::
      :maxdepth: 1

      simple-transformations
      privacy-id-basics
      more-with-privacy-ids

   |

