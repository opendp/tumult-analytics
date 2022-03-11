# Tumult Analytics

The Tumult Analytics (formerly SafeTables SDK) product allows users to execute differentially private operations on
data without having to worry about the privacy implementation which is handled
automatically by the API.

For more information, refer to the [software documentation](https://docs.tumultlabs.io/pkg/analytics/) and references therein.

<placeholder: add notice if required>

## Overview

In order to write these private queries, the user may leverage the following.
* QueryBuilder - a tool to create differentially private queries.
* Session - an object to keep track of privacy and interactively evaluate private queries.

See [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.

## Testing

To run the tests, install the required dependencies from the `test_requirements.txt`

```
pip install -r test_requirements.txt
```

*Fast Tests:*

```
nosetests test/unit test/system -a '!slow'
```

*Slow Tests:*

Slow tests are tests that we run less frequently because they take a long time to run, or the functionality has been tested by other fast tests.

```
nosetests test/unit test/system -a 'slow'
```
