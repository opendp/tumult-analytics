# Contributing

We are not yet accepting external contributions, but please let us know at [support@tmlt.io](mailto:support@tmlt.io) if you are interested in contributing.

## Local development

We use [Poetry](https://python-poetry.org/) for dependency management during development.
To work locally, install Poetry, and then install our dev dependencies by running `poetry install` from the root of this repository.

See the [installation instructions](https://docs.tmlt.dev/analytics/latest/installation.html#installation-instructions) for more information about prerequisites.

Our linters and tests can be run locally with
```bash
make lint
make test
```
from the repository root directory.
This requires having an activated virtual environment with our dev dependencies installed.

Note that some operating systems, including macOS, include versions of make that are too old to run this Makefile correctly. macOS users should [install a newer version of make using Homebrew](https://formulae.brew.sh/formula/make#default).
