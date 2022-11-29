# Contributing

We do not yet have a process to accept external contributions.
If you are interested in contributing, you can let us know on our [Slack workspace](https://join.slack.com/t/tmltdev/shared_invite/zt-1bky0mh9v-vOB8azKAVoxmzJDUdWd5Wg) or by email at [support@tmlt.io](mailto:support@tmlt.io).

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
