# Contributing

We are happy to accept external contributions! 💖

## Contact

First, let us know what you would like to contribute. Feel free to:

- report a bug or request a feature on [GitHub](https://github.com/opendp/tumult-analytics/issues);
- send general queries to info@opendp.org, or email security@opendp.org if it is related to security;
- ask any question on our [Slack][slack] instance.

[slack]: https://join.slack.com/t/opendp/shared_invite/zt-1aca9bm7k-hG7olKz6CiGm8htI2lxE8w

## Local development

### Installation

We recommend using the lowest supported Python version for local development. Tools like [pyenv](https://github.com/pyenv/pyenv) may be useful for managing your Python version.

We use [Poetry](https://python-poetry.org/) for dependency management during development. To set up your environment, start by installing the prerequisites listed in the [installation instructions](https://docs.tmlt.dev/platform/latest/installation.html#prerequisites), then install Poetry using `pip install poetry`, and finally install our dev dependencies by running `poetry install` from the root of this repository.

### Basic usage

You can then locally run our linters and tests by running:
```bash
make lint
make test
```
from the repository root directory.

Note that some operating systems, including macOS, include versions of `make` that are too old to run this project's [Makefile](./Makefile) correctly. macOS users should [install a newer version of make using Homebrew](https://formulae.brew.sh/formula/make#default).

Behind the scenes, these commands use the Poetry environment, and rely on [nox](https://nox.thea.codes/en/stable/index.html) for test automation. You can get a bit more fine-grained control and access additional tools by running nox commands directly (see [this tutorial](https://nox.thea.codes/en/stable/tutorial.html)). You can find a list of available nox sessions using `poetry run nox --list`, then run one of these sessions using e.g. `poetry run nox -s test-fast`.
### Testing

Our unit tests are run with [pytest](https://docs.pytest.org/en/stable/getting-started.html). You can run smaller subsets of tests by using pytest directly. E.g., to check tests in a particular test file, you can run:

```bash
poetry run pytest test/unit/a_test_file.py
### Documentation

The documentation is built using [Sphinx](https://www.sphinx-doc.org/), and relies on [autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html) to generate the API reference.

To build the documentation locally, run:
```bash
make docs
```
the generated HTML pages are available in the `public` directory.

Note that our API reference is manually organized, and does not follow the internal package structure. If you add a new public class or method to Tumult Analytics, add it to an `autosummary` directive in the relevant `.rst` file under `doc/reference`.

### Cleanup

Running linters, tests, or building docs tends to generate a lot of files in the repository that you generally don't want to keep around. Simply run `make clean` to get rid of all those. This is particularly useful when working on the documentation; Sphinx tends to get confused by files generated in previous documentation builds.

## Final thoughts

We want to actively encourage contributions and help you merge your bug fixes or new features. Please don't hesitate to ask us for help on [Slack][slack] if you encounter any difficulty during the process!
