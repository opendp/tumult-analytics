# Contributing

We are happy to accept external contributions! 💖

First, let us know what you would like to contribute. Feel free to:

- report a bug or request a feature by filing an issue on our [GitHub](https://github.com/opendp/tumult-analytics/issues);
- send general queries to info@opendp.org, or email security@opendp.org if it is related to security;
- ask any question on our [Slack][slack] instance. Tumult Analytics maintainers are active on most of the public channels, the `lib-dev` and `lib-support` channels are a great place to start interacting with the development community.

[slack]: https://join.slack.com/t/opendp/shared_invite/zt-1aca9bm7k-hG7olKz6CiGm8htI2lxE8w


Second, anyone can send us a Pull Request from a forked repo per Github's [documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Ideally Pull Requests are linked to an issue so the maintainers can easily understand the problem being solved. We try to link all Pull Requests to issues ourselves, so creating and commenting on issues is an easy way to get involved.

## Local development

### Installation

We use [`uv`](https://docs.astral.sh/uv/) for dependency management during development. To set up your environment, install `uv` by following its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then install the prerequisites listed in the [Tumult Analytics installation instructions](https://opendp.github.io/tumult-docs/analytics/latest/installation.html#prerequisites), and finally install our dev dependencies by running `uv sync` from the root of this repository.

To minimize compatibility issues, doing development on the oldest supported Python minor version (currently 3.9) is strongly recommended.
If you are using `uv` to manage your Python installations, running `uv sync` without an existing virtual environment should automatically install and use an appropriate Python version.

### Basic usage

You can then locally run our linters and tests by running:
```bash
make lint
make test
```
from the repository root directory.

Note that some operating systems, including macOS, include versions of `make` that are too old to run this project's [Makefile](./Makefile) correctly. macOS users should [install a newer version of make using Homebrew](https://formulae.brew.sh/formula/make#default).

Behind the scenes, these commands use the `uv` environment, and rely on [nox](https://nox.thea.codes/en/stable/index.html) for test automation. You can get a bit more fine-grained control and access additional tools by running nox commands directly (see [this tutorial](https://nox.thea.codes/en/stable/tutorial.html)). You can find a list of available nox sessions using `uv run nox --list`, then run one of these sessions using e.g. `uv run nox -s test-fast`.

### Testing

Our unit tests are run with [pytest](https://docs.pytest.org/en/stable/getting-started.html). You can run smaller subsets of tests by using pytest directly. For example, to check tests in a particular test file, run:

```bash
uv run pytest test/unit/a_test_file.py
```

You can also filter to specific tests or specific groups of tests using [pytest filters](https://docs.pytest.org/en/stable/how-to/usage.html#specifying-which-tests-to-run). We tag all of our longest-running tests with the `slow` tag, so they can be skipped easily when you want faster feedback (though make sure they pass before you submit!):

```bash
uv run pytest -m "not slow"
```

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
