# Environment Setup

First, install [Poetry](https://python-poetry.org) on your system for package management.

<!--- TODO: Remove this once this project becomes public and a password is no
longer required for accessing it.  --->
Set up a GitLab access token which can read from the Core repository's package registry (the `read_api` scope for a personal access token).
Run
```bash
poetry config http-basic.tumult-core <gitlab_username> <gitlab_token>
```
so that Poetry can authenticate with the Core package registry.

Once this is set up, you should be able to just run `poetry install` from the root of this repository to get Analytics, its dependencies, and our development tools all set up.
