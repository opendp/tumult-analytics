[![PyPI - Version](https://img.shields.io/pypi/v/tmlt-analytics?color=006dad)](https://pypi.org/project/tmlt-analytics/) |
[![Documentation - Latest](https://img.shields.io/badge/documentation-latest-cc3d56)](https://docs.tmlt.dev/analytics/latest/) |
[![Join our Slack!](https://img.shields.io/badge/Join%20our%20Slack!-634ad3?logo=slack)](https://tmlt.dev/slack)

# Tumult Analytics

Tumult Analytics is a library that allows users to execute differentially private operations on
data without having to worry about the privacy implementation, which is handled
automatically by the API. It is built atop the [Tumult Core library](https://gitlab.com/tumult-labs/core).

## Demo video

Want to see Tumult Analytics in action? Check out this video introducing the
interface fundamentals:

[![Screenshot of the demo video](https://img.youtube.com/vi/SNfbYOp0CEs/0.jpg)](https://www.youtube.com/watch?v=SNfbYOp0CEs)

A selection of more advanced features is shown on the second part of this demo,
in a [separate video](https://www.youtube.com/watch?v=BRUPlfwzHHo).

## Installation

See the [installation instructions in the documentation](https://docs.tmlt.dev/analytics/latest/installation.html#prerequisites)
for information about setting up prerequisites such as Spark.

Once the prerequisites are installed, you can install Tumult Analytics using [pip](https://pypi.org/project/pip).

```bash
pip install tmlt.analytics
```

## Documentation

The full documentation is located at https://docs.tmlt.dev/analytics/latest/.

## Support

If you have any questions, feedback, or feature requests, please reach out to us on [Slack](https://tmlt.dev/slack).

## Contributing

We do not yet have a process in place to accept external contributions, but we are open to collaboration opportunities.
If you are interested in contributing, please let us know [via Slack](https://tmlt.dev/slack).

See [CONTRIBUTING.md](https://gitlab.com/tumult-labs/analytics/-/blob/dev/CONTRIBUTING.md) for information about installing our development dependencies and running tests.

## Citing Tumult Analytics

If you use Tumult Analytics for a scientific publication, we would appreciate citations to the published software or/and its whitepaper. Both citations can be found below; for the software citation, please replace the version with the version you are using.

```
@software{tumultanalyticssoftware,
    author = {Tumult Labs},
    title = {Tumult {{Analytics}}},
    month = dec,
    year = 2022,
    version = {latest},
    url = {https://tmlt.dev}
}
```

```
@article{tumultanalyticswhitepaper,
  title={Tumult {{Analytics}}: a robust, easy-to-use, scalable, and expressive framework for differential privacy},
  author={Berghel, Skye and Bohannon, Philip and Desfontaines, Damien and Estes, Charles and Haney, Sam and Hartman, Luke and Hay, Michael and Machanavajjhala, Ashwin and Magerlein, Tom and Miklau, Gerome and Pai, Amritha and Sexton, William and Shrestha, Ruchit},
  journal={arXiv preprint arXiv:2212.04133},
  month = dec,
  year={2022}
}
```

## License

Copyright Tumult Labs 2024

Tumult Analytics' source code is licensed under the Apache License, version 2.0 (Apache-2.0).
Tumult Analytics' documentation is licensed under
Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0).
