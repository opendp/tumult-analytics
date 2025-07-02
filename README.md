[![PyPI - Version](https://img.shields.io/pypi/v/tmlt-analytics?color=006dad)](https://pypi.org/project/tmlt-analytics/) |
[![Documentation - Latest](https://img.shields.io/badge/documentation-latest-cc3d56)](https://docs.tmlt.dev/analytics/latest/) |
[![Join our Slack!](https://img.shields.io/badge/Join%20our%20Slack!-634ad3?logo=slack)][slack]

[slack]: https://join.slack.com/t/opendp/shared_invite/zt-1aca9bm7k-hG7olKz6CiGm8htI2lxE8w

# Tumult Analytics — an OpenDP project

Tumult Analytics is a Python library to execute differentially private
operations on data, with a strong emphasis on usability and scalability. It is
built atop the [Tumult Core library](https://github.com/opendp/tumult-core).

It was originally developed by
[Tumult Labs](https://www.linkedin.com/company/tmltlabs), and joined the
[OpenDP project](https://opendp.org) after the Tumult Labs team joined LinkedIn.

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

If you have any questions, feedback, or feature requests, please reach out via the [OpenDP Slack][slack].

## Contributing

We welcome external volunteers! If you are interested in contributing, please
let us know on [Slack][slack].

See [CONTRIBUTING.md](https://github.com/opendp/tumult-analytics/blob/main/CONTRIBUTING.md) for information about installing our development dependencies and running tests.

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

Tumult Analytics' source code is licensed under the Apache License, version 2.0
(Apache-2.0). Tumult Analytics' documentation is licensed under Creative Commons
Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0).
