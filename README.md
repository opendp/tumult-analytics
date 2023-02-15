# Tumult Analytics

Tumult Analytics is a library that allows users to execute differentially private operations on
data without having to worry about the privacy implementation, which is handled
automatically by the API. It is built atop the [Tumult Core library](https://gitlab.com/tumult-labs/core).

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

If you have any questions/concerns, please [create an issue](https://gitlab.com/tumult-labs/analytics/-/issues) or reach out to us on [Slack](https://tmltdev.slack.com/join/shared_invite/zt-1bky0mh9v-vOB8azKAVoxmzJDUdWd5Wg#).

## Contributing

We are not yet accepting external contributions, but please let us know if you are interested in contributing [via Slack](https://tmltdev.slack.com/join/shared_invite/zt-1bky0mh9v-vOB8azKAVoxmzJDUdWd5Wg#).

See [CONTRIBUTING.md](CONTRIBUTING.md) for information about installing our development dependencies and running tests.

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

Copyright Tumult Labs 2023

The Tumult Platform source code is licensed under the Apache License, version 2.0 (Apache-2.0).
The Tumult Platform documentation is licensed under
Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0).
