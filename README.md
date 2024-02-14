
# DataFetcher

Fetcher is a Python package designed for retrieving and preprocessing financial information. It gathers historical data for specified companies. The package leverages `yfinance` for financial data, `pandas` for data manipulation, and `requests` for web requests.

## Installation

Install Fetcher using pip:

```bash
pip install stockdatafetcher
```

## Quick Start

To use DataFetcher, import the `Fetcher` class and initialize it with the ticker symbol of the company you're interested in. For example:

```python
from datafetcher import Fetcher

# Initialize with a company ticker
df = Fetcher("AAPL")

# Fetch historical data
historical_data, actual_ticker = df.get_historical_data("AAPL")
print(historical_data)
```

## Dependencies

Fetcher requires the following Python packages:

- pandas
- yfinance
- requests

These dependencies will be automatically installed when you install Fetcher using pip.

## Contributing

Contributions to DataFetcher are welcome! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.

## Reporting Issues

If you encounter any problems or have feedback, please open an issue on the project's GitHub page. Your input is valuable in making DataFetcher a better tool for everyone.

## License

DataFetcher is released under the [MIT License](LICENSE). See the LICENSE file for more details.
