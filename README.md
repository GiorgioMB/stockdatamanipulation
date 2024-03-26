# StockDataManager
This Python library offers a combination of technical analysis tools and fundamental data retrieval functionalities, designed to support investors, researchers, and enthusiasts in the financial markets. Utilizing the yfinance library, it provides easy access to historical stock data, financial statements, and key financial metrics from Yahoo Finance, alongside a suite of technical indicators for market analysis.
## Features
- **Technical Analysis (`IndicatorCalculator` Class)**: Over 30 technical indicators, including Moving Averages, MACD, Bollinger Bands, RSI, Ichimoku Cloud, and more, to dissect stock market trends and volatility.
- **Fundamental Analysis (`Fetcher` Class)**: Fetch historical stock data, income statements, balance sheets, cash flows, and key financial ratios (e.g., P/E, ROE, current ratio) for in-depth fundamental analysis.

## Installation

```bash
pip install stockdatamanager
```
## Quick Start
```python
from stockdatamanager import Fetcher, IndicatorCalculator

# Fetch historical data and financial statements for analysis
fetcher = Fetcher(ticker='AAPL')
print(fetcher.get_pe_ratio())

# Apply technical analysis on fetched data
indicators = IndicatorCalculator(dataframe=fetcher.df)
df_with_rsi = indicators.calculate_RSI()
```
## Usage
### Fetching Data
```python
fetcher = Fetcher(ticker='AAPL')
income_statement = fetcher.get_income_statement()
```
### Calculating Technical Indicators
```python
transform = Transform(dataframe=fetcher.df)
df_with_macd = transform.calculate_MACD()
```
## Contributions
Contributions to stockdatamanager are welcome! If you have suggestions for improvement or new features, feel free to open an issue or submit a pull request.
***
## License
stockdatamanager is made available under the MIT License. See the LICENSE file for more details.
## Contacts
Email: giorgio.micaletto@studbocconi.it

