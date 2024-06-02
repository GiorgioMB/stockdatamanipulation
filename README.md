# StockDataManager
This Python library offers a comprehensive suite of tools for technical and fundamental analysis, along with options analysis capabilities. Utilizing the yfinance library, it facilitates access to historical stock data, financial statements, and key financial metrics from Yahoo Finance. It includes over 30 technical indicators and options analysis tools.

## Features
- **Technical Analysis (`IndicatorCalculator` Class)**: Offers over 30 technical indicators like Moving Averages, MACD, Bollinger Bands, RSI, Ichimoku Cloud, etc., for analyzing market trends and volatility.
- **Fundamental Analysis (`Fetcher` Class)**: Retrieve historical stock data, income statements, balance sheets, cash flows, and key financial ratios (e.g., P/E, ROE, current ratio).
- **Options Analysis (`Greeks` and `OptionPricing` Classes)**: tools to calculate options greeks and simulate option prices using various models, with enhanced methods to estimate risk-free rates and volatility.

## Installation

```bash
pip install stockdatamanager
```
## Quick Start
```python
from stockdatamanager import Fetcher, IndicatorCalculator
from stockdatamanager.options import Greeks, OptionPricing

# Fetching stock data and financial statements
fetcher = Fetcher(ticker='AAPL')
print(fetcher.get_pe_ratio())

# Applying technical analysis
indicators = IndicatorCalculator(dataframe=fetcher.df)
df_with_rsi = indicators.calculate_RSI()

# Calculating options Greeks
greeks = Greeks(ticker = 'AAPL', call = True, identification = 0)
delta = greeks.calculate_delta()

# Pricing an American-style option using the binomial tree method
option_pricing = OptionPricing(ticker='MSFT', call=False, american=True, risk_free_rate='13 weeks', identification=0, use_yfinance_volatility=True)
option_price = option_pricing.calculate_option_price(method='binomial', describe=False)
print(f"Option Price: {option_price}")
```
## Usage
### Fetching Data
```python
fetcher = Fetcher(ticker='AAPL')
income_statement = fetcher.get_income_statement()
```
### Calculating Technical Indicators
```python
transform = Transform(ticker='AAPL')
df_with_macd = transform.calculate_MACD()
```
### Option Analysis
Calculate the Delta of an option:
```python
greeks = Greeks(ticker='AAPL', call=True, identification='AAPL220121C00100000')
print(greeks.calculate_delta())
```
Simulate option pricing using the Crank-Nicholson method:
```python
option_pricing = OptionPricing(ticker='MSFT', call=False, american=True, risk_free_rate='13 weeks', identification='AAPL220121C00100000', use_yfinance_volatility=True)
option_price = option_pricing.calculate_option_price(method='crank-nicolson', describe=False)
print(f"Crank-Nicolson Method Option Price: {option_price}")
```
## Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request for improvements or new features.
***
## License
stockdatamanager is made available under the MIT License. See the LICENSE file for more details.
## Contacts
Email: giorgio.micaletto@studbocconi.it

