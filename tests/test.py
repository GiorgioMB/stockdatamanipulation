"""Module containing test cases for the package."""
import unittest
from datafetcher import Fetcher as ftc 
import pandas as pd
from indicators import IndicatorCalculator

class TestDataFetcher(unittest.TestCase):
    """Module containing test cases for the package."""
    def test_historical_data_retrieval(self):
        """Test that historical data is retrieved without errors."""
        df = ftc("AAPL")
        self.assertIsNotNone(df.df, "Historical data should not be None")

    def test_invalid_ticker_handling(self):
        """Test how an invalid ticker is handled."""
        with self.assertRaises(AssertionError):
            ftc("INVALID_TICKER")
class TestIndicatorCalculator(unittest.TestCase):
    def setUp(self):
        self.sample_df = ftc('AAPL').df

    def test_initialization_with_ticker(self):
        calculator = IndicatorCalculator(ticker='AAPL')
        self.assertIsNotNone(calculator.df)

    def test_initialization_with_dataframe(self):
        calculator = IndicatorCalculator(dataframe=self.sample_df)
        self.assertEqual(len(calculator.df), 3)

    def test_calculate_rsi(self):
        calculator = IndicatorCalculator(dataframe=self.sample_df)
        calculator.calculate_RSI()
        self.assertIn('RSI', calculator.df.columns)

    def test_calculate_macd(self):
        calculator = IndicatorCalculator(dataframe=self.sample_df)
        calculator.calculate_MACD()
        self.assertIn('MACD', calculator.df.columns)
        self.assertIn('Signal', calculator.df.columns)

    def test_invalid_ticker(self):
        with self.assertRaises(ValueError):
            IndicatorCalculator(ticker='INVALID')

    def test_missing_columns(self):
        incomplete_df = pd.DataFrame({'Close': [100, 101, 102]})
        calculator = IndicatorCalculator(dataframe=incomplete_df)
        with self.assertRaises(KeyError):
            calculator.calculate_on_balance_volume()

    def test_verbosity(self):
        calculator = IndicatorCalculator(dataframe=self.sample_df, verbose=2)
        with self.assertLogs(level='INFO') as log:
            calculator.calculate_RSI()
            self.assertIn('Calculating RSI', log.output[0])

if __name__ == '__main__':
    unittest.main()
