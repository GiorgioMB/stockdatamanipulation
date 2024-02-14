"""Module containing test cases for the package."""
import unittest
from datafetcher import Fetcher as ftc 

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

if __name__ == '__main__':
    unittest.main()
