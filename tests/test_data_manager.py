import unittest
import pandas as pd
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import load_ohlc_csv


class TestDataManager(unittest.TestCase):
    def setUp(self):
        # Create temporary CSV files with different formats
        self.test_dir = tempfile.mkdtemp()
        
        # Standard format
        self.csv1 = os.path.join(self.test_dir, 'test1.csv')
        with open(self.csv1, 'w') as f:
            f.write("""2023-01-01 00:00:00,100,105,95,102,1000
2023-01-01 01:00:00,102,108,100,105,1200
2023-01-01 02:00:00,105,110,103,107,1300""")

        # Different separator
        self.csv2 = os.path.join(self.test_dir, 'test2.csv')
        with open(self.csv2, 'w') as f:
            f.write("""2023-01-01 00:00:00;100;105;95;102;1000
2023-01-01 01:00:00;102;108;100;105;1200
2023-01-01 02:00:00;105;110;103;107;1300""")

        # Missing high/low columns (should fail)
        self.csv3 = os.path.join(self.test_dir, 'test3.csv')
        with open(self.csv3, 'w') as f:
            f.write("""2023-01-01 00:00:00,100,102
2023-01-01 01:00:00,102,105""")
        
        # Create directory and file for the missing test case
        data_dir = os.path.join(self.test_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        self.csv4 = os.path.join(data_dir, 'EURUSD_1H_2009-2025.csv')
        with open(self.csv4, 'w') as f:
            f.write("2023-01-01 00:00:00,100")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_load_csv_success(self):
        """Test loading valid CSV files."""
        
        # Test standard format
        df1 = load_ohlc_csv(self.csv1)
        print(df1)
        self.assertEqual(len(df1), 3)
        self.assertTrue(all(col in df1.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        self.assertIsInstance(df1.index, pd.DatetimeIndex)
        
        # Test with different separator
        df2 = load_ohlc_csv(self.csv2)  
        self.assertEqual(len(df2), 3)
        self.assertTrue(all(col in df2.columns for col in ['open', 'high', 'low', 'close', 'volume']))

    def test_load_csv_missing_columns(self):
        """Test that missing OHLC columns raise error."""
        
        with self.assertRaises(ValueError):
            load_ohlc_csv(self.csv3)
        
        # Test with file that has data but missing columns
        with self.assertRaises(ValueError):
            load_ohlc_csv(self.csv4)


    def test_column_standardization(self):
        """Test that columns are properly standardized."""
        df = load_ohlc_csv(self.csv1)
        
        # Should have standard names
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                self.assertIn(col, df.columns)


if __name__ == '__main__':
    unittest.main()