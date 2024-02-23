import unittest
import pandas as pd
from python.src.utils.realized_volatility_beta_calculators import *


class TestRVCalculators(unittest.TestCase):
    def test_realized_volatility_uses_right_price_type(self):
        toy_spy_prices = pd.read_csv("spy_data_tests.csv")
        with self.assertRaises(ValueError) as context:
            compute_realized_volatility_time_series_single_etf(
                etf_price_data=toy_spy_prices,
                price_type="NONE",
                rolling_window_days=5,
                price_frequency_hours=1,
            )
        self.assertEqual(str(context.exception), "Invalid Price Type! Cannot Calculate Realized Volatilities")

    def test_realized_volatility_window_is_too_large(self):
        toy_spy_prices = pd.read_csv("spy_data_tests.csv")
        with self.assertRaises(ValueError) as context:
            compute_realized_volatility_time_series_single_etf(
                etf_price_data=toy_spy_prices,
                price_type="Close",
                rolling_window_days=500,
                price_frequency_hours=1,
            )
        self.assertEqual(
            str(context.exception), "Cannot Compute Valid Rolling Realized Volatilities as the Window is too Large!"
        )

    def test_exact_realized_volatility_calculation(self):
        toy_spy_prices = pd.read_csv("spy_data_tests.csv")
        test_realized_volatility = compute_realized_volatility_time_series_single_etf(
            etf_price_data=toy_spy_prices,
            price_type="Close",
            rolling_window_days=8,
            price_frequency_hours=1,
        )
        expected_result = np.array([0.22882247, 0.24084355])
        self.assertTrue(np.allclose(test_realized_volatility.values, expected_result, atol=0.01))

    def test_rolling_betas_uses_right_price_type(self):
        toy_spy_prices = pd.read_csv("spy_data_tests.csv")
        toy_qqq_prices = pd.read_csv("qqq_data_tests.csv")
        with self.assertRaises(ValueError) as context:
            compute_rolling_betas_against_benchmark(
                target_etf_prices=toy_qqq_prices,
                benchmark_etf_prices=toy_spy_prices,
                price_type="NONE",
                rolling_window_days=5,
                price_frequency_hours=1,
            )
        self.assertEqual(str(context.exception), "Invalid Price Type! Cannot Calculate Rolling Betas")

    def test_rolling_betas_window_too_large(self):
        toy_spy_prices = pd.read_csv("spy_data_tests.csv")
        toy_qqq_prices = pd.read_csv("qqq_data_tests.csv")
        with self.assertRaises(ValueError) as context:
            compute_rolling_betas_against_benchmark(
                target_etf_prices=toy_qqq_prices,
                benchmark_etf_prices=toy_spy_prices,
                price_type="Close",
                rolling_window_days=50,
                price_frequency_hours=1,
            )
        self.assertEqual(str(context.exception), "Cannot Compute Valid Rolling Betas as the Window is too Large!")

    def test_exact_rolling_beta_calculation(self):
        toy_spy_prices = pd.read_csv("spy_data_tests.csv")
        toy_qqq_prices = pd.read_csv("qqq_data_tests.csv")

        test_rolling_beta = compute_rolling_betas_against_benchmark(
            target_etf_prices=toy_qqq_prices,
            benchmark_etf_prices=toy_spy_prices,
            price_type="Close",
            rolling_window_days=8,
            price_frequency_hours=1,
        )
        expected_result = np.array([0.993651, 0.999428])
        self.assertTrue(np.allclose(test_rolling_beta.values, expected_result, atol=0.01))

    def test_rolling_realized_correlation_uses_right_price_type(self):
        toy_spy_prices = pd.read_csv("spy_data_tests.csv")
        toy_qqq_prices = pd.read_csv("qqq_data_tests.csv")
        with self.assertRaises(ValueError) as context:
            compute_rolling_realized_correlation_against_benchmark(
                target_etf_prices=toy_qqq_prices,
                benchmark_etf_prices=toy_spy_prices,
                price_type="NONE",
                rolling_window_days=5,
                price_frequency_hours=1,
            )
        self.assertEqual(str(context.exception), "Invalid Price Type! Cannot Calculate Realized Volatilities")

    def test_rolling_realized_correlation_window_too_large(self):
        toy_spy_prices = pd.read_csv("spy_data_tests.csv")
        toy_qqq_prices = pd.read_csv("qqq_data_tests.csv")
        with self.assertRaises(ValueError) as context:
            compute_rolling_realized_correlation_against_benchmark(
                target_etf_prices=toy_qqq_prices,
                benchmark_etf_prices=toy_spy_prices,
                price_type="Close",
                rolling_window_days=50,
                price_frequency_hours=1,
            )
        self.assertEqual(
            str(context.exception), "Cannot Compute Valid Rolling Realized Volatilities as the Window is too Large!"
        )

    def test_exact_rolling_realized_correlation_calculation(self):
        toy_spy_prices = pd.read_csv("spy_data_tests.csv")
        toy_qqq_prices = pd.read_csv("qqq_data_tests.csv")

        rolling_realized_correlation = compute_rolling_realized_correlation_against_benchmark(
            target_etf_prices=toy_qqq_prices,
            benchmark_etf_prices=toy_spy_prices,
            price_type="Close",
            rolling_window_days=4,
            price_frequency_hours=1,
        )
        expected_result = np.array([0.95950583, 0.95767106, 0.951776])
        self.assertTrue(np.allclose(rolling_realized_correlation.values, expected_result, atol=0.01))
