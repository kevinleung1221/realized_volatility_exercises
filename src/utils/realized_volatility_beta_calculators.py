import pandas as pd
import numpy as np


def compute_realized_volatility_time_series_single_etf(
    etf_price_data: pd.DataFrame,
    price_type: str,
    rolling_window_days: int,
    price_frequency_hours: float,
):
    """
    Function to compute N-day Realized Volatility of an ETF using a specified frequency of price data

    :param etf_price_data: dataframe of daily ETF prices grabbed from yfinance (Yahoo Finance)
    :param price_type: one of 'Open', 'High', 'Low', 'Close'
    :param rolling_window_days: Rolling number of days to calculate realized volatility
    :param price_frequency_hours: granularity of ETF price data

    :return:
        Numpy Series for the realized volatility of an input ETF
    :raises:
        ValueError if the inputted price_type does not satisfy Yahoo Finance's prices
        ValueError if the rolling_window is too large to compute valid RVs
    """
    if price_type not in ["Open", "High", "Low", "Close"]:
        raise ValueError("Invalid Price Type! Cannot Calculate Realized Volatilities")

    price_frequency_scalar = round(6.5 / price_frequency_hours)  # num of hours in trading day / granularity of prices
    window = rolling_window_days * price_frequency_scalar

    if window > len(etf_price_data):
        raise ValueError("Cannot Compute Valid Rolling Realized Volatilities as the Window is too Large!")

    etf_price_data = etf_price_data.dropna()
    etf_price_data["log_returns"] = np.log(etf_price_data[price_type] / etf_price_data[price_type].shift(1))
    realized_volatility = np.sqrt(
        252 * price_frequency_scalar * np.square(etf_price_data["log_returns"]).rolling(window=window).mean()
    )

    realized_volatility = realized_volatility.dropna()
    return realized_volatility


def compute_rolling_betas_against_benchmark(
    target_etf_prices: pd.DataFrame,
    benchmark_etf_prices: pd.DataFrame,
    price_type: str,
    rolling_window_days: int,
    price_frequency_hours: float,
):
    """
    Function to compute rolling betas of an ETF v. a benchmark

    :param target_etf_prices: target ETF that we are interested in rolling betas for
    :param benchmark_etf_prices: benchmark ETF (e.g. SPY) to calculate a target ETF's rolling beta against
    :param price_type: one of 'Open', 'High', 'Low', 'Close'
    :param rolling_window_days: Rolling number of days to calculate rolling betas relative to a benchmark ETF
    :param price_frequency_hours: granularity of ETF price data
    :return:
        Numpy Series for the rolling beta between a target ETF and a benchmark ETF using np.linalg.lstsq
    :raises:
        ValueError if the inputted price_type does not satisfy Yahoo Finance's prices
        ValueError if the rolling_window is too large to compute valid betas
    """
    if price_type not in ["Open", "High", "Low", "Close"]:
        raise ValueError("Invalid Price Type! Cannot Calculate Rolling Betas")

    price_frequency_scalar = round(6.5 / price_frequency_hours)  # num of hours in trading day / granularity of prices
    window = rolling_window_days * price_frequency_scalar
    if window > len(benchmark_etf_prices) or window > len(target_etf_prices):
        raise ValueError("Cannot Compute Valid Rolling Betas as the Window is too Large!")

    target_etf_prices["target_log_returns"] = np.log(
        target_etf_prices[price_type] / target_etf_prices[price_type].shift(1)
    )
    benchmark_etf_prices["benchmark_log_returns"] = np.log(
        benchmark_etf_prices[price_type] / benchmark_etf_prices[price_type].shift(1)
    )

    merged_returns = pd.concat(
        [target_etf_prices["target_log_returns"], benchmark_etf_prices["benchmark_log_returns"]],
        axis=1,
    )
    merged_returns = merged_returns.dropna()

    rolling_betas = pd.Series(index=merged_returns.index)

    for i in range(len(merged_returns) - window + 1):
        rolling_benchmark_returns_with_intercept = np.vstack(
            [merged_returns["benchmark_log_returns"][i : (i + window)], np.ones(window)]
        ).T
        rolling_target_returns = merged_returns["target_log_returns"][i : (i + window)]
        beta, intercept = np.linalg.lstsq(rolling_benchmark_returns_with_intercept, rolling_target_returns)[0]
        rolling_betas.loc[rolling_target_returns.index.max()] = beta

    rolling_betas = rolling_betas.dropna()
    return rolling_betas


def compute_rolling_realized_correlation_against_benchmark(
    target_etf_prices: pd.DataFrame,
    benchmark_etf_prices: pd.DataFrame,
    price_type: str,
    rolling_window_days: int,
    price_frequency_hours: float,
):
    """
    Function to compute realized volatility time series for 2 ETFs, then compute a rolling correlation between them

    :param target_etf_prices: target ETF that we are interested in rolling betas for
    :param benchmark_etf_prices: benchmark ETF (e.g. SPY) to calculate a target ETF's rolling beta against
    :param price_type: one of 'Open', 'High', 'Low', 'Close'
    :param rolling_window_days: Rolling number of days to calculate rolling realized correlations with a benchmark ETF
    :param price_frequency_hours: granularity of ETF price data
    :return:
        Pandas Series of rolling correlations between two ETFs
    """
    target_realized_volatility = compute_realized_volatility_time_series_single_etf(
        etf_price_data=target_etf_prices,
        price_type=price_type,
        rolling_window_days=rolling_window_days,
        price_frequency_hours=price_frequency_hours,
    )

    benchmark_realized_volatility = compute_realized_volatility_time_series_single_etf(
        etf_price_data=benchmark_etf_prices,
        price_type=price_type,
        rolling_window_days=rolling_window_days,
        price_frequency_hours=price_frequency_hours,
    )

    price_frequency_scalar = round(6.5 / price_frequency_hours)  # num of hours in trading day / granularity of prices
    window = rolling_window_days * price_frequency_scalar

    rolling_correlation = target_realized_volatility.rolling(window=window).corr(
        benchmark_realized_volatility, pairwise=True,
    )
    rolling_correlation = rolling_correlation.dropna()
    return rolling_correlation
