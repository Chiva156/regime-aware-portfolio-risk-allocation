from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

DEFAULT_TICKERS: tuple[str, ...] = ("SPY", "TLT", "IEF", "GLD", "DBC", "UUP", "SHY")


@dataclass(frozen=True)
class DataConfig:
    """Configuration for reproducible public-data loading."""

    tickers: tuple[str, ...] = DEFAULT_TICKERS
    start: str = "2007-01-01"
    end: str | None = None
    source: str = "yfinance"


def _as_list(values: Iterable[str]) -> list[str]:
    return [str(v).upper() for v in values]


def _stooq_symbol(ticker: str) -> str:
    """Return Stooq US ETF symbol.

    Stooq expects US tickers as e.g. 'spy.us'.
    """

    symbol = ticker.lower()
    if "." not in symbol:
        symbol = f"{symbol}.us"
    return symbol


def fetch_stooq_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = "2007-01-01",
    end: str | None = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Fetch daily prices from Stooq through pandas-datareader.

    Parameters
    ----------
    tickers:
        ETF or stock tickers. US symbols are automatically mapped to the Stooq
        '.us' suffix.
    start, end:
        Date bounds understood by pandas.
    price_col:
        Usually 'Close'. Stooq does not provide adjusted close in the same way
        as Yahoo; for a research demo this is acceptable but should be documented.

    Returns
    -------
    pandas.DataFrame
        Wide price panel with uppercase tickers as columns.
    """

    from pandas_datareader import data as pdr

    frames: list[pd.Series] = []
    failures: dict[str, str] = {}
    for ticker in _as_list(tickers):
        try:
            raw = pdr.DataReader(_stooq_symbol(ticker), "stooq", start=start, end=end)
            if raw.empty:
                failures[ticker] = "empty response"
                continue
            series = raw.sort_index()[price_col].rename(ticker)
            frames.append(series)
        except Exception as exc:  # pragma: no cover - network dependent
            failures[ticker] = str(exc)

    if not frames:
        message = "No Stooq prices could be downloaded."
        if failures:
            message += f" Failures: {failures}"
        raise RuntimeError(message)

    prices = pd.concat(frames, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(how="all")
    if failures:
        prices.attrs["download_failures"] = failures
    return prices


def fetch_yfinance_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = "2007-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch prices from Yahoo Finance through yfinance.

    This is convenient for research but users should check Yahoo/yfinance terms
    before relying on it in production or redistributing data.
    """

    import yfinance as yf

    tickers_list = _as_list(tickers)
    raw = yf.download(
        tickers=" ".join(tickers_list),
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("No yfinance prices could be downloaded.")

    if isinstance(raw.columns, pd.MultiIndex):
        col = "Close" if "Close" in raw.columns.get_level_values(0) else "Adj Close"
        prices = raw[col].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers_list[0]})

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna(how="all")
    return prices


def load_price_panel(config: DataConfig | None = None, **overrides: object) -> pd.DataFrame:
    """Load a public price panel using a :class:`DataConfig`.

    Examples
    --------
    >>> prices = load_price_panel(start="2015-01-01", source="stooq")
    """

    base = config or DataConfig()
    params = {**base.__dict__, **overrides}
    source = params.pop("source")
    if source == "stooq":
        return fetch_stooq_prices(**params)
    if source == "yfinance":
        return fetch_yfinance_prices(**params)
    raise ValueError(f"Unknown source: {source}")


def compute_returns(
    prices: pd.DataFrame,
    method: Literal["log", "simple"] = "log",
    dropna: bool = True,
) -> pd.DataFrame:
    """Compute daily returns from a price panel."""

    prices = prices.sort_index().astype(float)
    if method == "log":
        out = np.log(prices).diff()
    elif method == "simple":
        out = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return out.dropna(how="all") if dropna else out


def fetch_fred_series(
    series: dict[str, str] | Iterable[str],
    start: str = "2007-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch FRED series through pandas-datareader.

    Parameters
    ----------
    series:
        Either an iterable of FRED series ids or a mapping {output_name: fred_id}.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date.
    """

    from pandas_datareader import data as pdr

    if isinstance(series, dict):
        mapping = series
    else:
        mapping = {s: s for s in series}

    frames: list[pd.Series] = []
    for output_name, fred_id in mapping.items():
        raw = pdr.DataReader(fred_id, "fred", start=start, end=end)
        frames.append(raw.iloc[:, 0].rename(output_name))

    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def align_to_returns(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    fill_method: Literal["ffill", "none"] = "ffill",
) -> pd.DataFrame:
    """Align lower-frequency or irregular features to a return calendar."""

    features = features.sort_index()
    aligned = features.reindex(returns.index)
    if fill_method == "ffill":
        aligned = aligned.ffill()
    elif fill_method != "none":
        raise ValueError("fill_method must be 'ffill' or 'none'")
    return aligned
