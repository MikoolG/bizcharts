"""Multi-source cryptocurrency price provider with rotation and fallback."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    """Price data for a single coin."""

    coin: str
    price_usd: float
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    volume_24h: float | None = None
    market_cap: float | None = None


@dataclass
class PriceFetchResult:
    """Result of a price fetch operation."""

    prices: dict[str, PriceData]
    source: str
    success: bool
    error: str | None = None


class PriceSource(ABC):
    """Abstract base class for price data sources."""

    name: str
    api_url: str

    @abstractmethod
    async def fetch_prices(
        self, coins: list[str], client: httpx.AsyncClient
    ) -> PriceFetchResult:
        """Fetch prices for the given coins."""
        pass


class CoinGeckoSource(PriceSource):
    """CoinGecko price source - aggregated from 900+ exchanges."""

    name = "coingecko"

    def __init__(self, config: dict[str, Any]):
        self.api_url = config.get("api_url", "https://api.coingecko.com/api/v3")
        self.coin_ids = config.get("coin_ids", {})

    async def fetch_prices(
        self, coins: list[str], client: httpx.AsyncClient
    ) -> PriceFetchResult:
        """Fetch prices from CoinGecko API."""
        try:
            # Map coin names to CoinGecko IDs
            ids = [self.coin_ids.get(c, c) for c in coins]
            ids_param = ",".join(ids)

            response = await client.get(
                f"{self.api_url}/simple/price",
                params={
                    "ids": ids_param,
                    "vs_currencies": "usd",
                    "include_24hr_vol": "true",
                    "include_market_cap": "true",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            prices = {}
            for coin in coins:
                coin_id = self.coin_ids.get(coin, coin)
                if coin_id in data:
                    prices[coin] = PriceData(
                        coin=coin,
                        price_usd=data[coin_id]["usd"],
                        source=self.name,
                        volume_24h=data[coin_id].get("usd_24h_vol"),
                        market_cap=data[coin_id].get("usd_market_cap"),
                    )

            return PriceFetchResult(prices=prices, source=self.name, success=True)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("CoinGecko rate limit hit")
            return PriceFetchResult(
                prices={}, source=self.name, success=False, error=str(e)
            )
        except Exception as e:
            logger.error(f"CoinGecko fetch error: {e}")
            return PriceFetchResult(
                prices={}, source=self.name, success=False, error=str(e)
            )


class BinanceSource(PriceSource):
    """Binance price source - real-time exchange data, no rate limits."""

    name = "binance"

    def __init__(self, config: dict[str, Any]):
        self.api_url = config.get("api_url", "https://data-api.binance.vision/api/v3")
        self.symbols = config.get("symbols", {})

    async def fetch_prices(
        self, coins: list[str], client: httpx.AsyncClient
    ) -> PriceFetchResult:
        """Fetch prices from Binance API."""
        try:
            prices = {}
            for coin in coins:
                symbol = self.symbols.get(coin)
                if not symbol:
                    continue

                response = await client.get(
                    f"{self.api_url}/ticker/price",
                    params={"symbol": symbol},
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

                prices[coin] = PriceData(
                    coin=coin,
                    price_usd=float(data["price"]),
                    source=self.name,
                )

            # Fetch 24h volume separately if needed
            for coin in coins:
                symbol = self.symbols.get(coin)
                if symbol and coin in prices:
                    try:
                        vol_response = await client.get(
                            f"{self.api_url}/ticker/24hr",
                            params={"symbol": symbol},
                            timeout=10.0,
                        )
                        vol_response.raise_for_status()
                        vol_data = vol_response.json()
                        prices[coin].volume_24h = float(vol_data.get("quoteVolume", 0))
                    except Exception:
                        pass  # Volume is optional

            return PriceFetchResult(prices=prices, source=self.name, success=True)

        except Exception as e:
            logger.error(f"Binance fetch error: {e}")
            return PriceFetchResult(
                prices={}, source=self.name, success=False, error=str(e)
            )


class CoinMarketCapSource(PriceSource):
    """CoinMarketCap price source - requires free API key."""

    name = "coinmarketcap"

    def __init__(self, config: dict[str, Any]):
        self.api_url = config.get("api_url", "https://pro-api.coinmarketcap.com/v1")
        self.slugs = config.get("slugs", {})
        self.api_key_env = config.get("api_key_env", "COINMARKETCAP_API_KEY")
        self._api_key = os.environ.get(self.api_key_env)

    async def fetch_prices(
        self, coins: list[str], client: httpx.AsyncClient
    ) -> PriceFetchResult:
        """Fetch prices from CoinMarketCap API."""
        if not self._api_key:
            return PriceFetchResult(
                prices={},
                source=self.name,
                success=False,
                error=f"API key not set in {self.api_key_env}",
            )

        try:
            # Map coin names to CMC slugs
            slugs = [self.slugs.get(c, c) for c in coins]
            slug_param = ",".join(slugs)

            response = await client.get(
                f"{self.api_url}/cryptocurrency/quotes/latest",
                params={"slug": slug_param, "convert": "USD"},
                headers={"X-CMC_PRO_API_KEY": self._api_key},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            prices = {}
            if "data" in data:
                # CMC returns data keyed by ID, need to map back
                slug_to_coin = {v: k for k, v in self.slugs.items()}
                for _id, coin_data in data["data"].items():
                    slug = coin_data.get("slug")
                    coin = slug_to_coin.get(slug, slug)
                    if coin in coins:
                        quote = coin_data.get("quote", {}).get("USD", {})
                        prices[coin] = PriceData(
                            coin=coin,
                            price_usd=quote.get("price", 0),
                            source=self.name,
                            volume_24h=quote.get("volume_24h"),
                            market_cap=quote.get("market_cap"),
                        )

            return PriceFetchResult(prices=prices, source=self.name, success=True)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("CoinMarketCap rate limit hit")
            return PriceFetchResult(
                prices={}, source=self.name, success=False, error=str(e)
            )
        except Exception as e:
            logger.error(f"CoinMarketCap fetch error: {e}")
            return PriceFetchResult(
                prices={}, source=self.name, success=False, error=str(e)
            )


class MultiSourcePriceProvider:
    """
    Multi-source price provider with rotation and fallback.

    Rotates through configured sources to distribute API calls
    and stay within free tier limits while maintaining faster polling.
    """

    SOURCE_CLASSES = {
        "coingecko": CoinGeckoSource,
        "binance": BinanceSource,
        "coinmarketcap": CoinMarketCapSource,
    }

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the multi-source provider.

        Args:
            config: Configuration dict from settings.toml [price_data] section
        """
        self.coins = config.get("tracked_coins", ["bitcoin", "ethereum", "solana"])
        self.enable_rotation = config.get("enable_rotation", True)
        self.rotation_order = config.get(
            "rotation_order", ["coingecko", "binance", "coinmarketcap"]
        )
        self.fallback_on_error = config.get("fallback_on_error", True)
        self.deviation_threshold = config.get("price_deviation_threshold", 2.0)

        # Initialize sources
        sources_config = config.get("sources", {})
        self.sources: dict[str, PriceSource] = {}
        for name in self.rotation_order:
            if name in sources_config and sources_config[name].get("enabled", True):
                source_class = self.SOURCE_CLASSES.get(name)
                if source_class:
                    self.sources[name] = source_class(sources_config[name])

        # Rotation state
        self._current_index = 0
        self._last_prices: dict[str, PriceData] = {}

    def _get_current_source(self) -> str:
        """Get the current source name based on rotation index."""
        if not self.rotation_order:
            return "coingecko"
        return self.rotation_order[self._current_index % len(self.rotation_order)]

    def _rotate(self) -> None:
        """Rotate to the next source."""
        if self.enable_rotation:
            self._current_index = (self._current_index + 1) % len(self.rotation_order)

    async def fetch_prices(self) -> PriceFetchResult:
        """
        Fetch prices using rotation and fallback strategy.

        Returns:
            PriceFetchResult with prices from the successful source
        """
        async with httpx.AsyncClient() as client:
            attempts = len(self.sources) if self.fallback_on_error else 1

            for _attempt in range(attempts):
                source_name = self._get_current_source()
                source = self.sources.get(source_name)

                if not source:
                    self._rotate()
                    continue

                logger.debug(f"Fetching prices from {source_name}")
                result = await source.fetch_prices(self.coins, client)

                if result.success and result.prices:
                    # Check for price deviations if we have previous data
                    self._check_price_deviation(result.prices)
                    self._last_prices = result.prices
                    self._rotate()  # Rotate for next fetch
                    return result

                logger.warning(
                    f"Price fetch from {source_name} failed: {result.error}"
                )
                self._rotate()  # Try next source

            return PriceFetchResult(
                prices=self._last_prices,  # Return cached if all fail
                source="cached",
                success=False,
                error="All price sources failed",
            )

    def _check_price_deviation(self, new_prices: dict[str, PriceData]) -> None:
        """Log warning if prices deviate significantly from last fetch."""
        for coin, new_data in new_prices.items():
            if coin in self._last_prices:
                old_price = self._last_prices[coin].price_usd
                new_price = new_data.price_usd
                if old_price > 0:
                    deviation = abs(new_price - old_price) / old_price * 100
                    if deviation > self.deviation_threshold:
                        logger.warning(
                            f"Price deviation for {coin}: {deviation:.2f}% "
                            f"({old_price:.2f} -> {new_price:.2f})"
                        )

    def get_source_status(self) -> dict[str, bool]:
        """Get enabled status of each source."""
        return {name: name in self.sources for name in self.rotation_order}


async def main() -> None:
    """Test the multi-source price provider."""
    import tomli

    # Load config
    with open("../config/settings.toml", "rb") as f:
        config = tomli.load(f)

    provider = MultiSourcePriceProvider(config.get("price_data", {}))

    print("Source status:", provider.get_source_status())
    print("\nFetching prices (will rotate through sources)...")

    for i in range(3):
        result = await provider.fetch_prices()
        print(f"\nFetch {i + 1} from {result.source}:")
        if result.success:
            for coin, data in result.prices.items():
                print(f"  {coin}: ${data.price_usd:,.2f}")
        else:
            print(f"  Error: {result.error}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
