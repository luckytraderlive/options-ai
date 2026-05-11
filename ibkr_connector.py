"""
IBKR connector skeleton for Options AI v4.

Important:
- This is for LOCAL use or a VPS where IB Gateway / TWS is running.
- Streamlit Cloud cannot connect to your local IB Gateway.
- Install: pip install ib_insync
- TWS / IB Gateway settings:
  Enable ActiveX and Socket Clients
  Paper trading port usually: 7497
  Live trading port usually: 7496

This module is intentionally separate from the Yahoo engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from ib_insync import IB, Stock, Option, util
except Exception:
    IB = None
    Stock = None
    Option = None
    util = None


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 7


class IBKRConnector:
    def __init__(self, config: Optional[IBKRConfig] = None):
        if IB is None:
            raise ImportError("ib_insync is not installed. Run: pip install ib_insync")

        self.config = config or IBKRConfig()
        self.ib = IB()

    def connect(self):
        self.ib.connect(
            self.config.host,
            self.config.port,
            clientId=self.config.client_id
        )
        return self.ib.isConnected()

    def disconnect(self):
        self.ib.disconnect()

    def get_stock_price(self, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        contract = Stock(symbol, exchange, currency)
        self.ib.qualifyContracts(contract)

        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(2)

        price = ticker.marketPrice()

        if price is None or price != price:
            price = ticker.close

        return price

    def get_option_contract(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
        currency: str = "USD"
    ):
        """
        expiry format: YYYYMMDD
        right: C or P
        """
        contract = Option(symbol, expiry, strike, right, exchange, currency=currency)
        self.ib.qualifyContracts(contract)
        return contract

    def get_option_market_data(self, symbol: str, expiry: str, strike: float, right: str):
        contract = self.get_option_contract(symbol, expiry, strike, right)
        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(2)

        return {
            "bid": ticker.bid,
            "ask": ticker.ask,
            "last": ticker.last,
            "mid": (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else None,
            "modelGreeks": ticker.modelGreeks,
            "contract": contract,
        }

    def place_paper_order_example(self):
        """
        Intentionally not implemented.
        Add live/paper order execution only after adding confirmations,
        risk limits, and trade logs.
        """
        raise NotImplementedError("Order execution is disabled by default for safety.")