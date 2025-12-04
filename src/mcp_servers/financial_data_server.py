"""
MCP сервер для финансовых данных
"""
import asyncio
from typing import Dict, Any, List
from mcp import Server
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinancialDataServer:
    """MCP сервер для предоставления финансовых данных"""

    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.server = Server("financial-data-server")
        self._register_tools()

    def _register_tools(self):
        """Регистрация инструментов MCP"""

        @self.server.tool()
        async def get_ohlcv(
                symbol: str,
                interval: str = "1d",
                period: str = "1y"
        ) -> Dict[str, Any]:
            """
            Получение ценовых данных OHLCV

            Args:
                symbol: Тикер акции (например, AAPL)
                interval: Интервал данных (1d, 1h, 1m)
                period: Период данных (1d, 1mo, 1y, max)
            """
            try:
                logger.info(f"Получение данных OHLCV для {symbol}")

                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

                # Преобразование в список словарей
                data = []
                for idx, row in df.iterrows():
                    data.append({
                        "date": idx.isoformat(),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"])
                    })

                return {
                    "status": "success",
                    "symbol": symbol,
                    "interval": interval,
                    "period": period,
                    "data": data,
                    "count": len(data),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Ошибка получения данных OHLCV: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        @self.server.tool()
        async def get_fundamentals(symbol: str) -> Dict[str, Any]:
            """Получение фундаментальных показателей"""
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                fundamentals = {
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "avg_volume": info.get("averageVolume"),
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "eps": info.get("trailingEps"),
                    "forward_eps": info.get("forwardEps")
                }

                # Фильтрация None значений
                fundamentals = {k: v for k, v in fundamentals.items() if v is not None}

                return {
                    "status": "success",
                    "symbol": symbol,
                    "fundamentals": fundamentals,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Ошибка получения фундаментальных данных: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

        @self.server.tool()
        async def get_quote(symbol: str) -> Dict[str, Any]:
            """Получение текущей котировки"""
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                quote = {
                    "price": info.get("currentPrice"),
                    "change": info.get("regularMarketChange"),
                    "change_percent": info.get("regularMarketChangePercent"),
                    "previous_close": info.get("regularMarketPreviousClose"),
                    "open": info.get("regularMarketOpen"),
                    "day_high": info.get("regularMarketDayHigh"),
                    "day_low": info.get("regularMarketDayLow"),
                    "volume": info.get("regularMarketVolume")
                }

                return {
                    "status": "success",
                    "symbol": symbol,
                    "quote": quote,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Ошибка получения котировки: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

    async def start(self):
        """Запуск MCP сервера"""
        logger.info(f"Запуск FinancialDataServer на {self.host}:{self.port}")

        try:
            # Здесь будет реализация запуска сервера
            # В реальной реализации используется asyncio и сетевой код
            await asyncio.sleep(3600)  # Держим сервер активным
        except KeyboardInterrupt:
            logger.info("Остановка FinancialDataServer")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Получение списка доступных инструментов"""
        return [
            {
                "name": "get_ohlcv",
                "description": "Получение исторических ценовых данных OHLCV",
                "parameters": {
                    "symbol": {"type": "string", "required": True},
                    "interval": {"type": "string", "default": "1d"},
                    "period": {"type": "string", "default": "1y"}
                }
            },
            {
                "name": "get_fundamentals",
                "description": "Получение фундаментальных показателей компании",
                "parameters": {
                    "symbol": {"type": "string", "required": True}
                }
            },
            {
                "name": "get_quote",
                "description": "Получение текущей котировки акции",
                "parameters": {
                    "symbol": {"type": "string", "required": True}
                }
            }
        ]