"""
Агент анализа данных
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
import talib

from src.agents.base_agent import BaseFinancialAgent
from src.chains.technical_analysis_chain import TechnicalAnalysisChain
from src.utils.financial_indicators import calculate_technical_indicators
from src.memory.short_term_memory import ShortTermMemory

from config.settings import settings


class AnalysisAgent(BaseFinancialAgent):
    """Агент технического и фундаментального анализа"""

    def __init__(self, llm):
        tools = self._initialize_tools()

        super().__init__(
            role="Chief Investment Strategist",
            goal="Analyze financial data to generate actionable investment insights",
            backstory=(
                "Former hedge fund portfolio manager with 15 years of experience "
                "in quantitative analysis. You combine deep technical expertise "
                "with macroeconomic understanding to identify market opportunities "
                "and risks. Known for your rigorous, data-driven approach and "
                "uncanny ability to spot trends before they become mainstream."
            ),
            tools=tools,
            verbose=True
        )

        self.llm = llm
        self.technical_chain = TechnicalAnalysisChain(llm)
        self.memory = ShortTermMemory()

    def _initialize_tools(self) -> List:
        """Инициализация инструментов анализа"""
        # Здесь будут инструменты для технического анализа
        return []

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение анализа данных

        Args:
            task_input: {
                "data_key": "memory_key_from_data_agent",
                "analysis_type": "technical|fundamental|sentiment|full",
                "parameters": {...}
            }
        """
        try:
            logger.info(f"AnalysisAgent: Начинаю анализ данных")

            # 1. Получение данных из памяти
            data = self.memory.retrieve(task_input["data_key"])
            if not data:
                raise ValueError("Данные не найдены в памяти")

            # 2. Технический анализ
            technical_results = self._perform_technical_analysis(data)

            # 3. Анализ настроений
            sentiment_results = self._perform_sentiment_analysis(data)

            # 4. LLM анализ и выводы
            insights = self._generate_insights(
                technical_results,
                sentiment_results,
                task_input
            )

            # 5. Оценка рисков
            risk_assessment = self._assess_risks(insights, technical_results)

            # 6. Подготовка результата
            result = {
                "status": "success",
                "ticker": data.get("ticker", "unknown"),
                "technical_analysis": technical_results,
                "sentiment_analysis": sentiment_results,
                "insights": insights,
                "risk_assessment": risk_assessment,
                "recommendation": self._generate_recommendation(insights, risk_assessment),
                "confidence_score": self._calculate_confidence(insights),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"AnalysisAgent: Анализ завершен")
            return result

        except Exception as e:
            logger.error(f"AnalysisAgent: Ошибка при анализе: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _perform_technical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение технического анализа"""
        results = {
            "indicators": {},
            "patterns": [],
            "signals": [],
            "summary": ""
        }

        if "prices" in data and data["prices"]:
            prices_df = pd.DataFrame(data["prices"])

            # Расчет индикаторов
            if len(prices_df) >= 20:  # Минимальное количество точек для анализа
                close_prices = prices_df["close"].values

                # RSI
                rsi = talib.RSI(close_prices, timeperiod=14)
                results["indicators"]["rsi"] = {
                    "current": float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
                    "values": [float(x) for x in rsi[-50:] if not np.isnan(x)],
                    "overbought": rsi[-1] > 70 if not np.isnan(rsi[-1]) else False,
                    "oversold": rsi[-1] < 30 if not np.isnan(rsi[-1]) else False
                }

                # MACD
                macd, signal, hist = talib.MACD(close_prices)
                results["indicators"]["macd"] = {
                    "macd": float(macd[-1]) if not np.isnan(macd[-1]) else None,
                    "signal": float(signal[-1]) if not np.isnan(signal[-1]) else None,
                    "histogram": float(hist[-1]) if not np.isnan(hist[-1]) else None,
                    "crossover": self._detect_macd_crossover(macd, signal)
                }

                # Moving Averages
                sma_20 = talib.SMA(close_prices, timeperiod=20)
                sma_50 = talib.SMA(close_prices, timeperiod=50)
                sma_200 = talib.SMA(close_prices, timeperiod=200)

                results["indicators"]["moving_averages"] = {
                    "sma_20": float(sma_20[-1]) if not np.isnan(sma_20[-1]) else None,
                    "sma_50": float(sma_50[-1]) if not np.isnan(sma_50[-1]) else None,
                    "sma_200": float(sma_200[-1]) if not np.isnan(sma_200[-1]) else None,
                    "golden_cross": sma_20[-1] > sma_50[-1] > sma_200[-1],
                    "death_cross": sma_20[-1] < sma_50[-1] < sma_200[-1]
                }

                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(close_prices)
                results["indicators"]["bollinger_bands"] = {
                    "upper": float(upper[-1]) if not np.isnan(upper[-1]) else None,
                    "middle": float(middle[-1]) if not np.isnan(middle[-1]) else None,
                    "lower": float(lower[-1]) if not np.isnan(lower[-1]) else None,
                    "percent_b": ((close_prices[-1] - lower[-1]) / (upper[-1] - lower[-1]))
                    if not np.isnan(upper[-1]) and not np.isnan(lower[-1]) else None
                }

                # Обнаружение паттернов
                results["patterns"] = self._detect_chart_patterns(prices_df)

                # Генерация сигналов
                results["signals"] = self._generate_trading_signals(results["indicators"])

        # Использование LLM для интерпретации
        if self.llm:
            results["summary"] = self.technical_chain.run(results)

        return results

    def _perform_sentiment_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ настроений"""
        results = {
            "news_sentiment": 0.0,
            "social_sentiment": 0.0,
            "overall_sentiment": 0.0,
            "sentiment_trend": "neutral",
            "key_topics": [],
            "sentiment_by_source": {}
        }

        if "news" in data and data["news"]:
            sentiments = []
            topics = []

            for news_item in data["news"]:
                if "sentiment" in news_item:
                    sentiments.append(news_item["sentiment"])
                if "topics" in news_item:
                    topics.extend(news_item["topics"])

            if sentiments:
                results["news_sentiment"] = np.mean(sentiments)

            if topics:
                from collections import Counter
                topic_counts = Counter(topics)
                results["key_topics"] = [topic for topic, _ in topic_counts.most_common(5)]

        # Расчет общего настроения
        results["overall_sentiment"] = results["news_sentiment"]

        # Определение тренда
        if results["overall_sentiment"] > 0.2:
            results["sentiment_trend"] = "bullish"
        elif results["overall_sentiment"] < -0.2:
            results["sentiment_trend"] = "bearish"
        else:
            results["sentiment_trend"] = "neutral"

        return results

    def _generate_insights(self, technical: Dict, sentiment: Dict,
                           task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация инвестиционных инсайтов"""
        insights = {
            "market_outlook": "",
            "key_drivers": [],
            "opportunities": [],
            "threats": [],
            "price_targets": {},
            "time_horizon": ""
        }

        # Анализ конвергенции/дивергенции
        if (technical["indicators"].get("rsi", {}).get("oversold") and
                sentiment["sentiment_trend"] == "bullish"):
            insights["market_outlook"] = "Strong buying opportunity"
            insights["key_drivers"].append("Oversold RSI with positive sentiment")

        elif (technical["indicators"].get("rsi", {}).get("overbought") and
              sentiment["sentiment_trend"] == "bearish"):
            insights["market_outlook"] = "Potential correction ahead"
            insights["key_drivers"].append("Overbought RSI with negative sentiment")

        # Расчет целевых цен
        if "prices" in task_input.get("data", {}):
            current_price = task_input["data"]["prices"][-1]["close"]

            # На основе скользящих средних
            sma_200 = technical["indicators"].get("moving_averages", {}).get("sma_200")
            if sma_200:
                insights["price_targets"] = {
                    "conservative": sma_200 * 0.95,  # 5% ниже 200 SMA
                    "moderate": current_price * 1.1,  # 10% рост
                    "aggressive": current_price * 1.25,  # 25% рост
                    "support": sma_200,
                    "resistance": current_price * 1.15
                }

        # Определение горизонта инвестиций
        if technical["indicators"].get("moving_averages", {}).get("golden_cross"):
            insights["time_horizon"] = "Long-term (6+ months)"
        else:
            insights["time_horizon"] = "Short-to-medium term (1-6 months)"

        return insights

    def _assess_risks(self, insights: Dict[str, Any],
                      technical: Dict[str, Any]) -> Dict[str, Any]:
        """Оценка рисков"""
        risk_score = 0.0
        risk_factors = []

        # Волатильность
        if technical.get("indicators", {}).get("bollinger_bands"):
            bb_width = (technical["indicators"]["bollinger_bands"]["upper"] -
                        technical["indicators"]["bollinger_bands"]["lower"])
            if bb_width and bb_width > technical["indicators"]["bollinger_bands"]["middle"] * 0.2:
                risk_score += 0.3
                risk_factors.append("High volatility")

        # Перекупленность/перепроданность
        rsi = technical.get("indicators", {}).get("rsi", {}).get("current")
        if rsi:
            if rsi > 80 or rsi < 20:
                risk_score += 0.2
                risk_factors.append("Extreme RSI levels")

        # MACD дивергенция
        macd_crossover = technical.get("indicators", {}).get("macd", {}).get("crossover")
        if macd_crossover == "bearish":
            risk_score += 0.15
            risk_factors.append("Bearish MACD crossover")

        # Нормализация оценки риска (0-1)
        risk_score = min(1.0, risk_score)

        return {
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "risk_factors": risk_factors,
            "hedging_suggestions": self._get_hedging_suggestions(risk_score)
        }

    def _generate_recommendation(self, insights: Dict[str, Any],
                                 risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация инвестиционной рекомендации"""
        recommendation = {
            "action": "HOLD",
            "strength": "NEUTRAL",
            "reasoning": "",
            "entry_point": None,
            "stop_loss": None,
            "take_profit": None
        }

        risk_level = risk_assessment.get("risk_level", "MEDIUM")
        market_outlook = insights.get("market_outlook", "")

        if "buying opportunity" in market_outlook.lower() and risk_level != "HIGH":
            recommendation["action"] = "BUY"
            recommendation["strength"] = "STRONG" if risk_level == "LOW" else "MODERATE"
            recommendation["reasoning"] = "Favorable technicals with positive sentiment"

            if insights.get("price_targets"):
                recommendation["entry_point"] = insights["price_targets"].get("support")
                recommendation["stop_loss"] = recommendation["entry_point"] * 0.95
                recommendation["take_profit"] = insights["price_targets"].get("moderate")

        elif "correction" in market_outlook.lower():
            recommendation["action"] = "SELL"
            recommendation["strength"] = "MODERATE"
            recommendation["reasoning"] = "Overbought conditions with negative sentiment"

        return recommendation

    def _calculate_confidence(self, insights: Dict[str, Any]) -> float:
        """Расчет уровня уверенности"""
        confidence = 0.7  # Базовая уверенность

        # Увеличиваем уверенность при наличии четких сигналов
        if insights.get("key_drivers"):
            confidence += 0.1 * len(insights["key_drivers"])

        if insights.get("price_targets"):
            confidence += 0.1

        # Ограничиваем максимальную уверенность
        return min(0.95, confidence)

    def _detect_macd_crossover(self, macd: np.ndarray, signal: np.ndarray) -> str:
        """Обнаружение пересечения MACD"""
        if len(macd) < 2 or len(signal) < 2:
            return "neutral"

        if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
            return "bullish"
        elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
            return "bearish"
        return "neutral"

    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[str]:
        """Обнаружение графических паттернов"""
        patterns = []

        try:
            open_prices = df["open"].values
            high_prices = df["high"].values
            low_prices = df["low"].values
            close_prices = df["close"].values

            # Doji pattern
            doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            if doji[-1] != 0:
                patterns.append("Doji")

            # Hammer pattern
            hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            if hammer[-1] != 0:
                patterns.append("Hammer")

            # Engulfing pattern
            engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            if engulfing[-1] > 0:
                patterns.append("Bullish Engulfing")
            elif engulfing[-1] < 0:
                patterns.append("Bearish Engulfing")

        except Exception as e:
            logger.warning(f"Ошибка при обнаружении паттернов: {e}")

        return patterns

    def _generate_trading_signals(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерация торговых сигналов"""
        signals = []

        # RSI сигналы
        rsi = indicators.get("rsi", {})
        if rsi.get("oversold"):
            signals.append({
                "type": "BUY",
                "indicator": "RSI",
                "strength": "STRONG",
                "description": "RSI indicates oversold conditions"
            })
        elif rsi.get("overbought"):
            signals.append({
                "type": "SELL",
                "indicator": "RSI",
                "strength": "STRONG",
                "description": "RSI indicates overbought conditions"
            })

        # MACD сигналы
        macd = indicators.get("macd", {})
        if macd.get("crossover") == "bullish":
            signals.append({
                "type": "BUY",
                "indicator": "MACD",
                "strength": "MODERATE",
                "description": "Bullish MACD crossover"
            })
        elif macd.get("crossover") == "bearish":
            signals.append({
                "type": "SELL",
                "indicator": "MACD",
                "strength": "MODERATE",
                "description": "Bearish MACD crossover"
            })

        # Moving Average сигналы
        ma = indicators.get("moving_averages", {})
        if ma.get("golden_cross"):
            signals.append({
                "type": "BUY",
                "indicator": "MA",
                "strength": "STRONG",
                "description": "Golden cross detected"
            })
        elif ma.get("death_cross"):
            signals.append({
                "type": "SELL",
                "indicator": "MA",
                "strength": "STRONG",
                "description": "Death cross detected"
            })

        return signals

    def _get_risk_level(self, risk_score: float) -> str:
        """Определение уровня риска"""
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.6:
            return "MEDIUM"
        else:
            return "HIGH"

    def _get_hedging_suggestions(self, risk_score: float) -> List[str]:
        """Предложения по хеджированию"""
        suggestions = []

        if risk_score >= 0.7:
            suggestions.append("Consider buying protective puts")
            suggestions.append("Reduce position size by 30-50%")
            suggestions.append("Implement stop-loss orders")
        elif risk_score >= 0.4:
            suggestions.append("Consider collar strategy")
            suggestions.append("Diversify across sectors")
            suggestions.append("Use trailing stops")

        return suggestions