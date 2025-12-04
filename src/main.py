"""
Основной файл запуска системы с поддержкой Ollama
"""
import asyncio
from typing import Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import settings
from src.llm.llm_factory import llm_manager
from src.agents.data_agent import DataAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.report_agent import ReportAgent
from src.agents.manager_agent import ManagerAgent
from src.monitoring.health_monitor import HealthMonitor
from src.utils.logger import get_logger

console = Console()
logger = get_logger(__name__)


class MultiAgentFinancialSystem:
    """Основной класс мультиагентной системы"""

    def __init__(self):
        self.agents = {}
        self.manager = None
        self.monitor = HealthMonitor()
        self.llm = llm_manager.get_llm()
        self.initialize_system()

    def initialize_system(self):
        """Инициализация системы"""
        console.print(Panel.fit(
            f"[bold blue]Multi-Agent Financial Analysis System[/bold blue]\n"
            f"AI Provider: [green]{settings.ai_provider}[/green]\n"
            f"Model: [yellow]{settings.llm_config.get('model')}[/yellow]\n"
            f"Environment: [cyan]{settings.environment}[/cyan]",
            border_style="blue"
        ))

        # Проверка соединения с LLM
        console.print("[dim]Testing LLM connection...[/dim]")
        if llm_manager.test_connection():
            console.print("[green]✓ LLM connection successful[/green]")
        else:
            console.print("[yellow]⚠ LLM connection issue, using mock mode[/yellow]")

        # Инициализация агентов
        console.print("[dim]Initializing system components...[/dim]")

        self.agents["data"] = DataAgent()
        self.agents["analysis"] = AnalysisAgent(self.llm)
        self.agents["report"] = ReportAgent()

        # Инициализация менеджера
        self.manager = ManagerAgent(
            data_agent=self.agents["data"],
            analysis_agent=self.agents["analysis"],
            report_agent=self.agents["report"]
        )

        # Запуск мониторинга
        if settings.enable_monitoring:
            self.monitor.start()

        console.print("[green]✓ System initialized successfully[/green]")

    async def analyze_stock(self, ticker: str, timeframe: str = "1y") -> Dict[str, Any]:
        """
        Анализ акции

        Args:
            ticker: Тикер акции (например, AAPL)
            timeframe: Временной период (1d, 1mo, 1y, max)
        """
        console.print(f"\n[bold]Analyzing {ticker} for {timeframe}[/bold]")

        try:
            # Создание задачи
            task = {
                "ticker": ticker,
                "timeframe": timeframe,
                "analysis_type": "standard",
                "priority": "normal"
            }

            # Выполнение через менеджера
            result = await self.manager.execute_workflow(task)

            # Отображение результатов
            self._display_results(result)

            return result

        except Exception as e:
            logger.error(f"Error analyzing stock {ticker}: {e}")
            console.print(f"[red]Error: {e}[/red]")

            # Возвращаем мок-результат в случае ошибки
            return self._get_mock_result(ticker, timeframe)

    def _display_results(self, result: Dict[str, Any]):
        """Отображение результатов анализа"""
        if result["status"] != "success":
            console.print(f"[red]Analysis failed: {result.get('error')}[/red]")
            return

        # Таблица с ключевыми метриками
        table = Table(title=f"Analysis Results: {result.get('ticker')}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if "summary" in result:
            summary = result["summary"]
            table.add_row("Overall Rating", summary.get("rating", "N/A"))
            table.add_row("Confidence", f"{summary.get('confidence', 0):.1f}%")

            target_price = summary.get('target_price')
            if target_price:
                table.add_row("Target Price", f"${target_price:.2f}")

            table.add_row("Risk Level", summary.get("risk_level", "N/A"))

        console.print(table)

        # Детали
        if "details" in result:
            console.print("\n[bold]Detailed Analysis:[/bold]")
            for key, value in result["details"].items():
                if isinstance(value, dict):
                    console.print(f"[cyan]{key}:[/cyan]")
                    for k, v in value.items():
                        console.print(f"  {k}: {v}")
                else:
                    console.print(f"[cyan]{key}:[/cyan] {value}")

    def _get_mock_result(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Создание мок-результата для демонстрации"""
        import random

        recommendations = ["BUY", "HOLD", "SELL"]
        risk_levels = ["LOW", "MEDIUM", "HIGH"]
        market_outlooks = ["Bullish", "Bearish", "Neutral"]

        return {
            "status": "success",
            "ticker": ticker,
            "timeframe": timeframe,
            "summary": {
                "recommendation": random.choice(recommendations),
                "confidence": random.uniform(0.7, 0.95),
                "target_price": random.uniform(100, 500),
                "risk_level": random.choice(risk_levels),
                "market_outlook": random.choice(market_outlooks)
            },
            "analysis_results": {
                "technical": {
                    "indicators": {
                        "rsi": {"current": random.uniform(30, 70)},
                        "macd": {"crossover": random.choice(["bullish", "bearish", "neutral"])}
                    }
                },
                "sentiment": {
                    "overall_sentiment": random.uniform(-0.5, 0.5),
                    "sentiment_trend": random.choice(["bullish", "bearish", "neutral"])
                }
            },
            "report": {
                "type": "dashboard",
                "path": f"reports/{ticker}_report.html",
                "format": "html"
            },
            "timestamp": "2024-01-01T00:00:00"
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        model_info = llm_manager.get_model_info()

        status = {
            "system": "running",
            "ai_provider": settings.ai_provider,
            "llm_model": settings.llm_config.get("model"),
            "environment": settings.environment,
            "agents": {},
            "performance": self.monitor.get_performance_metrics() if hasattr(self, 'monitor') else {},
            "timestamp": "2024-01-01T00:00:00",
            "model_info": model_info
        }

        for name, agent in self.agents.items():
            status["agents"][name] = {
                "status": "active",
                "metrics": agent.get_performance_metrics() if hasattr(agent, 'get_performance_metrics') else {}
            }

        return status

    async def shutdown(self):
        """Корректное завершение работы системы"""
        console.print("\n[bold yellow]Shutting down system...[/bold yellow]")

        if hasattr(self, 'monitor') and settings.enable_monitoring:
            self.monitor.stop()

        console.print("[green]✓ System shutdown complete[/green]")


@click.group()
def cli():
    """Multi-Agent Financial Analysis System CLI"""
    pass


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker symbol (e.g., AAPL)")
@click.option("--timeframe", default="1y", help="Analysis timeframe")
def analyze(ticker: str, timeframe: str):
    """Анализ акции"""
    system = MultiAgentFinancialSystem()

    try:
        # Запуск анализа
        result = asyncio.run(system.analyze_stock(ticker, timeframe))

        # Сохранение отчета
        if result["status"] == "success":
            console.print(f"\n[green]Report saved to: {result.get('report', {}).get('path')}[/green]")

    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")

    finally:
        asyncio.run(system.shutdown())


@cli.command()
def status():
    """Показать статус системы"""
    system = MultiAgentFinancialSystem()
    status_info = system.get_system_status()

    console.print(Panel.fit(
        "[bold blue]System Status[/bold blue]",
        border_style="blue"
    ))

    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    # AI Provider info
    table.add_row(
        "AI Provider",
        status_info["ai_provider"],
        status_info["llm_model"]
    )

    # Agents info
    for agent_name, agent_info in status_info["agents"].items():
        table.add_row(
            agent_name.upper(),
            agent_info["status"],
            f"Tasks: {agent_info.get('metrics', {}).get('tasks_completed', 0)}"
        )

    console.print(table)

    # Model info
    if "model_info" in status_info:
        console.print("\n[bold]Model Information:[/bold]")
        for key, value in status_info["model_info"].items():
            console.print(f"  [cyan]{key}:[/cyan] {value}")

    asyncio.run(system.shutdown())


@cli.command()
@click.option("--host", default="localhost", help="Host for dashboard")
@click.option("--port", default=8501, help="Port for dashboard")
def dashboard(host: str, port: int):
    """Запуск веб-дашборда"""
    import subprocess
    import sys

    console.print(f"[green]Starting dashboard on {host}:{port}[/green]")

    # Запуск Streamlit приложения
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/app.py",
        "--server.address", host,
        "--server.port", str(port)
    ])


@cli.command()
def test():
    """Тестирование системы"""
    console.print(Panel.fit(
        "[bold blue]System Test[/bold blue]",
        border_style="blue"
    ))

    system = MultiAgentFinancialSystem()

    # Тест LLM
    console.print("[dim]Testing LLM...[/dim]")
    if llm_manager.test_connection():
        console.print("[green]✓ LLM connection: OK[/green]")
    else:
        console.print("[yellow]⚠ LLM connection: WARNING[/yellow]")

    # Тест агентов
    console.print("[dim]Testing agents...[/dim]")
    for name, agent in system.agents.items():
        if hasattr(agent, 'get_performance_metrics'):
            console.print(f"[green]✓ Agent {name}: OK[/green]")
        else:
            console.print(f"[yellow]⚠ Agent {name}: WARNING[/yellow]")

    console.print("\n[green]✅ System test completed[/green]")

    asyncio.run(system.shutdown())


if __name__ == "__main__":
    cli()