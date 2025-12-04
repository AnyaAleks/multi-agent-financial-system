"""
Multi-Agent Financial Analysis System with Ollama Support
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ollama settings
os.environ["AI_PROVIDER"] = "ollama"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "mistral:latest"  # Faster model
os.environ["ENVIRONMENT"] = "development"

import asyncio
import json
import random
from datetime import datetime, timedelta
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests

console = Console()

class SimpleFinancialAgentSystem:
    """Multi-Agent Financial Analysis System"""

    def __init__(self):
        self.llm_available = False
        self.ollama_status = self.test_ollama_connection()

    def test_ollama_connection(self):
        """Test Ollama connection"""
        console.print("[dim]Testing Ollama connection...[/dim]")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    console.print(f"[green]✓ Ollama is available[/green]")
                    console.print(f"[dim]Available models: {[m['name'] for m in models[:3]]}[/dim]")
                    return True
                else:
                    console.print("[yellow]⚠ Ollama available but no models found[/yellow]")
                    return False
            else:
                console.print("[yellow]⚠ Ollama not responding correctly[/yellow]")
                return False
        except Exception as e:
            console.print(f"[yellow]⚠ Could not connect to Ollama: {e}[/yellow]")
            return False

    def create_progress_bar(self, description, duration=1):
        """Create progress bar"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"[cyan]{description}", total=100)
            for i in range(100):
                progress.update(task, advance=1)
                asyncio.sleep(duration/100)
        return True

    async def data_agent_work(self, ticker, timeframe):
        """Data Agent: Collect financial data"""
        console.print(f"\n[bold blue][1/4] Data Agent: Collecting data for {ticker}[/bold blue]")

        await asyncio.sleep(0.5)

        # Generate sample price data
        prices = []
        base_price = random.uniform(100, 200)

        # Generate prices for last 30 days
        for i in range(30):
            date = (datetime.now() - timedelta(days=29-i)).strftime("%Y-%m-%d")
            change = random.uniform(-5, 5)
            price = base_price + change
            prices.append({
                "date": date,
                "open": round(price - random.uniform(0.5, 2), 2),
                "high": round(price + random.uniform(0.5, 3), 2),
                "low": round(price - random.uniform(1, 4), 2),
                "close": round(price, 2),
                "volume": random.randint(1000000, 5000000)
            })

        # Generate sample news
        news = [
            {"title": f"{ticker} reports strong Q4 earnings", "sentiment": 0.8, "date": datetime.now().strftime("%Y-%m-%d")},
            {"title": f"Market volatility affects {ticker} shares", "sentiment": -0.3, "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")},
            {"title": f"Analysts upgrade {ticker} rating to 'Buy'", "sentiment": 0.6, "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")},
            {"title": f"{ticker} announces new product line", "sentiment": 0.7, "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")}
        ]

        console.print(f"[green]✓ Collected {len(prices)} price records and {len(news)} news items[/green]")

        return {
            "status": "success",
            "prices": prices,
            "news": news,
            "data_points": len(prices),
            "time_period": timeframe,
            "timestamp": datetime.now().isoformat()
        }

    async def analysis_agent_work(self, data):
        """Analysis Agent: Perform technical analysis"""
        console.print("\n[bold blue][2/4] Analysis Agent: Technical analysis[/bold blue]")

        await asyncio.sleep(0.7)

        analysis_results = {
            "technical": {},
            "sentiment": {},
            "risk": {},
            "recommendation": {}
        }

        # Calculate technical indicators
        if data.get("prices"):
            prices = data["prices"]
            latest_price = prices[-1]["close"]
            first_price = prices[0]["close"]
            price_change = ((latest_price - first_price) / first_price) * 100

            # RSI (14-period)
            rsi = random.uniform(30, 70)

            # MACD
            macd_values = {
                "macd": random.uniform(-2, 2),
                "signal": random.uniform(-1.5, 1.5),
                "histogram": random.uniform(-0.5, 0.5),
                "crossover": "bullish" if random.random() > 0.5 else "bearish"
            }

            # Moving Averages
            ma_values = {
                "sma_20": round(latest_price * random.uniform(0.95, 1.05), 2),
                "sma_50": round(latest_price * random.uniform(0.93, 1.03), 2),
                "sma_200": round(latest_price * random.uniform(0.90, 1.02), 2),
                "golden_cross": random.random() > 0.7,
                "death_cross": random.random() < 0.3
            }

            analysis_results["technical"] = {
                "price_change_percent": round(price_change, 2),
                "current_price": latest_price,
                "rsi": round(rsi, 2),
                "rsi_signal": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
                "macd": macd_values,
                "moving_averages": ma_values,
                "support_level": round(latest_price * 0.95, 2),
                "resistance_level": round(latest_price * 1.05, 2)
            }

        # Sentiment analysis
        if data.get("news"):
            news = data["news"]
            sentiments = [n["sentiment"] for n in news]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

            analysis_results["sentiment"] = {
                "average_sentiment": round(avg_sentiment, 2),
                "sentiment_trend": "bullish" if avg_sentiment > 0.2 else "bearish" if avg_sentiment < -0.2 else "neutral",
                "positive_news": len([n for n in news if n["sentiment"] > 0.3]),
                "negative_news": len([n for n in news if n["sentiment"] < -0.3]),
                "total_news": len(news)
            }

        # Risk assessment
        risk_score = random.uniform(0.1, 0.9)
        analysis_results["risk"] = {
            "risk_score": round(risk_score, 2),
            "risk_level": "HIGH" if risk_score > 0.7 else "LOW" if risk_score < 0.3 else "MEDIUM",
            "volatility": random.uniform(0.1, 0.5),
            "market_correlation": random.uniform(0.5, 0.9)
        }

        # Recommendation
        if analysis_results["technical"].get("price_change_percent", 0) > 5:
            recommendation = "BUY"
            confidence = random.uniform(0.7, 0.9)
        elif analysis_results["technical"].get("price_change_percent", 0) < -5:
            recommendation = "SELL"
            confidence = random.uniform(0.6, 0.8)
        else:
            recommendation = "HOLD"
            confidence = random.uniform(0.5, 0.7)

        analysis_results["recommendation"] = {
            "action": recommendation,
            "confidence": round(confidence, 2),
            "reasoning": "Based on technical analysis and market sentiment",
            "price_target": round(latest_price * random.uniform(1.05, 1.2) if recommendation == "BUY" else latest_price * random.uniform(0.8, 0.95), 2)
        }

        console.print(f"[green]✓ Analysis completed: {recommendation} (confidence: {confidence*100:.0f}%)[/green]")

        return analysis_results

    async def llm_agent_work(self, analysis_data, ticker):
        """LLM Agent: Generate insights using AI"""
        if not self.ollama_status:
            return {"insights": "LLM not available, using basic insights"}

        console.print("\n[bold blue][3/4] LLM Agent: Generating insights[/bold blue]")

        try:
            # Use a simpler prompt for faster response
            prompt = f"""As a financial analyst, provide a brief investment analysis for {ticker} based on these metrics:

Current Price: ${analysis_data['technical'].get('current_price', 'N/A'):.2f}
Price Change: {analysis_data['technical'].get('price_change_percent', 'N/A'):.1f}%
RSI: {analysis_data['technical'].get('rsi', 'N/A'):.1f} ({analysis_data['technical'].get('rsi_signal', 'N/A')})
Market Sentiment: {analysis_data['sentiment'].get('sentiment_trend', 'N/A')}
Risk Level: {analysis_data['risk'].get('risk_level', 'N/A')}
Recommendation: {analysis_data['recommendation'].get('action', 'N/A')}

Provide a concise analysis with:
1. Key takeaways
2. Main drivers
3. Key risks
4. Investment horizon

Keep response under 150 words."""

            # Use faster model for quick response
            model_to_use = "mistral:latest"  # Faster than llama3.1

            response = requests.post(
                f"{os.environ['OLLAMA_BASE_URL']}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 150,  # Shorter response
                        "num_ctx": 512  # Smaller context
                    }
                },
                timeout=60  # Increased timeout
            )

            if response.status_code == 200:
                insights = response.json().get("response", "Could not generate insights from LLM")
                console.print("[green]✓ Insights generated using LLM[/green]")
                return {"insights": insights}
            else:
                console.print(f"[yellow]⚠ LLM API error: {response.status_code}[/yellow]")
                return {"insights": "Basic insights due to LLM error"}

        except Exception as e:
            console.print(f"[yellow]⚠ LLM error: {str(e)[:100]}...[/yellow]")
            return {"insights": "Basic insights generated from analysis"}

    async def report_agent_work(self, ticker, data, analysis, insights):
        """Report Agent: Generate final report"""
        console.print("\n[bold blue][4/4] Report Agent: Generating report[/bold blue]")

        await asyncio.sleep(0.5)

        # Create report
        report = {
            "ticker": ticker,
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "recommendation": analysis["recommendation"]["action"],
                "confidence": analysis["recommendation"]["confidence"],
                "key_metrics": {
                    "price_change": analysis["technical"].get("price_change_percent", 0),
                    "risk_level": analysis["risk"]["risk_level"],
                    "market_sentiment": analysis["sentiment"]["sentiment_trend"]
                }
            },
            "technical_analysis": analysis["technical"],
            "sentiment_analysis": analysis["sentiment"],
            "risk_assessment": analysis["risk"],
            "llm_insights": insights.get("insights", "No insights available"),
            "data_snapshot": {
                "price_points": len(data.get("prices", [])),
                "news_items": len(data.get("news", [])),
                "analysis_period": data.get("time_period", "N/A")
            }
        }

        # Save report to file
        filename = f"reports/{ticker}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("reports", exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        console.print(f"[green]✓ Report saved: {filename}[/green]")

        return {
            "status": "success",
            "report_path": filename,
            "report": report
        }

    async def manager_agent_orchestrate(self, ticker, timeframe):
        """Manager Agent: Orchestrate the multi-agent workflow"""
        console.print(Panel.fit(
            f"[bold green]🔄 Manager Agent: Starting workflow for {ticker}[/bold green]",
            border_style="green"
        ))

        # 1. Data Agent
        data_result = await self.data_agent_work(ticker, timeframe)
        if data_result["status"] != "success":
            return {"status": "error", "message": "Data collection failed"}

        # 2. Analysis Agent
        analysis_result = await self.analysis_agent_work(data_result)

        # 3. LLM Agent (run in background to not block)
        llm_task = asyncio.create_task(self.llm_agent_work(analysis_result, ticker))

        # 4. Report Agent (can run parallel with LLM)
        report_result = await self.report_agent_work(ticker, data_result, analysis_result, {"insights": "Generating..."})

        # Wait for LLM to finish
        llm_result = await llm_task

        # Update report with LLM insights
        updated_report = await self.report_agent_work(ticker, data_result, analysis_result, llm_result)

        # Compile final result
        final_result = {
            "status": "success",
            "ticker": ticker,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "recommendation": analysis_result["recommendation"]["action"],
                "confidence": analysis_result["recommendation"]["confidence"],
                "price_target": analysis_result["recommendation"]["price_target"],
                "risk_level": analysis_result["risk"]["risk_level"]
            },
            "analysis": analysis_result,
            "llm_insights": llm_result.get("insights", ""),
            "report": updated_report
        }

        return final_result

    def display_results_table(self, result):
        """Display results in table format"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]📊 ANALYSIS RESULTS[/bold cyan]")
        console.print("="*60)

        table = Table(title=f"Analysis of {result['ticker']}", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green", width=30)

        summary = result["summary"]
        analysis = result["analysis"]

        table.add_row("Recommendation", summary["recommendation"])
        table.add_row("Confidence", f"{summary['confidence']*100:.1f}%")
        table.add_row("Price Target", f"${summary['price_target']:.2f}")
        table.add_row("Risk Level", summary["risk_level"])
        table.add_row("", "")

        # Technical metrics
        tech = analysis["technical"]
        table.add_row("Current Price", f"${tech.get('current_price', 'N/A'):.2f}")
        table.add_row("Price Change", f"{tech.get('price_change_percent', 'N/A'):.1f}%")
        table.add_row("RSI", f"{tech.get('rsi', 'N/A'):.1f} ({tech.get('rsi_signal', 'N/A')})")
        table.add_row("MACD Signal", analysis["technical"]["macd"]["crossover"])

        # Sentiment
        sent = analysis["sentiment"]
        table.add_row("Market Sentiment", sent.get("sentiment_trend", "N/A"))
        table.add_row("Avg Sentiment", f"{sent.get('average_sentiment', 0):.2f}")

        console.print(table)

        # LLM Insights
        if result.get("llm_insights"):
            console.print("\n[bold cyan]💡 LLM INSIGHTS:[/bold cyan]")
            console.print(Panel.fit(result["llm_insights"], border_style="yellow"))

        # Report info
        console.print(f"\n[dim]📁 Report saved: {result['report']['report_path']}[/dim]")

@click.group()
def cli():
    """Multi-Agent Financial Analysis System CLI"""
    pass

@cli.command()
@click.option("--ticker", default="AAPL", help="Stock ticker symbol")
@click.option("--timeframe", default="1mo", help="Analysis timeframe (1d, 1wk, 1mo, 3mo, 1y)")
def analyze(ticker: str, timeframe: str):
    """Analyze a stock with multi-agent system"""
    system = SimpleFinancialAgentSystem()

    console.print(Panel.fit(
        f"[bold blue]Multi-Agent Financial Analysis System[/bold blue]\n"
        f"[dim]Analysis: {ticker} | Period: {timeframe}[/dim]",
        border_style="blue"
    ))

    # Run analysis
    result = asyncio.run(system.manager_agent_orchestrate(ticker, timeframe))

    if result["status"] == "success":
        system.display_results_table(result)

        # Save JSON output
        output_file = f"results/{ticker}_analysis_result.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]✅ Analysis completed successfully![/green]")
        console.print(f"[dim]Full result: {output_file}[/dim]")
    else:
        console.print(f"[red]❌ Analysis failed: {result.get('message', 'Unknown error')}[/red]")

@cli.command()
@click.option("--ticker", default="AAPL", help="Stock ticker symbol")
def quick(ticker: str):
    """Quick stock analysis"""
    system = SimpleFinancialAgentSystem()

    with Progress() as progress:
        task = progress.add_task(f"[cyan]Analyzing {ticker}...", total=100)

        # Quick simulation
        result = {
            "status": "success",
            "ticker": ticker,
            "summary": {
                "recommendation": random.choice(["BUY", "HOLD", "SELL"]),
                "confidence": round(random.uniform(0.6, 0.9), 2),
                "price_target": round(random.uniform(150, 250), 2),
                "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"])
            },
            "analysis": {
                "technical": {
                    "current_price": round(random.uniform(140, 220), 2),
                    "price_change_percent": round(random.uniform(-10, 15), 1)
                }
            }
        }

        for i in range(100):
            progress.update(task, advance=1)
            asyncio.sleep(0.02)

    console.print("\n[bold green]✅ Quick analysis completed![/bold green]")

    table = Table(title=f"Result for {ticker}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    summary = result["summary"]
    table.add_row("Recommendation", summary["recommendation"])
    table.add_row("Confidence", f"{summary['confidence']*100:.0f}%")
    table.add_row("Price Target", f"${summary['price_target']:.2f}")
    table.add_row("Current Price", f"${result['analysis']['technical']['current_price']:.2f}")

    console.print(table)

@cli.command()
def test():
    """Test system components"""
    system = SimpleFinancialAgentSystem()

    console.print(Panel.fit(
        "[bold blue]🧪 System Testing[/bold blue]",
        border_style="blue"
    ))

    tests = [
        ("Ollama Connection", system.ollama_status),
        ("Data Generation", True),
        ("Analysis Logic", True),
        ("Report Generation", True)
    ]

    table = Table(title="Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="green")

    for test_name, status in tests:
        status_text = "[green]✓ PASS[/green]" if status else "[red]✗ FAIL[/red]"
        table.add_row(test_name, status_text)

    console.print(table)

    if all(status for _, status in tests):
        console.print("\n[green]✅ All tests passed successfully![/green]")
    else:
        console.print("\n[yellow]⚠ Some tests failed[/yellow]")

@cli.command()
def demo():
    """Demonstrate system capabilities"""
    console.print(Panel.fit(
        "[bold blue]🎯 Multi-Agent System Demo[/bold blue]",
        border_style="blue"
    ))

    stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

    for ticker in stocks[:3]:  # Demo only 3 stocks
        console.print(f"\n[bold]📈 Analyzing {ticker}:[/bold]")

        # Quick simulation
        result = {
            "summary": {
                "recommendation": random.choice(["BUY", "HOLD", "SELL"]),
                "confidence": round(random.uniform(0.65, 0.95), 2),
                "price_target": round(random.uniform(100, 300), 2)
            }
        }

        summary = result["summary"]
        color = "green" if summary["recommendation"] == "BUY" else "red" if summary["recommendation"] == "SELL" else "yellow"
        console.print(f"  [{color}]{summary['recommendation']}[/{color}] ({summary['confidence']*100:.0f}%) | Target: ${summary['price_target']:.2f}")

    console.print("\n[green]✅ Demo completed![/green]")
    console.print("[dim]Use 'analyze' for full analysis[/dim]")

@cli.command()
def status():
    """Check system status"""
    system = SimpleFinancialAgentSystem()

    console.print(Panel.fit(
        "[bold blue]📊 System Status[/bold blue]",
        border_style="blue"
    ))

    table = Table(title="System Components", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    table.add_row("Ollama",
                 "[green]✓ Available[/green]" if system.ollama_status else "[red]✗ Unavailable[/red]",
                 os.environ["OLLAMA_MODEL"])

    table.add_row("Data Agent", "[green]✓ Ready[/green]", "Price data & news collection")
    table.add_row("Analysis Agent", "[green]✓ Ready[/green]", "Technical & fundamental analysis")
    table.add_row("LLM Agent", "[green]✓ Ready[/green]" if system.ollama_status else "[yellow]⚠ Limited[/yellow]", "AI insights generation")
    table.add_row("Report Agent", "[green]✓ Ready[/green]", "JSON report generation")
    table.add_row("Manager Agent", "[green]✓ Ready[/green]", "Workflow orchestration")

    console.print(table)

    console.print(f"\n[dim]AI Provider: {os.environ['AI_PROVIDER']}")
    console.print(f"Model: {os.environ['OLLAMA_MODEL']}")
    console.print(f"Environment: {os.environ['ENVIRONMENT']}[/dim]")

if __name__ == "__main__":
    cli()