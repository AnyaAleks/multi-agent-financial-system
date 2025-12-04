# Multi-Agent Intelligent Financial Analysis System

A sophisticated multi-agent system for financial analysis using LangChain + CrewAI.

## 🚀 Features

- **Multi-Agent Architecture**: Three specialized agents (Data, Analysis, Report) with CrewAI orchestration
- **MCP Protocol Integration**: Unified access to financial data sources
- **RAG System**: Knowledge graph-enhanced retrieval for accurate analysis
- **Hierarchical Memory**: Short, medium, and long-term memory management
- **Fault Tolerance**: Circuit breakers, agent redundancy, graceful degradation
- **Continuous Learning**: Self-evolving agents with feedback loops
- **Quality Evaluation**: Automated metrics and human-in-the-loop validation

## 📋 Prerequisites

- Python 3.10+
- Redis (for memory management)
- Neo4j (optional, for knowledge graph)
- OpenAI API key

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-agent-financial-system.git
cd multi-agent-financial-system
Create virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -e .[dev,mcp,monitoring]
Configure environment:

bash
cp .env.example .env
# Edit .env with your API keys and settings
🚀 Quick Start
Start MCP servers:

bash
python -m src.mcp_servers.financial_data_server
python -m src.mcp_servers.news_sentiment_server
Run stock analysis:

bash
python -m src.main analyze --ticker AAPL --timeframe 1y
Check system status:

bash
python -m src.main status
Launch dashboard:

bash
python -m src.main dashboard
🏗 System Architecture
text
┌─────────────────┐
│   User Interface│
│   (CLI/Web)     │
└────────┬────────┘
         │
┌────────▼────────┐
│  CrewAI Manager │
│  (Orchestration)│
└────────┬────────┘
         │
┌────────▼────────┐    ┌──────────────┐    ┌──────────────┐
│   Data Agent    │◄──►│ MCP Servers  │◄──►│ Data Sources │
│                 │    │              │    │ (YFinance,   │
└────────┬────────┘    └──────────────┘    │  NewsAPI)    │
         │                                 └──────────────┘
┌────────▼────────┐
│ Analysis Agent  │
│ (LLM + TA)      │
└────────┬────────┘
         │
┌────────▼────────┐
│  Report Agent   │
│ (Visualization) │
└────────┬────────┘
         │
┌────────▼────────┐
│     Output      │
│ (PDF/Dashboard) │
└─────────────────┘
🧩 Core Components
Agents
Data Agent: Collects and prepares financial data

Analysis Agent: Performs technical and sentiment analysis

Report Agent: Generates visual reports and dashboards

Manager Agent: Orchestrates workflow and task dependencies

Chains
Financial Data Analysis Chain: Data validation and quality assessment

Technical Analysis Chain: Indicator calculation and pattern recognition

LLM Insight Chain: Investment insights and risk assessment

MCP Servers
Financial Data Server: OHLCV data, fundamentals, quotes

News Sentiment Server: News aggregation and sentiment analysis

🔧 Configuration
Edit config/settings.py or environment variables:

python
# API Keys
OPENAI_API_KEY=your_key_here

# MCP Servers
MCP_FINANCIAL_HOST=localhost
MCP_FINANCIAL_PORT=8001

# Performance
PARALLEL_EXECUTION=true
MAX_WORKERS=4
📊 Monitoring
Access monitoring dashboard:

Metrics: http://localhost:9090

Health checks: http://localhost:9090/health

Performance: http://localhost:9090/metrics

🧪 Testing
bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_agents.py -v
pytest tests/test_chains.py -v

# With coverage
pytest --cov=src tests/
📈 Performance Optimization
The system implements:

Parallel data collection

LLM batching and model sharding

Multi-tier caching

Resource-aware execution planning

Dynamic tier selection (Basic/Standard/Deep analysis)

🛡 Fault Tolerance
Agent health monitoring with heartbeats

Hot standby agents and agent pools

Circuit breakers for external services

Graceful degradation strategies

Automated recovery workflows

📚 Documentation
Full documentation available at:

System Architecture

Agent Design

API Reference

Deployment Guide

🤝 Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

📄 License
This project is licensed under the MIT License - see LICENSE file for details.

🙏 Acknowledgments
CrewAI for multi-agent orchestration

LangChain for LLM integration

MCP Protocol for tool standardization

📞 Contact
Anna Alekseeva - anna@example.com

Project Link: https://github.com/yourusername/multi-agent-financial-system