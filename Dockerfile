# Multi-Agent Financial Analysis System Dockerfile
# Multi-stage build для оптимизации размера образа

# Этап 1: Сборка зависимостей
FROM python:3.10-slim as builder

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --user --no-cache-dir -r requirements.txt

# Этап 2: Финальный образ
FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей для runtime
RUN apt-get update && apt-get install -y \
    curl \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Копирование Python зависимостей из builder
COPY --from=builder /root/.local /root/.local

# Добавление .local/bin в PATH
ENV PATH=/root/.local/bin:$PATH

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p data/raw data/processed data/knowledge_base \
    reports dashboards images logs

# Настройка переменных окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV REDIS_URL=redis://localhost:6379
ENV OPENAI_API_KEY=""
ENV NEO4J_URL="bolt://localhost:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="password"

# Открытие портов
EXPOSE 8501  # Streamlit дашборд
EXPOSE 8001  # MCP Financial Server
EXPOSE 8002  # MCP News Server
EXPOSE 9090  # Prometheus метрики

# Скрипт запуска
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Здоровье контейнера
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Точка входа
ENTRYPOINT ["/docker-entrypoint.sh"]

# Команда по умолчанию
CMD ["start-all"]