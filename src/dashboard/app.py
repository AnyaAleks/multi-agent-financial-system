"""
Упрощенный дашборд для демонстрации
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Заголовок
st.title("📈 Multi-Agent Financial Analysis System")
st.markdown("---")

# Боковая панель
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)

    st.header("Настройки анализа")

    ticker = st.text_input("Тикер акции", value="AAPL")
    timeframe = st.selectbox("Период", ["1d", "1wk", "1mo", "3mo", "6mo", "1y"])
    analysis_type = st.selectbox("Тип анализа", ["Базовый", "Расширенный", "Полный"])

    if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
        st.session_state["analysis_run"] = True
        st.session_state["ticker"] = ticker

# Основной контент
if "analysis_run" in st.session_state and st.session_state["analysis_run"]:
    ticker = st.session_state["ticker"]

    # Заголовок анализа
    st.header(f"Анализ {ticker}")

    # Имитация загрузки
    with st.spinner("Выполняется анализ..."):
        import time
        time.sleep(2)

    # Создаем мок-данные
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Рекомендация", "BUY", "↗️ Положительная")

    with col2:
        st.metric("Уверенность", "85%", "+5%")

    with col3:
        st.metric("Целевая цена", "$178.50", "+3.2%")

    with col4:
        st.metric("Риск", "Средний", "Стабильный")

    # График
    st.subheader("Ценовой график")

    # Создаем тестовые данные
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    prices = pd.DataFrame({
        'date': dates,
        'price': [150 + i*2 + (i%7)*3 for i in range(30)]
    })

    fig = go.Figure(data=[
        go.Scatter(x=prices['date'], y=prices['price'], mode='lines', name='Цена')
    ])

    fig.update_layout(
        title=f"Цена {ticker} за 30 дней",
        xaxis_title="Дата",
        yaxis_title="Цена ($)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Технические индикаторы
    st.subheader("Технические индикаторы")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("RSI", "65.2", "↗️ Перекупленность")
        st.metric("MACD", "1.25", "Бычий")

    with col2:
        st.metric("SMA 20", "$172.50", "↗️ Выше цены")
        st.metric("SMA 50", "$168.20", "↗️ Поддержка")

    # Анализ настроений
    st.subheader("Анализ настроений")

    sentiments = {
        "Положительные новости": 65,
        "Нейтральные новости": 25,
        "Отрицательные новости": 10
    }

    fig2 = go.Figure(data=[
        go.Pie(labels=list(sentiments.keys()), values=list(sentiments.values()))
    ])

    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # Кнопки действий
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Скачать отчет", use_container_width=True):
            st.success("Отчет скачивается...")

    with col2:
        if st.button("🔄 Обновить анализ", use_container_width=True):
            st.rerun()

    with col3:
        if st.button("📊 Подробный анализ", use_container_width=True):
            st.info("Переход к детальному анализу...")

else:
    # Главная страница
    st.markdown("""
    ## 🎯 Добро пожаловать в систему финансового анализа
    
    Эта система использует многоагентную архитектуру для анализа финансовых данных:
    
    ### 🤖 Агенты системы:
    1. **Data Agent** - Сбор и подготовка данных
    2. **Analysis Agent** - Технический и фундаментальный анализ
    3. **Report Agent** - Генерация отчетов
    4. **Manager Agent** - Оркестрация workflow
    
    ### 📊 Возможности:
    - Анализ акций в реальном времени
    - Технические индикаторы (RSI, MACD, Moving Averages)
    - Анализ настроений на рынке
    - Генерация инвестиционных рекомендаций
    - Визуализация результатов
    
    ### 🚀 Как начать:
    1. Введите тикер акции в боковой панели
    2. Выберите период анализа
    3. Нажмите "Запустить анализ"
    """)

    # Примеры тикеров
    st.subheader("📋 Популярные акции для анализа")

    popular_stocks = [
        {"Тикер": "AAPL", "Название": "Apple Inc.", "Сектор": "Технологии"},
        {"Тикер": "MSFT", "Название": "Microsoft", "Сектор": "Технологии"},
        {"Тикер": "GOOGL", "Название": "Alphabet (Google)", "Сектор": "Технологии"},
        {"Тикер": "TSLA", "Название": "Tesla", "Сектор": "Автомобили"},
        {"Тикер": "JPM", "Название": "JPMorgan Chase", "Сектор": "Финансы"},
    ]

    st.dataframe(pd.DataFrame(popular_stocks), use_container_width=True)

# Футер
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Multi-Agent Financial Analysis System v1.0.0</p>
    <p>© 2024 Financial AI Systems</p>
</div>
""", unsafe_allow_html=True)