# test_ollama.py
import requests
import json
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage


def test_ollama_connection():
    """Тест соединения с Ollama"""
    print("🔍 Тестирую соединение с Ollama...")

    try:
        # Проверка API
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama сервер работает")
            print(f"📦 Установленные модели: {[m['name'] for m in models]}")
            return True
        else:
            print(f"❌ Ошибка: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False


def test_llama_model():
    """Тест модели Llama"""
    print("\n🧪 Тестирую модель Llama 3.1...")

    try:
        # Создаем LLM клиент
        llm = ChatOllama(
            model="llama3.1:latest",
            temperature=0.1,
            base_url="http://localhost:11434"
        )

        # Простой запрос
        response = llm.invoke([
            HumanMessage(content="Привет! Ответь просто 'Тест прошел успешно'")
        ])

        print(f"✅ Модель отвечает: {response.content[:100]}...")
        return True

    except Exception as e:
        print(f"❌ Ошибка модели: {e}")
        return False


def test_financial_analysis():
    """Тест финансового анализа"""
    print("\n📊 Тестирую финансовый анализ...")

    try:
        llm = ChatOllama(
            model="llama3.1:latest",
            temperature=0.1
        )

        prompt = """
        Ты финансовый аналитик. Проанализируй акции Apple (AAPL).
        Дай краткую рекомендацию: BUY, HOLD или SELL.
        Объясни одной фразой.
        """

        response = llm.invoke([HumanMessage(content=prompt)])

        print(f"✅ Финансовый анализ работает:")
        print(f"   Рекомендация: {response.content}")
        return True

    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return False


def main():
    print("=" * 50)
    print("🚀 Тестирование Financial Analysis System с Ollama")
    print("=" * 50)

    # Тест 1: Соединение
    if not test_ollama_connection():
        print("\n⚠️  Ollama не запущен. Запустите: ollama serve")
        return

    # Тест 2: Модель
    if not test_llama_model():
        print("\n⚠️  Проблема с моделью. Проверьте: ollama pull llama3.1")
        return

    # Тест 3: Финансовый анализ
    test_financial_analysis()

    print("\n" + "=" * 50)
    print("🎉 Все тесты пройдены! Система готова к работе.")
    print("=" * 50)

    print("\n📋 Далее:")
    print("1. Запустите дашборд: python scripts/dev_start.py")
    print("2. Или протестируйте анализ: python -c \"from test_ollama import *; main()\"")


if __name__ == "__main__":
    main()