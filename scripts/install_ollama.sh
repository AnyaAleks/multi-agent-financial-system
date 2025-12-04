#!/bin/bash
# Скрипт установки и настройки Ollama

echo "========================================="
echo "🚀 Установка Ollama для Financial Analysis System"
echo "========================================="

# Проверка операционной системы
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🔍 Обнаружена Linux система"

    # Установка Ollama
    echo "📥 Устанавливаю Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🔍 Обнаружена macOS система"

    # Установка через Homebrew
    if command -v brew &> /dev/null; then
        echo "📥 Устанавливаю Ollama через Homebrew..."
        brew install ollama
    else
        echo "❌ Homebrew не найден. Установите Homebrew или скачайте Ollama с https://ollama.ai"
        exit 1
    fi

else
    echo "❌ Неподдерживаемая операционная система: $OSTYPE"
    echo "📥 Скачайте Ollama с https://ollama.ai"
    exit 1
fi

# Запуск Ollama
echo "🚀 Запускаю Ollama сервер..."
ollama serve &
OLLAMA_PID=$!

# Ждем запуска сервера
echo "⏳ Жду запуска Ollama сервера..."
sleep 5

# Проверка запуска
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama сервер запущен успешно"
else
    echo "❌ Не удалось запустить Ollama сервер"
    exit 1
fi

# Загрузка моделей
echo "📥 Загружаю модели..."
models=("llama3.1:latest" "mistral:latest" "phi:latest")

for model in "${models[@]}"; do
    echo "📦 Загружаю модель: $model"
    ollama pull $model

    if [ $? -eq 0 ]; then
        echo "✅ Модель $model загружена успешно"
    else
        echo "⚠ Не удалось загрузить модель $model"
    fi
done

echo ""
echo "========================================="
echo "🎉 Ollama успешно установлен и настроен!"
echo "========================================="
echo ""
echo "Доступные команды:"
echo "  ollama serve          - Запустить сервер Ollama"
echo "  ollama list           - Показать установленные модели"
echo "  ollama run llama3.1   - Запустить модель"
echo ""
echo "Для использования с Financial Analysis System:"
echo "1. Убедитесь что Ollama сервер запущен"
echo "2. В файле .env установите:"
echo "   AI_PROVIDER=ollama"
echo "   OLLAMA_MODEL=llama3.1:latest"
echo ""
echo "PID Ollama сервера: $OLLAMA_PID"
echo "Для остановки: kill $OLLAMA_PID"