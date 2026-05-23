# Сервис прогнозирования дефолта по кредитным картам

Готовый к продакшену ML-сервис для прогнозирования дефолта по кредитным картам на основе
[датасета UCI Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).
Реализован на Flask, scikit-learn, Docker с поддержкой A/B-тестирования.

---

## Содержание

1. [Обзор проекта](#обзор-проекта)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт (локально)](#быстрый-старт-локально)
4. [Развёртывание в Docker](#развёртывание-в-docker)
5. [Справочник по API](#справочник-по-api)
6. [A/B-тестирование](#ab-тестирование)
7. [Архитектура](#архитектура)
8. [Docker Hub](#docker-hub)

---

## Обзор проекта

**Область:** Финансы / Кредитный скоринг  
**Цель:** Предсказать, допустит ли клиент дефолт по платежу в следующем месяце (`1 = дефолт`, `0 = нет дефолта`).

| Модель | Алгоритм | Примечание |
|--------|----------|------------|
| v1 | LogisticRegression | Базовая — интерпретируемая, быстрая |
| v2 | GradientBoostingClassifier | Улучшенный захват нелинейных зависимостей |

---

## Структура репозитория

```
credit-card-ml-deployment/
├── app/
│   ├── __init__.py
│   ├── api.py              # Flask-приложение (эндпоинты)
│   └── model_handler.py   # Загрузка модели и инференс
├── models/
│   ├── train_model.py      # Скрипт обучения
│   ├── model_v1.pkl        # Сохранённая модель v1
│   ├── model_v2.pkl        # Сохранённая модель v2
│   └── metadata.json       # Метрики моделей
├── tests/
│   └── test_api.py         # Тест-сьют на pytest (10 тестов)
├── docker/
│   ├── Dockerfile
│   └── nginx.conf
├── docs/
│   └── ARCHITECTURE.md     # Архитектурные решения
├── data/                   # Поместите UCI_Credit_Card.csv сюда
├── ab_test_plan.md         # План A/B-теста
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Быстрый старт (локально)

### 1. Клонирование и настройка окружения

```bash
git clone https://github.com/yaroslav775507/credit-card-ml-deployment.git
cd credit-card-ml-deployment

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Опционально) Добавьте реальный датасет

Скачайте [UCI_Credit_Card.csv](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
и поместите файл в `data/UCI_Credit_Card.csv`.

### 3. Обучение моделей

```bash
python models/train_model.py
```

### 4. Запуск сервиса

```bash
python app/api.py
# Сервис доступен по адресу http://localhost:5000
```

### 5. Запуск тестов

```bash
pytest tests/test_api.py -v
```

---

## Развёртывание в Docker

### Сборка и запуск

```bash
# Сборка образа
docker build -f docker/Dockerfile -t credit-card-default-predictor .

# Запуск контейнера
docker run -p 5000:5000 credit-card-default-predictor
```

### Docker Compose (с NGINX и монитором логов)

```bash
docker-compose up --build
# Сервис доступен по адресу http://localhost (через NGINX, порт 80)
# Прямой доступ:           http://localhost:5000
```

### Получить образ с Docker Hub

```bash
docker pull redyar/credit-card-default-predictor:latest
docker run -p 5000:5000 redyar/credit-card-default-predictor:latest
```

---

## Справочник по API

### `GET /health`

Проверка работоспособности для Docker и балансировщиков нагрузки.

```bash
curl http://localhost:5000/health
```

```json
{
  "status": "healthy",
  "service": "credit-card-default-predictor",
  "timestamp": "2024-11-15T09:23:11Z"
}
```

---

### `POST /predict`

Прогноз дефолта для одного клиента.

**Query-параметры:**

| Параметр | Значения | По умолчанию | Описание |
|----------|----------|--------------|----------|
| `version` | `v1`, `v2`, `ab` | `ab` | Версия модели. `ab` = случайная маршрутизация 50/50 |

**Тело запроса:**

```json
{
  "LIMIT_BAL":  50000,
  "SEX":        2,
  "EDUCATION":  2,
  "MARRIAGE":   1,
  "AGE":        35,
  "PAY_0":      0,
  "PAY_2":      0,
  "PAY_3":      0,
  "PAY_4":      0,
  "PAY_5":      0,
  "PAY_6":      0,
  "BILL_AMT1":  15000,
  "BILL_AMT2":  14000,
  "BILL_AMT3":  13000,
  "BILL_AMT4":  12000,
  "BILL_AMT5":  11000,
  "BILL_AMT6":  10000,
  "PAY_AMT1":   1500,
  "PAY_AMT2":   1400,
  "PAY_AMT3":   1300,
  "PAY_AMT4":   1200,
  "PAY_AMT5":   1100,
  "PAY_AMT6":   1000
}
```

**Ответ:**

```json
{
  "prediction":    0,
  "probability":   0.2786,
  "model_version": "v1",
  "risk_label":    "LOW"
}
```

**Примеры curl:**

```bash
# Модель v1 (явное указание)
curl -X POST "http://localhost:5000/predict?version=v1" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":50000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":35,
       "PAY_0":0,"PAY_2":0,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":0,
       "BILL_AMT1":15000,"BILL_AMT2":14000,"BILL_AMT3":13000,
       "BILL_AMT4":12000,"BILL_AMT5":11000,"BILL_AMT6":10000,
       "PAY_AMT1":1500,"PAY_AMT2":1400,"PAY_AMT3":1300,
       "PAY_AMT4":1200,"PAY_AMT5":1100,"PAY_AMT6":1000}'

# Модель v2 (явное указание)
curl -X POST "http://localhost:5000/predict?version=v2" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":50000,"AGE":35,"PAY_0":2,"BILL_AMT1":40000}'

# A/B-маршрутизация (50% — v1, 50% — v2)
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":50000,"AGE":35}'
```

---

### `GET /models`

Список доступных моделей и их метрики обучения.

```bash
curl http://localhost:5000/models
```

```json
{
  "available_versions": ["v1", "v2"],
  "metadata": {
    "v1": {"algorithm": "LogisticRegression",         "f1_score": 0.4613},
    "v2": {"algorithm": "GradientBoostingClassifier", "f1_score": 0.4688}
  },
  "ab_split_v2": 0.5
}
```

---

### `GET /features`

Возвращает список ожидаемых входных признаков.

```bash
curl http://localhost:5000/features
```

---

## A/B-тестирование

Полный план см. в файле [`ab_test_plan.md`](ab_test_plan.md).

**Краткое резюме:**
- **Контроль (v1):** LogisticRegression — 50% трафика
- **Эксперимент (v2):** GradientBoosting — 50% трафика
- **Основная метрика:** F1-score (класс дефолта)
- **Вторичная метрика:** Recall (минимизация пропущенных дефолтов)
- **Бизнес-метрики:** снижение ожидаемых потерь, уровень одобрений
- **Критерий успеха:** `ΔF1 ≥ 0.03`, `p-value < 0.05`, без регрессии по Recall
- **Длительность:** минимум 4 недели / 1 000 образцов на группу

---

## Архитектура

См. файл [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md):
- Монолит vs микросервисы
- Концепция асинхронной обработки на RabbitMQ
- Стек логирования ELK
- Инструменты MLOps: DVC + MLflow
- Экспорт в ONNX-ML
- Продакшен-стек: uWSGI + NGINX

---

## Docker Hub

```
docker pull redyar/credit-card-default-predictor:latest
```

Ссылка на образ: https://hub.docker.com/r/redyar/credit-card-default-predictor