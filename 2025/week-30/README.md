# Математические основы машинного обучения

**Математические основы машинного обучения** - это [Jupyter Book](https://jupyterbook.org/en/stable/intro.html), содержащий Jupyter ноутбуки, которые реализуют и математически выводят фундаментальные концепции математического анализа, лежащие в основе алгоритмов машинного обучения.

Каждый ноутбук содержит интерактивные визуализации и подробные математические выводы, помогающие понять, как математические концепции применяются в машинном обучении.

Результат каждого ноутбука - это визуализация математических концепций с практическими примерами и приложениями в области машинного обучения.

Этот репозиторий является частью еженедельного обзора arXiv исследований в области ML и AI. На высоком уровне Jupyter Books позволяют создавать веб-сайты с Markdown файлами и Jupyter ноутбуками. Обратите внимание, что Jupyter ноутбуки находятся непосредственно в этом репозитории. Веб-сайт обновляется с помощью GitHub Action в `.github/workflows/ci.yml` после каждого коммита или pull request. Для локальной сборки веб-сайта см. раздел "Использование" ниже.

## Jupyter Ноутбуки

### Глава 0: Математические основы

- [**Производные и их применение**](chapter-0/derivative.ipynb) - Фундаментальные концепции производных, их геометрическая интерпретация и применение в оптимизации

## Информация о Jupyter Book

Оглавление и структура книги указаны в файле `_toc.yml`.

Конфигурация указана в файле `_config.yml`.

Для получения дополнительной информации см. [Jupyter Book Docs](https://jupyterbook.org/en/stable/intro.html).

## Использование

### Шаг 1: Клонирование репозитория

```sh
git clone https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review.git
cd Weekly-arXiv-ML-AI-Research-Review/2025/week-30/machine-learning-visualized
```

### Шаг 2: Сборка Jupyter Book

#### Вариант 1: jupyter-book CLI

```sh
pip install -U jupyter-book
jupyter-book build .
```

#### Вариант 2: Docker Compose

```sh
docker compose up
docker compose down --volumes --rmi local
```

#### Вариант 3: Docker

```sh
docker build -t jupyter-book .
docker run --rm -v "$(pwd)":/usr/src/app jupyter-book

docker stop jupyter-book
docker rm jupyter-book
docker rmi jupyter-book
```

### Шаг 3: Открытие Jupyter Book

Перейдите к `_build/html/index.html`

## Результат

### Интерактивные Jupyter Ноутбуки

![Jupyter](gifs/marimo.gif)

### Математически объяснено

![latex](gifs/latex.gif)

## Структура проекта

```
machine-learning-visualized/
├── _config.yml          # Конфигурация Jupyter Book
├── _toc.yml            # Структура оглавления
├── index.md            # Главная страница
├── chapter-0/          # Глава 0: Математические основы
│   └── derivative.ipynb # Ноутбук с производными
├── _static/            # Статические файлы (CSS)
├── gifs/              # Анимации и GIF
├── compose.yml        # Docker Compose конфигурация
├── Dockerfile         # Docker образ
└── README.md          # Этот файл
```


