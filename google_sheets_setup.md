# Как подключить к Google Sheets

## Вариант через Streamlit

1. Сначала запусти Streamlit app через GitHub + Streamlit Cloud.
2. Google Sheets можно использовать как интерфейс вручную:
   - вводишь тикер в таблице;
   - открываешь Streamlit dashboard;
   - выгружаешь CSV;
   - импортируешь CSV в Google Sheets.

## Более автоматический вариант

Для прямого автообновления из Google Sheets нужен API endpoint, например на Render/FastAPI.
Streamlit сам по себе не является удобным API endpoint для Apps Script.

Минимальная архитектура:

Google Sheets → Apps Script → FastAPI endpoint → options_engine.py → Yahoo Finance

Это можно добавить следующим этапом.