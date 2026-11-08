# NeptuneAI — русскоязычная модерация

Компактный классификатор одного сообщения: возвращает ровно `SAFE` или `VIOLATION`, без длинного ответа и без истории соседних сообщений. Основа — `cointegrated/rubert-tiny2` (около 29 млн параметров); GitHub Actions дообучает её и сохраняет готовые веса.

**Точность 100% гарантировать нельзя.** Модель может пропустить редкую завуалированную форму или ошибиться на цитате, шутке и контексте. `SAFE` — результат классификации, не абсолютная гарантия. Скрипт выбирает порог по калибровочной части с целевым precision 0.95 и независимо измеряет метрики на тестовой части.

## Источники Hugging Face

Workflow проверяет схему и скачивает настоящие русскоязычные датасеты:

- [`Mnwa/russian-toxic`](https://huggingface.co/datasets/Mnwa/russian-toxic) — крупный корпус с метками `negative/positive`;
- [`AlexSham/Toxic_Russian_Comments`](https://huggingface.co/datasets/AlexSham/Toxic_Russian_Comments) — бинарная разметка комментариев;
- [`dbrovkin/toxic-russian-comments-multilabel`](https://huggingface.co/datasets/dbrovkin/toxic-russian-comments-multilabel) — метки profanity/threat/illegal;
- [`s-nlp/ru_paradetox`](https://huggingface.co/datasets/s-nlp/ru_paradetox) — пары: исходный токсичный текст → `VIOLATION`, нейтральная версия → `SAFE`;
- [`textdetox/multilingual_toxicity_dataset`](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset) — русская часть бинарного корпуса.

Примеры не сочиняются для обучения. Тексты из разных источников нормализуются только для сравнения, дубликаты удаляются, а одинаковый текст с противоречивыми метками исключается. Для ограниченного времени Action берёт случайную выборку до 20 000 примеров **на класс из каждого корпуса** (настраивается в форме запуска), а не копирует все сотни тысяч строк. `ilyiniv1755/toxic-russian-dataset` не загружается отдельно: по схеме, размеру и видимым строкам он дублирует `AlexSham/Toxic_Russian_Comments`, поэтому повторная загрузка увеличила бы повторы.

Список из запроса не используется вслепую: некоторые ID моделей/наборов не разрешаются как существующие страницы, у разных наборов несовместимые метки, а несколько корпусов — зеркала друг друга. Workflow использует только источники с проверенными страницами и ожидаемой схемой; если схема на HF изменится, обучение завершится с ошибкой, а не угадает классы.

## Обучить через GitHub Actions

1. Откройте **Actions → Train Russian moderation model → Run workflow**.
2. Оставьте `2` эпохи и `20000` примеров на класс на корпус (или выберите больший объём; обучение может идти дольше).
3. После завершения скачайте артефакт `neptune-ru-toxic-model-<run number>` из страницы запуска. Он содержит веса, токенизатор, порог, метрики и отчёт smoke-test.
4. Распакуйте его как `artifacts/neptune-ru-toxic` или задайте `MODEL_DIR` на распакованную папку.

Action также проверяет четыре контрольных фразы владельца, включая «Пошел нахуй ебаный петух», а также безобидное сообщение. Эти фразы используются **только как smoke-test после обучения**, не добавляются в train data и не подменяют датасеты. В `smoke_test.json` записывается фактический ответ модели; несовпадение видно в логе, но не скрывается.

Артефакт GitHub хранится 30 дней. Чтобы Action дополнительно опубликовал модель в Hugging Face, задайте в репозитории variable `HF_MODEL_REPO` (например `my-user/neptune-ru-toxic`) и secret `HF_TOKEN` с правом записи в этот model repo. Без секрета модель всё равно будет приложена к запуску как artifact.

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: use the venv activation script for your shell
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m training.train --output-dir artifacts/neptune-ru-toxic
MODEL_DIR=artifacts/neptune-ru-toxic uvicorn app:app --host 0.0.0.0 --port 8000
```

Откройте `http://localhost:8000`. API принимает только одно поле `text` и отвечает обычным текстом `SAFE` или `VIOLATION`:

```bash
curl -X POST http://localhost:8000/api/classify -H 'Content-Type: application/json' -d '{"text":"Ты сегодня отлично справился"}'
```

Запросы независимы: приложение не хранит историю и ничего не склеивает с соседними сообщениями. Пустой текст и тексты больше 20 000 символов отклоняются. Пока обученной модели нет, API возвращает `503` — никаких фиктивных предсказаний.

## Для будущего Minecraft-плагина

Модель — обычный локальный `transformers` классификатор; серверный endpoint `/api/classify` можно вызывать из Java по HTTP. Здесь нет генеративного LLM и chain-of-thought: один forward-pass выдаёт бинарную метку. Позже можно завернуть веса в ONNX/Java runtime или оставить отдельный локальный сервис; подробные reasoning-объяснения не следует выдумывать из бинарного классификатора.
