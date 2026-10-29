# POI recognition — код сервиса после сброса моделей

28 октября 2026 года прежний bundle и результаты моделей удалены.
Исходный код принимает фотографию и возвращает POI, но для запуска необходим новый выпуск модели.
Имя `models/poi_service_dinov2` пока остаётся техническим путём в шаблоне Dockerfile,
а не обозначением выбранной текущей модели.

## Запуск

Из корня репозитория:

```powershell
# Сначала подготовьте новый bundle: прежний models/poi_service_dinov2 удалён.
docker compose up --build -d
docker compose ps
docker compose logs -f poi-api
```

Форма загрузки фото и ответ на той же странице: http://localhost:8000/.
Выберите файл и нажмите «Распознать POI»: появятся предпросмотр, решение, POI, значения сходства и полный JSON.
При отказе предположение модели явно отделено от подтверждения. Страница работает через существующий `/predict`, без внешних CDN.
После изменения кода пересоберите контейнер: `docker compose up --build -d --force-recreate`.
Swagger API: http://localhost:8000/docs. Проверка готовности: http://localhost:8000/health/ready.

```powershell
curl.exe -X POST "http://localhost:8000/predict?top_k=3" -F "file=@C:/photos/photo.jpg"
curl.exe http://localhost:8000/pois
docker compose down
```

По умолчанию порт опубликован на `127.0.0.1:8000`. `POI_PORT` меняет host-порт.
Например, `$env:POI_PORT="8080"` перед `docker compose up`.
Bundle копируется в образ при сборке. Bind mount с диска Windows не требуется;
это устраняет ошибку `invalid mount config ... /run/desktop/mnt/host/i/...: input/output error`.
Имя Compose-проекта явно задано `nevidal-poi`, поэтому кириллица в имени каталога не делает имя проекта пустым.

## Контракт

- `POST /predict`: multipart/form-data, обязательное поле `file`; query `top_k=1..10`, по умолчанию 3.
- `GET /pois`: доступные `poi_id` и фиксированные `class_index`.
- `GET /health/live`: процесс отвечает.
- `GET /health/ready` и `/health`: модель загружена; версия и число POI.

Ключевые поля ответа, сокращённый пример:

```json
{
  "poi_id": "moscow/ostankinskiy/cosmos",
  "predicted_poi_id": "moscow/ostankinskiy/cosmos",
  "accepted": true,
  "reason": "poi_verified",
  "score": 0.85,
  "margin": 0.24,
  "score_type": "cosine_similarity",
  "score_is_probability": false,
  "encoder": "dinov2_small",
  "model_version": "72b195ca09176a2df8a7"
}
```

`predicted_poi_id` всегда содержит наиболее похожий известный POI. `poi_id` содержит эту метку только при
прохождении score/margin-порогов; при отказе `poi_id=null`, `accepted=false`. Cosine score не является вероятностью.
Поэтому клиент может показывать предположение, а для подтверждения задания использовать `accepted`.
Положительное решение означает прохождение порогов модели, а не гарантированную истинность метки.

Полный ответ также содержит top-k `candidates`, пороги и ближайший эталон с `source`, `author`, `licence`, URL.
Файловые пути сервера не возвращаются. Класса unknown не создаётся.

Причины решения: `poi_verified`, `below_score_threshold`, `ambiguous_poi`, `tie`.
Коды ошибок: 413 — превышен размер; 415 — неподдерживаемый формат; 422 — повреждённое/пустое фото или неверные параметры;
503 `service_busy` — достигнут лимит одновременно обрабатываемых загрузок. Нечитаемое фото не получает POI-метку.

## Лимиты и хранение

По умолчанию: JPEG/PNG/WebP и первый кадр MPO; файл до 10 MiB, изображение до 40 млн пикселей.
Анимированный WebP отклоняется. Проверяется фактический формат, а не расширение имени или присланный MIME.
HTTP-body ограничен размером файла плюс 64 KiB для multipart, в том числе при chunked-передаче.
Допускаются две активные загрузки; inference выполняется последовательно, чтобы ограничить CPU/RAM.

Фотографии не добавляются в датасет и не записываются в постоянное хранилище или логи приложения.
Multipart parser может использовать временный файл: в контейнере `/tmp` — tmpfs, файлы закрываются после запроса.
Сервис работает непривилегированным пользователем с read-only filesystem; модель находится внутри образа.
В build context входят inference-код и семь файлов экспортированного bundle (веса, векторы и метаданные).
Исходные фотографии, processed-копии и EDA исключены через `.dockerignore` и контейнеру не нужны.
Веса загружаются один раз при startup, SHA-256 всех артефактов проверяются. При ошибке startup прекращается;
автоматических скачиваний и случайных весов нет.

Переменные окружения сервиса: `MODEL_DIR`, `MAX_UPLOAD_MB`, `MAX_IMAGE_PIXELS`, `CPU_THREADS`, `MAX_INFLIGHT_REQUESTS`.
Изменяйте лимиты в `compose.yaml`. Для стандартного запуска используется один Uvicorn worker.

## Подготовка другого bundle

Экспорт требует готовый выпуск `models/encoder_comparison_manual_all`. Существующий bundle не перезаписывается.

```powershell
python scripts/export_poi_service_bundle.py --output models/poi_service_dinov2_v2
# Можно явно экспортировать другой encoder того же сравнения:
python scripts/export_poi_service_bundle.py --encoder mobilenetv3_small --output models/poi_service_small
```

Для другого bundle измените источник `COPY ... /models/` в Dockerfile и соответствующий whitelist в `.dockerignore`,
затем выполните `docker compose up --build -d --force-recreate`. Для обновления стандартного bundle достаточно пересборки.
Bundle содержит веса, gallery vectors/CSV,
class mapping, policy, provenance и manifest SHA. Исходные фото в него не входят; авторство и лицензии эталонов сохраняются.

## Локальная проверка без Docker

```powershell
python -m pip install -r requirements-service-test.txt
python -m pytest tests/test_poi_service.py -q
python scripts/smoke_poi_service.py
python -m uvicorn service.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Для окружения с рабочим Python и вторичными site-packages smoke/export поддерживают
`--extra-site .venv/Lib/site-packages`. Полный результат smoke — `reports/poi_service/smoke_checks.json`:
положительный и отрицательный HTTP-запросы сопоставляются с сохранёнными scores эксперимента.

## Качество нового выпуска

Результаты предыдущих моделей удалены. После нового обучения и экспорта требуется заново
измерить качество и проверить совпадение подготовки изображений и предсказаний в Python и сервисе.

Реализация загрузки и контейнера следует [FastAPI UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/),
[FastAPI Docker](https://fastapi.tiangolo.com/deployment/docker/) и [Docker Compose services](https://docs.docker.com/reference/compose-file/services/).
