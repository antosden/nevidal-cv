# «Невидаль»: сбор фотографий Wikimedia Commons

## Docker-сервис распознавания POI

Подготовлен HTTP-сервис на DINOv2-S с выбранной галереей и порогами отказа. Bundle уже находится в
`models/poi_service_dinov2` и копируется в образ при сборке. Bind mounts с диска Windows не используются;
исходные фотографии контейнеру не нужны.

```powershell
docker compose up --build -d
curl.exe -X POST "http://localhost:8000/predict" -F "file=@C:/photos/photo.jpg"
```

Форма загрузки с предпросмотром и результатом на странице: http://localhost:8000/.
Документация Swagger: http://localhost:8000/docs. `predicted_poi_id` — предположение; `poi_id` заполняется
при `accepted=true`, иначе равен `null`. Score — cosine similarity. Модель экспериментальная: test true accept 46,77%,
external FAR 1,05%. [Контракт, настройки и проверки](docs/poi_service.md).

В проекте находятся Wikimedia-парсер, EDA-пайплайн подготовки размеченного датасета и baseline верификации POI на frozen MobileNetV3. Веса модели не дообучаются. Анализ охватывает все найденные POI и районы; прежний архив и папка `ВДНХ/` в него не включаются.

## Baseline: MobileNetV3 → эмбеддинги → галерея → отказ

Основной документ — `notebooks/02_poi_verification_baseline.ipynb`. Новый baseline использует **MobileNetV3-Small, IMAGENET1K_V1**, 576D avgpool-эмбеддинги с L2-нормализацией. Прежние classifiers, веса дообученных моделей и пороги из архива не используются. Сохранённый официальный ImageNet-checkpoint допускается повторно использовать после проверки полного SHA-256.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt
.\.venv\Scripts\python.exe scripts/baseline_mobilenetv3.py run --download-weights
.\.venv\Scripts\python.exe -m jupyterlab notebooks/02_poi_verification_baseline.ipynb
```

Скачивание исходных весов (~10 MB) нужно только один раз; `--download-weights` действует только при отсутствии файла. При доступных весах запуск полностью offline:

```powershell
.\.venv\Scripts\python.exe scripts/baseline_mobilenetv3.py run --weights-file .runtime/torch/checkpoints/mobilenet_v3_small-047dcff4.pth
.\.venv\Scripts\python.exe scripts/execute_baseline_notebook.py
```

При известной ошибке pandas в `.venv` используйте рабочий Python с pandas и параметр `--extra-site .venv/Lib/site-packages` **перед** подкомандой `run`/`verify`. Параметр также есть у исполнителя ноутбука. Он не меняет данные или результаты модели. Фактические версии библиотек записываются в metadata.json; полный локальный snapshot пакетов — `reports/baseline_mobilenetv3/environment_freeze.txt`.

Выпуск сохраняется в `models/baseline_mobilenetv3/`: исходные веса, train/val/test snapshots с атрибуцией, эмбеддинги и их индексы, `gallery.csv`, `gallery_embeddings.npy`, `rejection_policy.json`, `metadata.json`, результаты и таблицы ошибок. Скрипт не перезаписывает существующий выпуск: для нового запуска передайте `--output models/baseline_mobilenetv3_run2`. Ноутбук при повторном Run All проверяет хэши и читает зафиксированные результаты. При изменении данных/конфигурации нужен новый выпуск.

Правила фиксируются в `config/baseline_mobilenetv3.json`, SEED=42. Галерея — до 20 разнообразных эталонов на POI из **train**, максимум один кадр split_group на POI. Проверяются пересечения ID, SHA-256, duplicate_group, split_group, series_id и Commons file ID. До обработки сверяется SHA-256 каждого используемого фото.

Для зафиксированной CPU-среды используйте `requirements-baseline-lock.txt` (Python 3.12.14); общий файл зависимостей допускает совместимые версии. Независимая проверка сохранённого выпуска: `python scripts/check_baseline_reproducibility.py`. Она повторно строит галерею и val-пороги из сохранённых эмбеддингов, сверяет test-метрики и повторно извлекает эмбеддинги нескольких фото. Этот запуск не меняет пороги и не дообучает модель.

`score(target)` — максимальная cosine similarity с эталонами цели. `margin` — отрыв от лучшего другого POI. Подтверждение требует прохождения обоих порогов и строго положительного margin; равенство лидеров даёт отказ. Cosine **не является вероятностью**. Пороги выбираются только на **val**: максимум true accept при эмпирическом known-wrong FAR ≤1% и hard-wrong FAR ≤5%. Known-wrong перебирает все чужие известные POI, hard-wrong берёт наиболее похожую неверную цель каждого фото. Пары одного фото зависимы; количество пар не равно числу независимых примеров. После этого test используется для оценки, без подбора параметров.

Локальная проверка одной фотографии:

```powershell
.\.venv\Scripts\python.exe scripts/baseline_mobilenetv3.py verify --image path/to/photo.jpg --target-poi-id moscow/ostankinskiy/ostankino_tower
```

Результат содержит `accepted`, `reason`, `score`, `margin`, ближайший эталон с source/author/licence и версии галереи/порогов. Причины отказа: `poi_not_matched` (другая цель лидирует или ничья), `poi_below_threshold`, `poi_ambiguous`, `target_poi_not_supported`. Это локальный Python/CLI baseline; сервер или API не создаётся. Программный интерфейс — `src.poi_baseline.POIVerifier`.

Метрики и графики находятся в `reports/baseline_mobilenetv3/summary.md`; детальные пары/ошибки — в `models/baseline_mobilenetv3/test_pairs.csv`. **Внешних unknown-негативов пока нет:** отказ на произвольных объектах не проверен. Нужны отдельные проверенные unknown-val/unknown-test и полевой мобильный тест. Малые классы делают текущие оценки нестабильными. Для авторства и лицензии сохраняются исходные поля; manual — Shilkova A.M., Public Domain по подтверждению пользователя.

## EDA и подготовка разметки

### Полное обогащение manual и сравнение encoder

В текущем выпуске учтены все 784 manual-фото, включая 288 ранее не использовавшихся. После очистки выпущены 1 263 изображения 23 POI: train=886, val=191, test=186. Семь кадров с противоречащими POI-метками исключены и находятся в `data/annotations/review_conflicts.csv`; contact sheet — `reports/figures/label_conflicts.jpg`.

Проверены 21 validation-вариант галерей. Выбранный по validation DINOv2-S (`random_20`, 404 эталона) на диагностическом test дал accuracy 89,25%, macro-F1 89,22%, true accept 46,77%, external any-POI FAR 1,05% (6/573). У Small — true accept 34,95%, FAR 0,87%; у Large — 38,71%, FAR 2,44%. Цель 85% true accept не достигнута; FAR DINOv2 немного выше ограничения 1%, основная модель автоматически не заменена. Полная таблица — `reports/encoder_comparison/summary.md`.

Текущая политика `config/dataset_sources.json`: `manual_selection=all`. В датасет включаются все доступные изображения из зарегистрированных папок manual, независимо от числа Wikimedia-фото. Автор manual — Shilkova A.M., licence — Public Domain по подтверждению пользователя. Неизменённые решения review_edits.csv сохраняются; нечитаемые, точные копии и конфликтующие кадры остаются в audit-таблицах, но исключаются из активных выборок.

`data/annotations/split_anchors.csv` хранит прежние назначения train/val/test и SHA исходников. Новые независимые группы распределяются с SEED=42 к пропорциям 70/15/15. Группы, связывающие разные прежние splits, изолируются с причиной `cross_split_anchor_conflict`; потенциальные связи с фиксированными external negatives — `external_negative_overlap_candidate`. Это исключение из выпуска, а не автоматическое изменение подтверждённой POI-метки. Старые processed-копии могут оставаться на диске: актуальный состав определяется CSV, загрузчики не обходят папки через glob.

Сравнение — `notebooks/07_encoder_gallery_comparison.ipynb`, настройки — `config/encoder_comparison.json`. Три frozen encoder: MobileNetV3-Small V1, MobileNetV3-Large V2, DINOv2 ViT-S/14 без register tokens. Для всех используются L2 embeddings, max cosine и глобальные score/margin пороги. Галереи: 5/10/20/all независимых train-групп на POI; diverse/farthest-first и случайный отбор с SEED=42. У каждого encoder документированный preprocessing до crop 224; DINOv2 использует CLS 384D. Обучение весов не выполняется.

Официальные веса положите в `.runtime/torch/checkpoints/`:

- [mobilenet_v3_large-5c1a4163.pth](https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth)
- [dinov2_vits14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)

Small уже доступен локально. При доступной сети можно скачать оба недостающих файла командой `python scripts/download_encoder_weights.py`; сравнение автоматически скачивание не запускает. Для DINOv2 реализована eval-only совместимая архитектура на стандартном PyTorch: официальный state_dict загружается strict=True, без случайного fallback. Полные SHA, preprocessing и версии сохраняются. Новые зависимости не требуются.

```powershell
python scripts/compare_encoders.py --only-encoder mobilenetv3_small
python scripts/compare_encoders.py --only-encoder mobilenetv3_large --stage validation --reuse-input-audit
python scripts/compare_encoders.py --only-encoder mobilenetv3_large --stage test --reuse-input-audit
python scripts/compare_encoders.py --only-encoder dinov2_small --stage validation --reuse-input-audit
python scripts/compare_encoders.py --only-encoder dinov2_small --stage test --reuse-input-audit
python scripts/execute_baseline_notebook.py --notebook notebooks/07_encoder_gallery_comparison.ipynb
python scripts/check_encoder_comparison.py
```

Используйте рабочий Python с pandas; команды поддерживают `--extra-site .venv/Lib/site-packages`. Можно запустить всё одной командой без фильтров; этапы выше удобнее для длительных CPU-запусков. Эмбеддинги сохраняются после каждого батча с идентификатором входных данных и весов; продолжение не повторяет готовые батчи. Готовые кэши проверяются по SHA. `--stage test` требует ранее зафиксированного validation-выбора.

Первый запуск выполняет полный SHA-аудит входных фото. `--reuse-input-audit` допустим только с теми же CSV/config: для файлов с прежним размером и mtime не позже завершённого аудита используется его результат, остальные хэшируются заново. Это кэш по метаданным файловой системы, а не новая криптографическая проверка; для полной повторной проверки опустите флаг.

Отсутствующие веса дают `missing_weights`, а не вымышленные метрики; после добавления весов повторите сравнение, завершённые encoder не пересчитываются. Результаты — `models/encoder_comparison_manual_all/` и `reports/encoder_comparison/summary.md`; дополнительная диагностика по источникам — `by_source.csv`. Новый состав данных или настройки требуют нового `--output` (пути OUTPUT/report в ноутбуке и проверяющем скрипте для такого выпуска следует изменить). Скрипт `refresh_encoder_comparison_metrics.py` пересобирает отчётные поля из фиксированных scores, сохраняя прежние отчёты в .runtime; численные метрики, выбор и пороги он не меняет.

Выбор галереи/порогов и предпочтительного encoder выполняется по val/negative-val при эмпирическом FAR ≤1% на val. Test обрабатывается после фиксации выбора каждого encoder и служит диагностикой; пороги по нему не меняются. Прежние выпуски моделей относятся к прежним snapshots датасета, поэтому сравнивать влияние encoder следует внутри нового выпуска. Для окончательного подтверждения качества нужен новый закрытый полевой test. Автоматического переключения используемой модели нет.

### Геометрическая проверка конкретного POI

Проверенный результат этого эксперимента на test: accuracy **84,93%**, macro-F1 **84,70%**, true accept **30,82% (45/146)**, external any-POI FAR **1,05% (6/573)**. У прежнего строгого baseline true accept **25,34%**, FAR **0,87%**. Новый вариант не заменяет прежний: ограничение FAR ≤1% на test не выполнено; цель 85% true accept не достигнута. Выбран `best + hybrid_ridge` только по validation; test-пороги не подстраивались. Этот test уже использовался для диагностики предыдущих экспериментов, поэтому для окончательной оценки нужен новый закрытый полевой набор.

`notebooks/06_mobilenetv3_geometric_verification.ipynb` проверяет локальные детали объекта. MobileNetV3-Small + обученный Ridge предлагает 5 POI; для каждого берутся 3 ближайших эталона из прежней **train-галереи**. SIFT на grayscale копии ≤960 px, FLANN ratio test 0.72 и RANSAC homography проверяют согласованность деталей. Для геометрической поддержки нужны ≥8 inliers и площадь convex hull ≥2% каждого кадра. Два query keypoint не голосуют за один ref keypoint. Исходные фото остаются неизменными.

```powershell
.\.venv\Scripts\python.exe scripts/geometric_poi.py
.\.venv\Scripts\python.exe scripts/execute_baseline_notebook.py --notebook notebooks/06_mobilenetv3_geometric_verification.ipynb
.\.venv\Scripts\python.exe scripts/check_geometric_model.py
```

Настройки — `config/mobilenetv3_geometric.json`; новые зависимости и скачивание весов не требуются. Исполнители поддерживают `--extra-site .venv/Lib/site-packages` для рабочего Python. Кэш с проверкой SHA-256 находится в `.runtime/geometric_features`. Подготовка локальных признаков больших Commons-оригиналов и сопоставление требуют нескольких минут CPU; последующие проверки используют кэш.

Этапы можно запускать отдельно: `scripts/geometric_poi.py --stage validate` фиксирует выбор по positive-val/negative-val, затем `--stage finalize` впервые вычисляет geometric test. Незавершённый validate можно продолжить с той же конфигурацией; завершённая validation-выборка не пересчитывается автоматически. Если дополнительно сравниваете новые способы fusion, делайте это по validation, до test; функция `select_geometry_policy` читает только сохранённые validation matching. Завершённый выпуск не перезаписывается; для нового эксперимента нужен новый `--output`.

Сравниваются четыре validation-варианта: максимум/среднее двух geometric scores × чистая геометрия/гибрид с Ridge. `geometry_only` не подтверждает без локальной поддержки (score floor 0.01). `hybrid_ridge` использует geometric score + непробабилистический Ridge-отклик; без локальных совпадений остаётся только строгий глобальный Ridge-fallback. Пороги score/margin общие для всех POI, с validation-ограничениями external any-POI FAR ≤1%, known hard-wrong FAR ≤5%. Test не участвует в выборе. Score не является вероятностью.

Выпуск — `models/mobilenetv3_geometric/`: веса исходного encoder, Ridge-параметры, train-галерея, локальные дескрипторы эталонов, snapshots, `selection.json`, политика отказа, `*_matching.npz`, `*_evidence.csv` и метрики. Модель самодостаточна для вывода: исходные train-фото не нужны после загрузки дескрипторов. Автор, источник и лицензия каждого эталона сохранены в gallery.csv. Признаки и match-таблицы не загружаются во внешние сервисы.

```powershell
.\.venv\Scripts\python.exe scripts/baseline_mobilenetv3.py verify --artifacts models/mobilenetv3_geometric --image path/to/photo.jpg --target-poi-id moscow/ostankinskiy/ostankino_tower
```

CLI выбирает загрузчик по schema_version=4. Ответ дополнен geometry_version и verification_method (`geometry_supported` или `strict_ridge_fallback`). Seed зависит от train-эталона, а не ID query, поэтому имя файла не меняет результат. Исследовательская схема дороже MobileNet по времени и памяти; в metadata приводится только matching latency, без декодирования, SIFT и encoder. Для применения на целевом оборудовании нужен полный latency-тест. Homography не гарантирует совпадения разных сторон здания, скульптур и движущейся воды. Положительный test уже диагностический; финальное качество нужно подтвердить новым закрытым полевым набором. Старые модели сохранены; автоматического переключения версии нет. Результаты — `reports/mobilenetv3_geometric/summary.md`.

### Улучшенный классификатор и экспериментальные per-POI пороги

Итог сравнения — `notebooks/05_mobilenetv3_classifier_and_calibration.ipynb`. Лучший на validation вариант использует исходный **MobileNetV3-Small ImageNet encoder + сбалансированный Ridge(alpha=0.1)**. Веса CNN остаются замороженными. Обучаемый линейный слой строится только по положительным train-эмбеддингам. Это отдельный scorer: его регрессионный отклик не является cosine или вероятностью и может выходить за [0,1]. Галерея сохраняется для контрольного варианта и ближайшего эталона с атрибуцией; при scorer=Ridge решение принимает обученный слой.

```powershell
.\.venv\Scripts\python.exe scripts/improve_mobilenetv3.py
.\.venv\Scripts\python.exe scripts/execute_baseline_notebook.py --notebook notebooks/05_mobilenetv3_classifier_and_calibration.ipynb
.\.venv\Scripts\python.exe scripts/check_calibrated_model.py
```

Новый выпуск — `models/mobilenetv3_calibrated`; для повторного расчёта укажите новый `--output`. Общие зависимости те же; исполнители поддерживают `--extra-site`. Настройки — `config/mobilenetv3_calibrated.json`; варианты и выбор сохранены в `validation_candidates.csv` и `selection.json`. Проверены gallery cosine и Ridge alpha=0.01/0.1/1.0. Выбор по val true accept, затем macro-F1 и accuracy; test не меняет выбор.

Per-POI score/margin пороги выбираются совместно по positive-val и negative-val. Динамическое программирование максимизирует число верных подтверждений при **общем** бюджете external false accepts и known-wrong accepts. Бюджет не сбрасывается для каждого класса. В `thresholds_by_poi.json` сохранены конкретные пороги. Калибровка по 1–2 положительным val-фото у малых классов может сильно переучитывать validation.

**Измеренный результат test:** accuracy 84.93%, macro-F1 84.63%, true accept 45.89%, external any-POI FAR 5.06%. Классификация улучшилась относительно 74.66%/73.26%, но требование FAR ≤1% нарушено. Поэтому статус — `not_promoted_far_exceeded`: кандидат не рекомендован как замена прежней строгой политики. Цель 85% true accept не достигнута. Порогами по test это не исправляется; нужен более крупный независимый validation и новый закрытый полевой test.

```powershell
.\.venv\Scripts\python.exe scripts/baseline_mobilenetv3.py verify --artifacts models/mobilenetv3_calibrated --image path/to/photo.jpg --target-poi-id moscow/ostankinskiy/ostankino_tower
```

Отчёт — `reports/mobilenetv3_calibrated/summary.md`. Прежний строгий baseline доступен в `models/baseline_mobilenetv3_negatives`; CLI без `--artifacts` по-прежнему использует первоначальный `models/baseline_mobilenetv3`. Эксперименты не переключают используемую модель автоматически. Исходники и все положительные/отрицательные splits сохранены.

### Дообучение MobileNetV3 для различения POI

`notebooks/04_mobilenetv3_finetuning.ipynb` и `config/mobilenetv3_finetune.json` задают новый эксперимент. Обучаются последние четыре блока MobileNetV3-Small; BatchNorm заморожен. Веса обновляются **только по положительным train-фото**. Сбалансированные батчи 8 POI × 3 независимых снимка используют cosine classification и supervised contrastive loss. Negative-val участвует только в выборе checkpoint и порогов. Целевые ограничения FAR сохранены: known-wrong ≤1%, hard-wrong ≤5%, external any-POI ≤1% на validation.

```powershell
.\.venv\Scripts\python.exe scripts/finetune_mobilenetv3.py
.\.venv\Scripts\python.exe scripts/execute_baseline_notebook.py --notebook notebooks/04_mobilenetv3_finetuning.ipynb
.\.venv\Scripts\python.exe scripts/check_finetuned_model.py
```

При необходимости используйте рабочий Python с pandas и `--extra-site .venv/Lib/site-packages`. Новые зависимости или скачивание весов не нужны, если предыдущий baseline уже работает. Для точного повторения используйте зафиксированную CPU-среду `requirements-baseline-lock.txt`. Новый эксперимент с тем же seed и конфигурацией запускайте с `--output models/mobilenetv3_finetuned_run2`; имеющийся выпуск не перезаписывается. Ноутбук по умолчанию читает `models/mobilenetv3_finetuned` после проверки хэшей; для просмотра другого выпуска измените OUTPUT.

Изображение сохраняется целиком: первый кадр MPO, EXIF orientation, JPEG draft 640, thumbnail до 320, поля до 224×224, ImageNet normalization. Общая процедура используется в обучении и одиночном выводе; это новая версия preprocessing. Кэш PNG с хэшами находится в `.runtime/finetune_cache/`, исходники не меняются. Train-аугментации: небольшое affine-преобразование, горизонтальное отражение, умеренное изменение цвета. Val/test обрабатываются детерминированно, без аугментаций.

Checkpoint выбирается по максимальному val true accept при ограничениях FAR, затем по macro-F1 и accuracy. Каждая эпоха строит новую галерею из train и калибрует отказ только на positive-val/negative-val. Epoch 0 фиксирует контроль frozen encoder с новым preprocessing. Test-фото впервые декодируются этим экспериментом **после** сохранения выбранной эпохи и порогов; их оценка не меняет выбор. Это прежний диагностический положительный test, поэтому окончательное подтверждение качества требует нового закрытого полевого набора. Достижение 85% не гарантируется.

На validation также сравниваются три преобразования метрики: identity, вычитание train-среднего и PCA-whitening до 128D с регуляризацией 5% от максимального собственного значения covariance. Все параметры fit вычисляются **только по positive-train**; val выбирает вариант, а test его не меняет. Матрица и среднее сохраняются в `embedding_transform.npz` и применяются загрузчиком при каждом запросе. Число проверенных вариантов увеличивает оптимизм val-оценки; окончательный отчёт отдельно показывает отложенные результаты. Первый пробный запуск без преобразования метрики сохранён в `_archive/finetune_pilot_without_centering_20261026`; он остановлен до расчёта test-метрик и не является готовым выпуском.

Артефакты: `models/mobilenetv3_finetuned/encoder.pt`, `best_checkpoint.pt`, `training_history.csv`, `selection.json`, `gallery.csv`, `rejection_policy.json`, snapshots splits и эмбеддинги с атрибуцией. `metadata.json` публикуется после завершения расчётов и содержит хэши, версии библиотек и входных CSV. Отчёт и кривые — `reports/mobilenetv3_finetuned/`.

```powershell
.\.venv\Scripts\python.exe scripts/baseline_mobilenetv3.py verify --artifacts models/mobilenetv3_finetuned --image path/to/photo.jpg --target-poi-id moscow/ostankinskiy/ostankino_tower
```

CLI выбирает загрузчик по схеме артефактов; дообученный encoder не подменяется исходными ImageNet-весами. Предыдущие модели сохранены, автоматического переключения используемой версии нет.

Основной документ — [notebooks/01_dataset_eda.ipynb](notebooks/01_dataset_eda.ipynb). Число POI определяется из источников; охватываются районы `arbat`, `khamovniki`, `ostankinskiy`, `tverskoy`. Список вычисляется из CSV и папок, ограничений на прежние шесть объектов нет.

### Обновление после ручной очистки и дополнение manual

`config/manual_pois.csv` связывает русские имена папок `data/raw/manual` с полными poi_id Wikimedia. Текущая политика включает **все manual-фото**; прежнее ограничение «меньше 30 Wikimedia-фото» снято. Фактическое число файлов и причины отбора отражены в `reports/manual_enrichment_plan.csv`.

Ручная очистка подтверждена пользователем: оставшиеся Wikimedia-кадры и выбранные manual-кадры отмечены `verified`, reviewer=`user`. Удалённые Wikimedia-файлы сохраняются в audit-строках `review_edits.csv` со статусом `rejected`, причиной `removed_by_user` и `file_exists=False` в манифесте. Они не восстанавливаются и не попадают в splits. Поэтому число audit-строк может превышать число существующих изображений.

Во всех итоговых таблицах, включая review_edits.csv и annotations.csv, присутствуют **source, author, licence**. Для manual указаны `manual`, `Shilkova A.M.`, `Public Domain` — по прямому подтверждению пользователя. Для Wikimedia сохраняются исходные персональные атрибуции, лицензии и URL. Дополнение не копирует изображения в Wikimedia: источник остаётся различимым по пути и полю source.

Явная синхронизация после подтверждённой проверки и пересборка:

```powershell
.\.venv\Scripts\python.exe scripts/update_dataset_sources.py --confirm-wikimedia --confirm-manual
.\.venv\Scripts\python.exe scripts/audit_enriched_negatives.py
.\.venv\Scripts\python.exe scripts/build_eda_notebook.py
.\.venv\Scripts\python.exe scripts/execute_eda_notebook.py --reuse-inventory
```

Не передавайте флаги `--confirm-*` без фактической ручной проверки: они фиксируют утверждение пользователя, а не автоматическую оценку модели. До синхронизации прежняя разметка и отчёты архивируются в `_archive/dataset_update_*/`. Старые визуализации переносятся туда, чтобы не показывать удалённые кадры как текущие. `--reuse-inventory` использует метрики и crop-хэши из только что выполненной инвентаризации; в конце всё равно проверяются SHA-256 исходников и состав выбранного набора. Индексы прежних классов сохраняются, новые дописываются.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab notebooks/01_dataset_eda.ipynb
```

Выберите Python-ядро с установленными зависимостями и выполните Run All. Перед запуском остановите Wikimedia-парсер: исходный набор должен оставаться неизменным во время аудита. Для запуска без браузера:

```powershell
.\.venv\Scripts\python.exe scripts/execute_eda_notebook.py
```

Ноутбук использует pandas, NumPy, OpenCV, matplotlib, Pillow и tqdm. В локальной `.venv` обнаружена ошибка DLL у pandas 3.1.0; для стандартного запуска переустановите совместимую версию: `python -m pip install --force-reinstall "pandas>=2.2,<3.1"`. Первоначальный аудит выполнен через настоящее Jupyter-ядро с доступным рабочим pandas другого локального Python; фактические версии записаны в metadata ноутбука. Опция `--extra-site <site-packages>` у исполнителя позволяет добавить вторичные библиотеки после пакетов выбранного Python; она не скачивает и не подменяет результаты вычислений.

Для Linux/Colab установите requirements.txt и откройте весь проект с сохранённой структурой `data/raw/wikimedia`. Относительные пути к фотографиям и сопоставление metadata по город/район/POI/имени файла поддерживают перенос с Windows. Автоматическая загрузка данных во внешние сервисы не выполняется.

### Артефакты

- `src/dataset_eda.py` — новые вспомогательные функции аудита, проверки и группового разделения.
- `data/annotations/poi_catalog.csv` — соответствие папок, полного poi_id и индекса класса.
- `data/annotations/class_mapping.json` — фиксированный mapping; существующие индексы сохраняются, новые дописываются.
- `data/annotations/initial_manifest.csv` — исходные предварительные метки, технические метрики и происхождение.
- `data/annotations/review_edits.csv` — редактируемая таблица решений; не перезаписывается повторным запуском.
- `data/annotations/annotations.csv` — применённые решения; `review_queue.csv` — актуальная очередь.
- `data/annotations/duplicate_pairs.csv`, `duplicate_groups.csv`, `review_conflicts.csv` — копии, граф групп и конфликты подтверждённых меток.
- `data/annotations/visual_review_observations.csv` — замечания при просмотре выборочных contact sheets. Это предложения для ручной проверки; они не меняют статусы и метки автоматически.
- `data/annotations/dataset_manifest.csv`, `train.csv`, `val.csv`, `test.csv`, `split_feasibility.csv` — итоговое состояние и возможность выпуска выборок.
- `data/processed/dataset.csv` — только выпущенные изображения; `data/processed/images/` появляется при выпуске и содержит побайтовые копии, без аугментаций.
- `reports/eda_summary.md`, `eda_statistics.json`, `class_distribution.csv` — качество, готовность и распределения.
- `reports/figures/`, `reports/contact_sheets.csv` — contact sheets всех изображений, подборки и графики; thumbnails используются только для просмотра.
- `reports/raw_integrity_snapshot.json`, `integrity_checks.json` — контроль неизменности исходников и результатов.

`poi_id` имеет форму `moscow/arbat/pashkov_house`. Исходный `poi_slug` сохранён отдельно. Источник устанавливается как `wikimedia_commons` при подтверждённом сопоставлении metadata/URL, иначе `unknown`. Поля `author`, `author_raw`, `licence` (и alias `license`), `license_url`, `source_url`, `commons_title` сохраняют атрибуцию и происхождение. Исходные metadata.csv, фото и CSV прогресса парсера не редактируются.

### Ручная проверка и пересборка

1. Просмотрите **все страницы** `reports/figures/all_*.jpg`, подборки `quality_*.jpg` и пары `duplicate_pairs_*.jpg`. В подписях есть image_id и poi_id. Полный размер доступен в ноутбуке через `show_image(initial, ROOT, image_id)`.
2. В `review_edits.csv` не меняйте `image_id`, `file_path`, `sha256`. При ошибке исправьте `poi_id` по poi_catalog.csv. Укажите `verified` только после просмотра кадра и заполните `reviewer`. Для отказа поставьте `rejected` и причину (`target_absent`, `interior`, `non_scene`, `collage`, `target_occluded` и т. п.). Спорное — `uncertain`, непросмотренное — `needs_review`.
3. Известным сериям съёмки и дополнительно найденным копиям присвойте общий глобальный `series_id`; при необходимости добавьте `review_note`. Сохраните UTF-8 CSV.
4. Повторно выполните **секции 5–8** ноутбука. После нового открытия ядра секция 5 загрузит исходный манифест с диска. Проверки остановят сборку при изменённых хэшах, ошибочных ID, статусах или отсутствии обязательных полей.
5. Проверьте отчёт и split_feasibility.csv. Пока нет ≥3 независимых подтверждённых групп каждого класса и полного покрытия классов, train/val/test остаются пустыми с заголовками. Это ожидаемая блокировка выпуска, а не готовый обучающий датасет.

Цель split — 70/15/15, SEED=42. Связи SHA-256, Commons file ID, pHash, кропов и series_id не пересекают выборки, в том числе через исключённые кадры. Near-duplicates — только кандидаты; они не удаляются автоматически. Резкость не служит автоматическим порогом отказа. Для первого выпуска сохраняйте консервативную группировку; pHash не гарантирует обнаружение всех копий.

Число независимых групп важнее точного соответствия пропорциям. Три группы на класс — лишь технический минимум для трёх splits; небольшие val/test не дадут устойчивой оценки. После ручной правки split может измениться: перед будущим обучением необходимо зафиксировать версию CSV. Старые processed-копии автоматически не удаляются; используйте только актуальные CSV.

Для теста отказа на посторонних объектах нужен отдельный проверенный набор негативов; ошибочные фотографии целевых POI не превращаются в unknown. Правила нескольких POI, перекрытий, интерьеров и коллажей приведены в ноутбуке. Авторские/лицензионные сведения сохраняются, условия каждого файла и атрибуция должны учитываться при дальнейшем использовании.

## Структура

- `scripts/wikimedia_parser.py` — загрузка фотографий и сохранение прогресса.
- `config/wikimedia_pois.csv` — города, районы, POI, категории Commons и статусы.
- `data/raw/wikimedia/<city>/<district>/<poi>/` — загруженные фотографии.
- `data/raw/wikimedia/metadata.csv` — URL, авторы, лицензии и принадлежность файлов.
- `tests/test_wikimedia_parser.py` — проверки без сетевых запросов.
- `ВДНХ/` — сохранённые исходники прежнего сбора; автоматически в новый набор не включаются.
- `_archive/` — резервные копии прежних CV-разработок, не участвующие в текущем процессе.

## Установка и запуск

Python 3.11+. PowerShell из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe scripts/wikimedia_parser.py
```

На Linux используйте `.venv/bin/python`. Существующее локальное окружение можно продолжать использовать.

## Настройка POI

### Проверенные negatives и новый выпуск отказа

После подтверждённого просмотра оставшихся файлов `data/raw/wikimedia_negatives`:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_negative_dataset.py --confirm-reviewed
.\.venv\Scripts\python.exe scripts/calibrate_negative_baseline.py
.\.venv\Scripts\python.exe scripts/execute_baseline_notebook.py --notebook notebooks/03_negative_dataset_and_rejection.ipynb
```

Флаг `--confirm-reviewed` фиксирует явное подтверждение пользователя; без проверки его передавать нельзя. Новый аудит уже выполненного выпуска не перезаписывается: повторный просмотр результатов доступен в `notebooks/03_negative_dataset_and_rejection.ipynb`. Все три исполнителя поддерживают `--extra-site .venv/Lib/site-packages` для рабочего Python с pandas, если основная `.venv` всё ещё содержит повреждённый pandas.

Артефакты подготовки: `data/annotations/negative_manifest.csv`, `negative_review_edits.csv`, `negative_review_queue.csv`, `negative_duplicate_pairs.csv`, `negative_val.csv`, `negative_test.csv`; сводка — `reports/negatives/dataset_summary.json`. В manifest остаются удалённые строки metadata как rejected / removed_by_user. Атрибуция source/author/licence, исходные авторские сведения и URL сохранены. `source_poi_id` обозначает посторонний объект; `poi_id` остаётся пустым — negatives не добавляются в mapping целевых классов.

После ручной очистки не запускайте сбор повторно в очищенный каталог: текущий парсер умеет восстанавливать отсутствующие файлы из metadata и может вернуть удалённые кадры. Для нового сбора используйте отдельный каталог/выпуск. Готовые CSV и source_snapshot.json фиксируют проверенный состав. Проверка выпуска: `python scripts/check_negative_baseline.py` (при необходимости с `--extra-site`).

SHA-256, Commons-ID, pHash и кропы сопоставляются также с существующими положительными фото. Консервативные пересечения получают `needs_review` / `positive_overlap_candidate` и исключаются; это технические кандидаты, а не автоматическое опровержение ручной проверки. Точные копии negatives сохраняются в аудите, но активен один экземпляр. Negative val/test делятся примерно 50/50 **по целым source POI и связанным группам копий**, SEED=42. Положительные CSV не меняются. Редкий тип с единственным посторонним POI не может присутствовать в обоих splits при таком разделении.

`negative_review_edits.csv` сохраняет принятые на момент выпуска решения для аудита. Текущий исполнитель **не применяет правки к этой таблице и не пересобирает существующий выпуск**. Если нужно вернуть исключённый кандидат, сначала проверьте его связь с положительным набором и выпускайте новую версию manifests/splits с явным пересмотром ограничений; простое изменение CSV не меняет готовые выборки.

Новая политика — `models/baseline_mobilenetv3_negatives/`. Веса, положительные эмбеддинги и train-галерея переиспользуются после проверки хэшей; сами negative-фото в галерею не добавляются. Пороговые ограничения из `config/baseline_mobilenetv3_negatives.json`: known-wrong FAR ≤1%, hard-wrong FAR ≤5%, **external any-POI FAR ≤1%** на validation. Настройка использует только positive-val и negative-val; оба test оцениваются после фиксации политики. Это эмпирические ограничения validation, не гарантия частоты ошибок новых фотографий. True accept может снизиться: настройка отказа не улучшает сами эмбеддинги.

```powershell
.\.venv\Scripts\python.exe scripts/baseline_mobilenetv3.py verify --artifacts models/baseline_mobilenetv3_negatives --image path/to/photo.jpg --target-poi-id moscow/ostankinskiy/ostankino_tower
```

Прежний baseline остаётся в `models/baseline_mobilenetv3`; без `--artifacts` CLI использует его. Итог нового выпуска — `reports/baseline_mobilenetv3_negatives/summary.md`; предсказания каждого negative-test фото и его атрибуция — `models/baseline_mobilenetv3_negatives/negative_test_predictions.csv`. Этот negative-набор покрывает прежде всего достопримечательности Commons, а не все условия мобильной съёмки.

### Сбор похожих посторонних POI для проверки отказа

`config/wikimedia_negatives.csv` — отдельный конфиг **25 POI-кандидатов в negatives**. Все строки начинаются со `status=pending`, `downloaded_count=0`, `label_status=needs_review`. Существующие целевые POI и их прогресс остаются в `config/wikimedia_pois.csv`. Загрузка кандидатов:

```powershell
.\.venv\Scripts\python.exe scripts/wikimedia_parser.py --config config/wikimedia_negatives.csv --output data/raw/wikimedia_negatives --max-per-poi 50
```

Для повторного запуска после прерывания используйте ту же команду. `done` означает завершение загрузки категории, а не подтверждённую отрицательную метку. Парсер сохраняет дополнительные колонки конфига при обновлении прогресса. Сам сбор не был запущен при подготовке списка.

Состав списка:

| Группа | Объекты |
|---|---|
| Павильоны ВДНХ | «Беларусь» №18, «Узбекистан» №66, «Азербайджан» №14, «Карелия» №67, «Украина / Земледелие» №58, «Транспорт» №26, «Цветоводство» №29 |
| Фонтаны | «Золотой колос» |
| Арки | Главный вход в парк Горького, Триумфальная арка |
| Башни и колёса | Шуховская башня, Большое колесо обозрения Измайловского парка |
| Театры и классические фасады | Театр Российской армии, Манеж, Северный речной вокзал после реконструкции, здание мэрии Москвы, Ленинградский вокзал, Центральный Московский ипподром |
| Декоративные здания | Саввинское подворье, Петровский путевой дворец, Казанский вокзал |
| Гостиницы | «Украина», «Пекин», бывший Ritz-Carlton / The Carlton, Four Seasons Hotel Moscow |

В CSV для каждого объекта указаны русское название (`name_ru`), тип негатива (`negative_type`), потенциально похожие целевые POI (`confusable_with`), ссылка Commons, дата проверки категории и замечания для просмотра. Проверка существования категории выполнена по странице Commons или списку подкатегорий родительской страницы на 2026-10-26; это не проверка содержимого всех фотографий. Категории выбраны с прямыми файлами: текущий парсер не обходит подкатегории. Например, для Ленинградского вокзала указаны экстерьеры, для Северного речного вокзала — категория после реконструкции. Количество доступных изображений может меняться.

Лимит 50 даёт **верхнюю границу 1250 файлов** до очистки, не гарантирует 1000 пригодных независимых снимков: некоторые категории меньше лимита, возможны неподдерживаемые форматы, дубликаты и нерелевантные изображения. При нехватке можно увеличить лимит; ранее скачанные файлы с metadata не скачиваются заново.

После загрузки необходим просмотр: в категориях «Узбекистана» и «Украины» встречаются фотографии целевого фонтана «Каменный цветок», а в панорамах других объектов могут присутствовать целевые здания или Останкинская башня. Такие кадры **не являются внешними negatives**. Интерьеры, марки, чертежи, исторические изображения, мероприятия и снимки объекта на экране следует отделять от основного набора современной городской съёмки. Снимки объектов из одной серии и копии нельзя разносить между negative-val/test; для более строгой проверки незнакомых объектов желательно разделять их также по самому POI.

Файлы сохраняются в `data/raw/wikimedia_negatives/<city>/<district>/<poi>/`, происхождение — в отдельном `metadata.csv` этого корня. Сохраняются `source_url`, `artist`, `license`, `license_url`; при подготовке разметки они станут source=`wikimedia_commons`, author, licence. Лицензия берётся у каждого файла, Public Domain для Commons автоматически не назначается. Этот корень не входит в текущий EDA целевых классов; новые объекты не добавляются в class_mapping и галерею. При проверке negatives потребуется сравнить их SHA-256/Commons-ID/near-duplicate группы также с положительными train/val/test. Обычные городские фото с телефона нужны дополнительно: этот список покрывает прежде всего похожие достопримечательности.

```csv
city,district,poi,category,status,downloaded_count,last_error,progress_key
moscow,ostankinskiy,main_entrance,VDNKh Main Entrance Arch,pending,0,,
```

`city`, `district`, `poi` задают каталоги: строчные латинские буквы, цифры, `_` и `-`. `category` — точное имя Commons-категории, допускается префикс `Category:`. Одна тройка город/район/POI соответствует одной строке. CSV поддерживает UTF-8 и UTF-8 с BOM.

Добавляя строку, заполните первые четыре колонки и установите `pending`; остальные поля прогресса оставьте пустыми. Старые CSV без колонок прогресса тоже поддерживаются.

```powershell
.\.venv\Scripts\python.exe scripts/wikimedia_parser.py --config config/wikimedia_pois.csv --output data/raw/wikimedia --max-per-poi 50
```

По умолчанию пути считаются от корня проекта; явно переданные относительные пути — от текущей папки shell. Импорт модуля загрузку не запускает.

## Продолжение после сбоя

- `pending` — ещё не обработано.
- `in_progress` — начато или прервано.
- `done` — достигнут лимит или закончилась категория.
- `error` — ошибка, её сообщение находится в `last_error`.

`downloaded_count` — число сохранённых файлов с метаданными. `progress_key` связывает прогресс с категорией, выходной папкой и лимитом; вручную менять его не нужно.

После сбоя повторите ту же команду. Завершённые POI пропускаются без запросов категории. Незавершённая категория перечитывается, но сохранённые файлы повторно не скачиваются. Метаданные и прогресс фиксируются после каждого файла атомарной заменой CSV. Файл без metadata после прерывания восстанавливается при повторном обходе; `.part`-файлы не считаются готовыми.

Изменение категории, выходной папки или лимита запускает повторную проверку. `done` с нулевым счётчиком означает пустую категорию или отсутствие поддерживаемых файлов; поставьте `pending`, чтобы проверить её повторно. Запускайте один экземпляр парсера на один CSV и держите CSV доступным для записи.

## Метаданные и тесты

Сохраняются город, район, POI, категория, путь, URL, название Commons, размеры, автор, лицензия и ссылка на лицензию. Повторные запуски сохраняют прежние записи и дополнительные колонки. При дальнейшем использовании учитывайте условия конкретных файлов и требования к атрибуции.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Тесты используют временные папки и имитируют загрузку, ошибки и продолжение работы. Реальные фотографии при тестировании не скачиваются.

## Архив прежней работы

Старый CV-контур вынесен из рабочих каталогов в `_archive/legacy_cv_2026-10-26_6b6427dc_files/`. Дополнительно сохранён ZIP `_archive/legacy_cv_2026-10-26_6b6427dc.zip`, проверенный по SHA-256. В архиве находятся прежние модели, веса, код, API, ноутбуки, отчёты, разметка и подготовленные выборки. Исходные README и зависимости сохранены в ZIP. Для восстановления используйте отдельную папку.

Wikimedia-фотографии, метаданные, конфигурация со статусами и исходники в `ВДНХ/` сохранены. Прежние файлы не подключаются к новому сбору автоматически.
