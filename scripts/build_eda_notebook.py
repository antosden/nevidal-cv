"""Создание нового EDA-ноутбука для данных Wikimedia."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
book=nbf.v4.new_notebook(cells=[
md('''# 01. Wikimedia: EDA, проверка разметки и подготовка датасета

Распознавание конкретного POI, image-level classification / instance recognition.
Обрабатываются **все** города/районы/POI из `data/raw/wikimedia`, независимо от прежнего пилота.
Manual включается целиком согласно config/dataset_sources.json; POI и атрибуция — из config/manual_pois.csv.
data/annotations/split_anchors.csv фиксирует прежние назначения splits: новые группы распределяются с SEED=42,
а группы, связывающие разные прежние splits, изолируются от выпуска.
Удалённые пользователем файлы остаются audit-строками rejected и в выборки не попадают.
Текущий каталог содержит четыре района; названия и число классов вычисляются из данных.
Папка даёт предварительный poi_id, а не Ground Truth. Старые архивы и исходники «ВДНХ» в этот анализ не входят.

Исходники и `metadata.csv` доступны только для чтения. Остановите загрузчик перед аудитом, чтобы состав файлов был фиксирован.
Никаких CV-моделей, весов, аугментаций, bbox и сегментации здесь нет.
'''),
code('''from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
from PIL import Image
from IPython.display import display, Markdown, Image as DisplayImage

ROOT=Path.cwd().resolve()
if not (ROOT/'config/wikimedia_pois.csv').exists(): ROOT=ROOT.parent
assert (ROOT/'data/raw/wikimedia/metadata.csv').exists(), 'Откройте проект целиком из корня или notebooks/'
sys.path.insert(0,str(ROOT))
from src.dataset_eda import *
ROOT=prepare_paths(ROOT)
np.random.seed(SEED)
print('SEED:',SEED)
'''),
md('''## 1. Инвентаризация и происхождение

`poi_id = city/district/poi_slug` предотвращает коллизии имён разных районов. Фиксированное отображение
`poi_id → class_index` хранится в JSON: новые классы дописываются, существующие не перенумеровываются.

Метаданные сопоставляются по city/district/poi/имени файла; старые абсолютные Windows-пути не мешают Colab/Linux.
`author` — текст автора; `author_raw` сохраняет исходную HTML-атрибуцию. `licence` и совместимый alias `license`
содержат исходную лицензию; source_url, license_url и commons_title сохраняются. Исходный metadata.csv не редактируется.
Источник `wikimedia_commons` устанавливается по сопоставленной записи с URL Wikimedia; без свидетельства — `unknown`.
Для manual: source=manual, author=Shilkova A.M., licence из config/manual_pois.csv (Public Domain по сообщению пользователя).
'''),
code('''initial, hashes, mapping=inventory(ROOT)
current=initial[selected_present(initial)]
print('Текущих изображений:',len(current),'| Audit-записей:',len(initial),'| POI:',current.poi_id.nunique(),'| Районов:',current.district.nunique())
display(pd.read_csv(ROOT/'data/annotations/poi_catalog.csv'))
display(current.groupby(['city','district']).size().rename('images').reset_index())
display(current.source.value_counts().to_frame('images'))
display(current['format'].value_counts().to_frame('images'))
display(pd.read_csv(ROOT/'reports/manual_enrichment_plan.csv'))
counts=current.poi_id.value_counts()
print('Дисбаланс max/min:',counts.max()/counts.min())
display(initial[['image_id','poi_id','source','author','licence','source_url']].head())
'''),
md('''## 2. Техническое качество

Размеры после учёта EXIF, aspect ratio, яркость, Laplacian variance и доли почти чёрных/белых пикселей.
Яркость/резкость считаются при длинной стороне ≤1024 px. Разный масштаб и текстура влияют на резкость.
Сторона <224 px, яркость <40/>215 и aspect вне [0.5,2] — **флаги для просмотра**, не автоматический отказ.
Автоматический технический отказ применяется только к нечитаемым изображениям.
'''),
code('''display(current[['width','height','aspect_ratio','file_size','brightness','sharpness','dark_fraction','bright_fraction']].describe())
display(current[~current.readable][['image_id','file_path','rejection_reason','read_error']])
display(current[current.quality_flags.ne('')][['image_id','poi_id','quality_flags']])
quality_plots(initial,ROOT)
for name in ['class_counts.png','technical_histograms.png','technical_boxplots.png']:
    display(DisplayImage(filename=str(ROOT/'reports/figures'/name)))
'''),
md('''## 3. Точные и потенциальные дубликаты

SHA-256, идентичный Commons file ID и pHash проверяются **между всеми POI**.
pHash: DCT 32×32, блок 8×8 без DC (63 бита), Hamming ≤6.
Полный кадр сравнивается также с десятью кропами 80%/60% — углы и центр на копии ≤1024 px, Hamming ≤4.
Метод обнаруживает часть копий с изменённым размером/сжатием/кадрированием, но не гарантирует поиск всех вариантов.

Потенциальные связи объединяются транзитивно. Near-копии автоматически не удаляются:
группа только запрещает пересечение splits. Ложные связи могут укрупнить группы; проверьте их визуально.
'''),
code('''initial, duplicate_pairs=find_duplicates(initial,hashes,ROOT)
display(duplicate_pairs)
display(pd.read_csv(ROOT/'data/annotations/duplicate_groups.csv'))
display(duplicate_pairs[duplicate_pairs.cross_poi])
'''),
md('''## 4. Визуальное EDA

Все фотографии сохраняются в contact sheets `all_<city>__<district>__<poi>_<page>.jpg`, по 24 изображения.
Подборки `quality_*` содержат случайные кадры и экстремумы яркости/резкости/aspect; повторные ID убираются.
В `duplicate_pairs_*` каждая строка — одна пара. Подписи содержат image_id и полный poi_id.
Ниже показаны первые страницы каждого POI и первые страницы дубликатов; полный список — reports/contact_sheets.csv.
Технические thumbnails — только для визуализации, в обучение они не попадают.
'''),
code('''sheet_paths=visual_eda(apply_reviews(initial,ROOT,mapping),duplicate_pairs,ROOT)
display(pd.read_csv(ROOT/'reports/contact_sheets.csv'))
for path in sheet_paths:
    if path.name.startswith('all_') and path.name.endswith('_01.jpg'):
        display(DisplayImage(filename=str(path)))
for path in [p for p in sheet_paths if p.name.startswith('duplicate_pairs_')][:3]:
    display(DisplayImage(filename=str(path)))
'''),
code('''# Подборки качества каждого POI: случайные + тёмные/светлые + нерезкие + необычный aspect.
for path in sheet_paths:
    if path.name.startswith('quality_'):
        display(DisplayImage(filename=str(path)))

# Полное разрешение по image_id; раскомментировать и подставить ID с contact sheet.
# show_image(initial, ROOT, 'wm_...')
'''),
md('''## 5. Ручная разметка

`review_edits.csv` — редактируемый источник решений. Создаётся один раз и не перезаписывается при повторном EDA.
`annotations.csv` — применённые решения; `review_queue.csv` — очередь с приоритетом качества/дубликатов.
Не меняйте image_id, file_path, sha256 и provenance source/author/licence. Исправляйте poi_id только на существующий полный ID из poi_catalog.csv.
Оставшийся Wikimedia и выбранный manual подтверждены пользователем при обновлении; это явно отражено в review_edits.csv.

| Статус | Значение |
|---|---|
| verified | Кадр просмотрен, POI подтверждён; обязателен reviewer |
| needs_review | Ещё не просмотрено |
| uncertain | Недостаточно оснований для уверенной метки |
| rejected | Исключить; обязательна rejection_reason |

| Случай | Правило |
|---|---|
| Другой целевой POI | Исправить poi_id и подтвердить после просмотра |
| Целевой объект отсутствует | rejected/target_absent; при сомнении uncertain |
| Несколько POI | Подтверждать один явно доминирующий и однозначный; иначе uncertain, без размножения кадра в классы |
| Частичный вид/перекрытие | verified только при узнаваемых уникальных признаках; иначе uncertain или rejected/target_occluded |
| Интерьер или фрагмент | Ручная проверка; для baseline внешнего вида rejected/interior или detail, если POI не распознаётся |
| Карта, афиша, стенд, экран | rejected/non_scene; неясные случаи uncertain |
| Коллаж | rejected/collage для baseline отдельных фотографий |

Неподходящий снимок не становится искусственным классом unknown. Для оценки отказа потребуется отдельный набор негативов.
В series_id указывайте известную общую серию съёмки или вручную найденные пропущенные копии; ID должен быть глобально уникален.
EXIF время не группируется автоматически. Embeddings можно использовать позднее для ранжирования подозрений, не для автоматического verified.
'''),
code('''# Эту секцию и последующие можно повторять после сохранения CSV.
if 'initial' not in globals():
    initial=pd.read_csv(ROOT/'data/annotations/initial_manifest.csv',keep_default_na=False)
    for col in ['width','height','aspect_ratio','file_size','brightness','sharpness','dark_fraction','bright_fraction']:
        initial[col]=pd.to_numeric(initial[col],errors='coerce')
    mapping=json.loads((ROOT/'data/annotations/class_mapping.json').read_text(encoding='utf-8'))
    duplicate_pairs=pd.read_csv(ROOT/'data/annotations/duplicate_pairs.csv',keep_default_na=False)
reviewed=apply_reviews(initial,ROOT,mapping)
display(reviewed[REVIEW_COLUMNS].head(10))
'''),
code('''# Пример точечного редактирования после просмотра:
# edits=pd.read_csv(ROOT/'data/annotations/review_edits.csv',dtype=str,keep_default_na=False)
# selected=edits.image_id.eq('wm_...')
# assert selected.sum()==1
# edits.loc[selected,['poi_id','label_status','reviewer','review_note']]=[
#     'moscow/arbat/pashkov_house','verified','имя проверяющего','Подтверждён фасад']
# edits.to_csv(ROOT/'data/annotations/review_edits.csv',index=False,encoding='utf-8-sig')
# reviewed=apply_reviews(initial,ROOT,mapping)
'''),
md('''## 6. Очистка и train/val/test

SEED=42, цель 70/15/15 по числу изображений каждого POI с ограничениями компонент.
Доступны только readable+verified. Из SHA-копий оставляется одна подтверждённая;
конфликт подтверждённых меток одной фотографии исключает конфликтующие строки и сохраняет review_conflicts.csv.
Все near-duplicates/Commons-ID/series_id объединяются до фильтрации, включая связи через исключённые кадры.

Для выпуска требуется ≥3 независимых verified-группы **каждого** класса и найденное покрытие всех классов во всех splits.
При недостатке данных CSV остаются с заголовками и нулём строк. Три группы — технический минимум, не надёжный размер теста.
Processed-копии сохраняются побайтово, без перекодирования. Источник истины после пересборки — актуальные CSV;
устаревшие processed-копии автоматически не удаляются. При повторной сборке split может измениться после изменения разметки,
поэтому перед будущим обучением потребуется зафиксировать версию CSV.
'''),
code('''reviewed=apply_reviews(initial,ROOT,mapping)
manifest,ready=build_dataset(reviewed,ROOT,mapping)
display(pd.read_csv(ROOT/'data/annotations/split_feasibility.csv'))
display(manifest.groupby(['poi_id','label_status','split']).size().rename('images').reset_index())
print('Готовность выпущенных выборок:',ready)
'''),
md('''## 7. Проверки целостности

Проверяются SHA-256 всех исходников, metadata.csv и конфигурации, состав raw,
пересечения image_id/SHA/duplicate_group/split_group, verified-статусы и хэши prepared-копий.
Пустые splits не являются доказательством готовности к обучению.
'''),
code('''integrity=verify_integrity(ROOT,manifest)
display(pd.Series(integrity).to_frame('value'))
(ROOT/'reports/integrity_checks.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding='utf-8')
'''),
md('''## Будущие аугментации (пока не применяются)

1. Умеренное изменение яркости, контраста и цвета для погоды и освещения.
2. Небольшие повороты и перспективные преобразования для положения телефона.
3. Осторожный crop/resize с сохранением уникальных признаков POI.
4. Небольшой шум/размытие/JPEG-сжатие для мобильной камеры.

Применять только к train после split. Отражение нежелательно для надписей и асимметричных фасадов.
Val/test остаются детерминированными.
'''),
md('''## 8. Автоматический итоговый отчёт

Каждый вывод: наблюдение → интерпретация → действие. Отдельно отражены готовность и блокеры,
причины исключений, происхождение и лицензии, число независимых групп и распределение классов.
'''),
code('''statistics=make_report(manifest,duplicate_pairs,ready,ROOT)
display(pd.Series(statistics).to_frame('value'))
display(Markdown((ROOT/'reports/eda_summary.md').read_text(encoding='utf-8')))
''')],metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}})
(ROOT/'notebooks').mkdir(exist_ok=True)
nbf.write(book,ROOT/'notebooks/01_dataset_eda.ipynb')
