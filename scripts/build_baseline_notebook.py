"""Создать ноутбук воспроизводимого MobileNetV3-baseline."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
book = nbf.v4.new_notebook(cells=[
md('''# 02. Верификация конкретного POI: MobileNetV3 → эмбеддинги → галерея → отказ

Frozen **MobileNetV3-Small / IMAGENET1K_V1**: не обучаем classifier и не дообучаем backbone.
Используем все POI из фиксированного mapping. Эталонные фото — только train;
пороги cosine и отрыва от другого POI — только val. Test оцениваем после фиксации порогов.
Нет искусственного unknown-класса и нет bbox. Все вычисления локальны, исходные фото не изменяются.
'''),
code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image as DisplayImage

ROOT = Path.cwd().resolve()
if not (ROOT/'config/baseline_mobilenetv3.json').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.poi_baseline import (load_splits, run_baseline, POIVerifier, sha256_file)
OUTPUT = ROOT/'models/baseline_mobilenetv3'
CONFIG = ROOT/'config/baseline_mobilenetv3.json'
WEIGHTS = ROOT/'.runtime/torch/checkpoints/mobilenet_v3_small-047dcff4.pth'
splits, mapping = load_splits(ROOT)
display(pd.DataFrame({name:frame.groupby('poi_id').size() for name,frame in splits.items()}))
print('POI:', len(mapping), 'SEED=42')
'''),
md('''## 1. Веса, извлечение эмбеддингов и галерея

Предобученные веса должны быть доступны локально; сеть из ноутбука автоматически не вызывается.
При необходимости один раз запустите CLI с `--download-weights` (около 10 MB).
Изображение: первый кадр MPO → EXIF orientation → RGB → resize 256 → center crop 224 → ImageNet normalization.
После `features → avgpool → flatten` получаем 576D и L2-нормируем.
Эталонов максимум 20 на POI: farthest-first из train, по одному кадру группы на POI.
Автор, источник и лицензия каждой фотографии сохраняются в gallery.csv и snapshots splits.

Повторный Run All использует существующий зафиксированный выпуск и проверяет его хэши.
Для нового эксперимента задайте новую OUTPUT-папку; не подбирайте параметры по test.
'''),
code('''if (OUTPUT/'metadata.json').exists():
    metadata = json.loads((OUTPUT/'metadata.json').read_text(encoding='utf-8'))
    for name in splits:
        assert sha256_file(ROOT/f'data/annotations/{name}.csv') == metadata['split_sha256'][name], 'Датасет изменён: нужен новый выпуск'
    assert json.loads(CONFIG.read_text(encoding='utf-8')) == json.loads((OUTPUT/'config.json').read_text(encoding='utf-8')), 'Конфигурация изменена: нужен новый выпуск'
    for name, digest in metadata['artifact_sha256'].items():
        assert sha256_file(OUTPUT/name) == digest, f'Изменён артефакт: {name}'
    assert sha256_file(ROOT/'data/annotations/class_mapping.json') == metadata['class_mapping_sha256']
    policy = json.loads((OUTPUT/'rejection_policy.json').read_text(encoding='utf-8'))
    metrics = json.loads((OUTPUT/'metrics.json').read_text(encoding='utf-8'))
    print('Использован зафиксированный выпуск; хэши проверены.')
else:
    metadata, policy, metrics = run_baseline(ROOT, OUTPUT, WEIGHTS, CONFIG)
gallery = pd.read_csv(OUTPUT/'gallery.csv', keep_default_na=False)
display(gallery.groupby('poi_id').agg(references=('image_id','size')))
display(gallery[['image_id','poi_id','source','author','licence']].head())
print('Embedding dimension:', metadata['embedding_dim'], 'Reference version:', metadata['reference_version'])
'''),
md('''## 2. Калибровка отказа только на val

`score(target)` — max cosine с эталонами цели; `margin` — разница с лучшим другим POI.
Принимаем только при прохождении обоих порогов и строго положительном margin.
Каждое val-фото даёт одну правильную цель и C−1 ошибочных запросов известных POI.
Выбираем максимальный true accept rate при эмпирическом known-wrong FAR ≤1%
и hard-wrong FAR ≤5% (по одному наиболее похожему неверному POI на фото).
Это не вероятность, не гарантия FAR и не калибровка отказа на посторонних объектах.
Пары одного фото зависимы; объём независимой выборки равен числу изображений/групп.
'''),
code('''display(pd.Series(policy, name='val-only policy'))
display(pd.DataFrame(metrics).T[['query_count','top1_accuracy','macro_top1_accuracy','true_accept_rate','false_reject_rate','known_wrong_far','hard_wrong_far']])
'''),
md('''## 3. Test после фиксации политики: качество и ошибки

Top-1 показывает только выбор лидера. Для задачи подтверждения цели нужны true accept rate,
false reject rate и known-wrong FAR. Внешний open-set FAR пока не измерен.
'''),
code('''display(pd.read_csv(OUTPUT/'test_per_poi.csv'))
for name in ['test_confusion.png','test_score_distribution.png']:
    display(DisplayImage(filename=str(ROOT/'reports/baseline_mobilenetv3'/name)))
pairs = pd.read_csv(OUTPUT/'test_pairs.csv')
errors = pairs[(pairs.genuine & ~pairs.accepted) | (~pairs.genuine & pairs.accepted)]
display(errors.head(30))
'''),
md('''## 4. Локальная проверка заданного POI

Передайте полный poi_id из mapping. Неизвестный галерее poi_id возвращает
`target_poi_not_supported`. Отрицательное решение сохраняет причины и ближайший эталон цели.
Выбранный ниже пример фиксирован по image_id и не отбирается по правильности ответа.
'''),
code('''verifier = POIVerifier(OUTPUT)
example = splits['test'].iloc[0]
image_path = ROOT/(example.processed_path or example.file_path)
result = verifier.verify(image_path, example.poi_id)
display(pd.Series(result))
print('Все поддерживаемые POI:', list(mapping))
# Свой запрос:
# verifier.verify(Path('path/to/photo.jpg'), 'moscow/ostankinskiy/ostankino_tower')
'''),
md('''## 5. Готовность и следующие шаги

Baseline готов для локального сравнения подходов; production-готовность отказа не подтверждена.
Нужны отдельные проверенные unknown-val/unknown-test и независимые полевые фотографии.
Малые классы с одним test-фото не дают устойчивой оценки. Повторный выбор порогов по test запрещён.
Проверьте различия Wikimedia/manual, ошибки POI и лицензионную атрибуцию.
Ни аугментации, ни оптимизация весов на этом этапе не выполняются.

[Официальные веса и transforms TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
'''),
code('''display(Markdown((ROOT/'reports/baseline_mobilenetv3/summary.md').read_text(encoding='utf-8')))
''')
], metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python'}, 'baseline':'mobilenet_v3_small_imagenet1k_v1'})
path = ROOT/'notebooks/02_poi_verification_baseline.ipynb'
nbf.validate(book)
nbf.write(book, path)
print(path)
