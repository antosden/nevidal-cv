"""Ноутбук геометрической проверки экземпляра POI."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
book=nbf.v4.new_notebook(cells=[
md('''# 06. MobileNetV3 + SIFT/RANSAC: проверка конкретного экземпляра POI

Предыдущие варианты путали похожие здания. Здесь проверяется согласованное расположение локальных деталей.
MobileNetV3/Ridge предлагает 5 POI; для каждого проверяются 3 ближайших train-эталона.
SIFT извлекается без обучения и скачивания весов. FLANN/ratio test и RANSAC дают геометрическую поддержку.
Исходники, прежние модели и разбиения сохраняются.
'''),
code('''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from IPython.display import display,Markdown,Image as DisplayImage
ROOT=Path.cwd().resolve()
if not (ROOT/'config/mobilenetv3_geometric.json').exists():ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from src.poi_baseline import sha256_file
from src.poi_geometry import GeometricPOIVerifier
OUTPUT=ROOT/'models/mobilenetv3_geometric'
assert (OUTPUT/'metadata.json').exists(), 'Сначала завершите scripts/geometric_poi.py'
metadata=json.loads((OUTPUT/'metadata.json').read_text())
for name,digest in metadata['artifact_sha256'].items():assert sha256_file(OUTPUT/name)==digest,name
for name,digest in metadata['input_split_sha256'].items():assert sha256_file(ROOT/f'data/annotations/{name}.csv')==digest,name
display(pd.Series(json.loads((OUTPUT/'config.json').read_text())))
'''),
md('''## 1. Протокол

Train-галерея состоит из прежних 404 эталонов, с source/author/licence. Новые POI и negatives в неё не добавляются.
Локальные признаки вычисляются на EXIF-corrected grayscale копиях с длинной стороной ≤960.
Ratio test=0.72. Разные query keypoints не голосуют за один ref keypoint.
Геометрическая поддержка требует ≥8 RANSAC inliers и покрытия ≥2% каждого кадра.
Это критерий решения модели, а не автоматическое исключение изображений из датасета.

Сравниваются два способа агрегации и две fusion-стратегии:
`geometry_only` требует геометрии; `hybrid_ridge` добавляет Ridge-отклик и допускает строгий fallback.
Глобальные score/margin пороги и вариант выбираются только на positive-val/negative-val,
с ограничениями external any-POI FAR ≤1%, known hard-wrong FAR ≤5% на validation.
Test вычисляется после фиксации selection.json. Пороги по test не подстраиваются.
'''),
code('''display(pd.read_csv(OUTPUT/'validation_candidates.csv'))
display(pd.Series(json.loads((OUTPUT/'selection.json').read_text())))
display(pd.Series(json.loads((OUTPUT/'rejection_policy.json').read_text())))
display(pd.read_csv(OUTPUT/'gallery.csv',keep_default_na=False)[['image_id','poi_id','source','author','licence']].head())
'''),
md('''## 2. Test: классификация, true accept и external FAR

Без геометрии ranking может опираться на coarse Ridge; это не равнозначно подтверждению цели.
Score — непробабилистическая комбинация геометрической согласованности и, в hybrid, Ridge-отклика.
Глобальный FAR не гарантирует одинаковое качество для каждого POI или невиданных типов negatives.
'''),
code('''metrics=json.loads((OUTPUT/'metrics.json').read_text())
display(pd.DataFrame([{'split':s,'accuracy':metrics[s]['accuracy'],'macro_f1':metrics[s]['macro_f1'],
 'true_accept':metrics[s]['true_accept_rate'],'external_any_poi_far':metrics[f'negative_{s}']['any_poi_far']}
 for s in ['val','test']]))
display(pd.read_csv(OUTPUT/'test_per_poi.csv'))
display(pd.read_csv(OUTPUT/'negative_test_groups.csv'))
print('Статус:',metadata['deployment_status'])
print('Время только matching, без SIFT/decode/encoder:',metadata['matching_only_latency_seconds'])
display(DisplayImage(filename=str(ROOT/'reports/mobilenetv3_geometric/test_tradeoff.png')))
'''),
md('''## 3. Разбор решений

Файлы `*_evidence.csv` содержат query ID, цель, ref ID, число совпадений и inliers,
доли покрытия и geometric_score. Таблицы `*_predictions.csv` сохраняют разметку и атрибуцию.
Смотреть ошибки test можно для диагностики; текущий выпуск после этого не перенастраивается.
'''),
code('''evidence=pd.read_csv(OUTPUT/'test_evidence.csv',keep_default_na=False)
display(evidence.sort_values('geometric_score',ascending=False).head(20))
negative=pd.read_csv(OUTPUT/'negative_test_predictions.csv',keep_default_na=False)
display(negative[negative.accepted_any_poi][['image_id','source_poi_id','predicted_poi_id','score','margin']])
'''),
md('''## 4. Одиночная проверка

Seed зависит только от train-эталона, поэтому одинаковые пиксели дают одинаковую геометрию
в пакетном и одиночном выводе независимо от ID или имени загружаемого файла.
'''),
code('''verifier=GeometricPOIVerifier(OUTPUT)
test=pd.read_csv(OUTPUT/'splits/test.csv',keep_default_na=False)
example=test.iloc[0]
display(pd.Series(verifier.verify(ROOT/example.file_path,example.poi_id)))
# verifier.verify(Path('path/to/photo.jpg'),'moscow/ostankinskiy/ostankino_tower')
'''),
md('''## 5. Ограничения

Homography подходит для локально плоского фрагмента; другие стороны здания, вода, растения и освещение
могут не совпасть. Съёмка карт/экранов и повторяющиеся фасадные детали остаются отдельными рисками.
Схема дороже по CPU и памяти, чем один MobileNet; полную задержку нужно измерить на целевом оборудовании.
Положительный test уже был диагностическим в прошлых экспериментах; нужен новый закрытый полевой набор.
Цель 85% true accept не объявляется достигнутой без измерения.

[OpenCV matching](https://docs.opencv.org/4.10.0/dc/dc3/tutorial_py_matcher.html).
[OpenCV homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html).
'''),
code('''display(Markdown((ROOT/'reports/mobilenetv3_geometric/summary.md').read_text(encoding='utf-8')))
''')
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
path=ROOT/'notebooks/06_mobilenetv3_geometric_verification.ipynb'
nbf.validate(book);nbf.write(book,path);print(path)
