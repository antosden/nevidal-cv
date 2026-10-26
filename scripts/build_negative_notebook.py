"""Создать ноутбук аудита negatives и отдельного выпуска политики отказа."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
book=nbf.v4.new_notebook(cells=[
md('''# 03. Проверенные external-negatives: аудит, разделение и отказ

Пользователь подтвердил просмотр оставшихся изображений. Это подтверждение записано в manifest,
но технические конфликты с положительным набором дополнительно исключаются.
Negative POI не становятся классом unknown и не добавляются в галерею.
Backbone MobileNetV3-Small, положительные splits и train-галерея сохранены.
'''),
code('''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from IPython.display import display,Markdown,Image as DisplayImage
ROOT=Path.cwd().resolve()
if not (ROOT/'config/wikimedia_negatives.csv').exists():ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from src.poi_baseline import sha256_file,POIVerifier
from src.negative_baseline import recalibrate_with_negatives
OUTPUT=ROOT/'models/baseline_mobilenetv3_negatives'
manifest=pd.read_csv(ROOT/'data/annotations/negative_manifest.csv',keep_default_na=False)
summary=json.loads((ROOT/'reports/negatives/dataset_summary.json').read_text())
display(pd.Series(summary))
'''),
md('''## 1. Происхождение, подтверждённые метки и исключения

`source_poi_id` обозначает посторонний объект и не является целевым классом.
Физически удалённые файлы сохраняются в аудите как rejected / removed_by_user.
Признаки source/author/licence и исходные artist/license URL сохранены.
Сравнение с положительными фото: SHA-256, Commons-ID, pHash и фиксированные кропы.
Потенциальные пересечения исключены консервативно; это не доказательство ошибочной ручной проверки.
'''),
code('''display(manifest.groupby(['source_poi_id','label_status','exclusion_reason'],dropna=False).size().rename('images').to_frame())
display(manifest[['image_id','source_poi_id','source','author','licence','label_status']].head())
display(pd.read_csv(ROOT/'data/annotations/negative_review_queue.csv',keep_default_na=False))
overlap_sheet=ROOT/'reports/negatives/positive_overlap_examples.jpg'
if overlap_sheet.exists():display(DisplayImage(filename=str(overlap_sheet)))
'''),
md('''## 2. Negative-val и negative-test

SEED=42. Целевое соотношение 50/50 приближается с учётом целых POI и связей копий.
Один source POI, серия одного объекта и группа дубликатов не попадают в обе выборки.
Распределения negative_type могут отличаться: при единственном объекте типа
невозможно представить его в обоих splits без пересечения идентичности.
'''),
code('''negatives={s:pd.read_csv(ROOT/f'data/annotations/negative_{s}.csv',keep_default_na=False) for s in ['val','test']}
for field in ['image_id','sha256','commons_file_id','source_poi_id','duplicate_group','split_group']:
    assert not (set(negatives['val'][field])-{''})&(set(negatives['test'][field])-{''}),field
display(pd.crosstab(manifest.loc[manifest.active,'negative_type'],manifest.loc[manifest.active,'split']))
print('Negative POI val:', sorted(negatives['val'].source_poi_id.unique()))
print('Negative POI test:', sorted(negatives['test'].source_poi_id.unique()))
'''),
md('''## 3. Новая калибровка без дообучения

Только positive-val + negative-val. Цели: known-wrong FAR ≤1%, hard-wrong FAR ≤5%,
external any-POI FAR ≤1%. Из допустимых порогов выбирается максимальный true accept.
External фото считается ошибочно принятым, если подтверждается хотя бы один из поддерживаемых POI.
Это строже средней ошибки по всем парам «фото — цель».
Test оценивается после фиксации порогов; его результаты не меняют пороги.
Повторный запуск читает зафиксированный выпуск, проверяя его артефакты и текущие negative splits.
'''),
code('''if not (OUTPUT/'metadata.json').exists():
    policy,metrics=recalibrate_with_negatives(ROOT,ROOT/'models/baseline_mobilenetv3',OUTPUT)
else:
    meta=json.loads((OUTPUT/'metadata.json').read_text())
    for name,digest in meta['artifact_sha256'].items():assert sha256_file(OUTPUT/name)==digest,name
    for split,digest in meta['negative_split_sha256'].items():assert sha256_file(ROOT/f'data/annotations/negative_{split}.csv')==digest
    policy=json.loads((OUTPUT/'rejection_policy.json').read_text())
    metrics=json.loads((OUTPUT/'metrics.json').read_text())
display(pd.Series(policy))
comparison=[]
for name in ['val','test']:
    old=json.loads((ROOT/'models/baseline_mobilenetv3/metrics.json').read_text())[name]
    comparison.append({'split':name,'old_true_accept':old['true_accept_rate'],'new_true_accept':metrics[name]['true_accept_rate'],
                       'new_hard_wrong_far':metrics[name]['hard_wrong_far'],'external_any_poi_far':metrics[f'negative_{name}']['any_poi_far']})
display(pd.DataFrame(comparison))
'''),
md('''## 4. Ошибки отказа на independent negative-test

Все исходные фотографии остаются на месте. Результаты предназначены для диагностики;
после просмотра ошибок test не используют для ручной настройки порогов текущего выпуска.
Набор содержит прежде всего достопримечательности Commons; обычная мобильная съёмка требует отдельной проверки.
'''),
code('''predictions=pd.read_csv(OUTPUT/'negative_test_predictions.csv',keep_default_na=False)
display(predictions[predictions.accepted_any_poi][['image_id','source_poi_id','predicted_poi_id','score','margin']])
display(pd.read_csv(OUTPUT/'negative_test_groups.csv'))
for name in ['negative_test_false_accepts.jpg','test_score_distribution.png']:
    path=ROOT/'reports/baseline_mobilenetv3_negatives'/name
    if path.exists():display(DisplayImage(filename=str(path)))
display(Markdown((ROOT/'reports/baseline_mobilenetv3_negatives/summary.md').read_text(encoding='utf-8')))
'''),
md('''## 5. Проверка своей цели

Используйте новый каталог артефактов; предыдущая версия baseline сохранена отдельно.
Неизвестный target_poi_id возвращает target_poi_not_supported.
'''),
code('''verifier=POIVerifier(OUTPUT)
example=negatives['test'].iloc[0]
display(pd.Series(verifier.verify(ROOT/example.file_path,verifier.poi_ids[0])))
# verifier.verify(Path('path/to/photo.jpg'),'moscow/ostankinskiy/ostankino_tower')
''')
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
path=ROOT/'notebooks/03_negative_dataset_and_rejection.ipynb'
nbf.validate(book);nbf.write(book,path);print(path)
