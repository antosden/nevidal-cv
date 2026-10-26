"""Ноутбук сравнения MobileNetV3-классификатора и per-POI отказа."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
book=nbf.v4.new_notebook(cells=[
md('''# 05. MobileNetV3: улучшение классификатора и калибровка POI

Эксперимент сравнивает gallery cosine и сбалансированный Ridge на исходных MobileNetV3-Small эмбеддингах.
Линейный слой обучается только на positive-train. Alpha и per-POI score/margin пороги выбираются на val/negative-val.
Test не влияет на выбор. Предыдущий CNN fine-tuning не улучшил validation-верификацию и не был принят.
Достижение accuracy около 85% не равнозначно true accept 85%.
'''),
code('''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from IPython.display import display,Markdown
ROOT=Path.cwd().resolve()
if not (ROOT/'config/mobilenetv3_calibrated.json').exists():ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from src.poi_metric import export_calibrated,CalibratedPOIVerifier
from src.poi_baseline import sha256_file
OUTPUT=ROOT/'models/mobilenetv3_calibrated'
if not OUTPUT.exists():export_calibrated(ROOT,OUTPUT)
metadata=json.loads((OUTPUT/'metadata.json').read_text())
for name,digest in metadata['artifact_sha256'].items():assert sha256_file(OUTPUT/name)==digest,name
for name,digest in metadata['input_split_sha256'].items():assert sha256_file(ROOT/f'data/annotations/{name}.csv')==digest,name
metrics=json.loads((OUTPUT/'metrics.json').read_text())
display(pd.Series(metadata))
'''),
md('''## 1. Выбор только по validation

Классы имеют равный суммарный вес при обучении Ridge, число фотографий в большом классе не доминирует.
Ответ регрессии может выходить за [0,1]; он не объявляется вероятностью.
Каждому POI подбираются два порога. Динамическое программирование распределяет **общий**, а не отдельный
для каждого класса бюджет false accepts на negative-val и ошибочных известных целях.
В малых классах такая калибровка может переучитывать validation; test нужен для выявления этого риска.
'''),
code('''display(pd.read_csv(OUTPUT/'validation_candidates.csv'))
display(pd.Series(json.loads((OUTPUT/'selection.json').read_text())))
thresholds=json.loads((OUTPUT/'thresholds_by_poi.json').read_text())
display(pd.DataFrame(thresholds).T)
'''),
md('''## 2. Test: классификация и отказ оцениваются отдельно

Accuracy и macro-F1 описывают выбор одного из 23 POI без отказа.
True accept учитывает пороги для правильной цели.
External any-POI FAR считает negative-фото ошибкой, если принято хотя бы за один целевой POI.
Превышение FAR не исправляется подбором по test и не маскируется средней ошибкой по всем парам.
'''),
code('''display(pd.DataFrame([{ 'split':name,'accuracy':metrics[name]['accuracy'],'macro_f1':metrics[name]['macro_f1'],
 'true_accept':metrics[name]['true_accept_rate'],'external_any_poi_far':metrics[f'negative_{name}']['any_poi_far']}
 for name in ['val','test']]))
display(pd.read_csv(OUTPUT/'test_per_poi.csv'))
display(pd.read_csv(OUTPUT/'negative_test_groups.csv'))
print('Статус выпуска:', metadata['deployment_status'])
'''),
md('''## 3. Локальная проверка цели

При Ridge решении галерея используется для ближайшего эталона и происхождения; итоговый score принадлежит
обученному линейному слою. Это отличается от исходного max-cosine baseline и зафиксировано в metadata.
Автоматического переключения используемой версии нет.
'''),
code('''verifier=CalibratedPOIVerifier(OUTPUT)
test=pd.read_csv(OUTPUT/'splits/test.csv',keep_default_na=False).sort_values('image_id')
example=test.iloc[0]
display(pd.Series(verifier.verify(ROOT/example.file_path,example.poi_id)))
'''),
md('''## 4. Вывод

Классификация может улучшиться одновременно с ухудшением отказа на посторонних POI.
Кандидат, нарушивший ограничение FAR, не рекомендуется как замена строгой политики.
Для повышения true accept при низком FAR нужны больше независимых данных и проверка на новом закрытом полевом test.
Источник, автор и лицензия всех снимков сохранены в snapshots и галерее.
'''),
code('''display(Markdown((ROOT/'reports/mobilenetv3_calibrated/summary.md').read_text(encoding='utf-8')))
''')
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
path=ROOT/'notebooks/05_mobilenetv3_classifier_and_calibration.ipynb'
nbf.validate(book);nbf.write(book,path);print(path)
