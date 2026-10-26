"""Сформировать ноутбук дообучения и оценки без подбора по test."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
book=nbf.v4.new_notebook(cells=[
md('''# 04. Дообучение MobileNetV3-Small для конкретных POI

Задача: поднять true accept при ограниченном FAR. Веса обучаются только на positive-train.
Negative-val используется для выбора эпохи и порогов, negative-test — только для оценки.
Прежние модели и исходные фотографии сохраняются.
Эксперимент не гарантирует достижения 85%; результат фиксируется до просмотра test.
'''),
code('''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from IPython.display import display,Markdown,Image as DisplayImage
ROOT=Path.cwd().resolve()
if not (ROOT/'config/mobilenetv3_finetune.json').exists():ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from src.poi_finetune import train_model,FinetunedPOIVerifier
from src.poi_baseline import sha256_file
OUTPUT=ROOT/'models/mobilenetv3_finetuned'
CONFIG=ROOT/'config/mobilenetv3_finetune.json'
display(pd.Series(json.loads(CONFIG.read_text())))
'''),
md('''## 1. Протокол обучения

ImageNet1K V1 → последние четыре блока Small; BatchNorm frozen.
Сбалансированные батчи: 8 POI × 3 независимых фото. Loss: cosine classification CE + 0.15 SupCon.
Умеренные аугментации применяются только к train после разделения.
Preprocessing сохраняет весь кадр: EXIF → JPEG draft 640 → thumbnail 320 → letterbox 224.
В каждой эпохе заново строится train-галерея и калибруется отказ на validation.
Сравниваются identity, центрирование и регуляризованное PCA-whitening до 128D;
среднее и матрица преобразования вычисляются только из positive-train эмбеддингов.
Выбор checkpoint: максимальный val true accept при ограничениях FAR, затем macro-F1 и accuracy.
Точка epoch 0 — контроль frozen encoder с тем же новым preprocessing.
Test-фото декодируются только после фиксации эпохи и порогов.

Run All повторно использует завершённый выпуск с проверкой хэшей. Новый эксперимент требует другой OUTPUT.
'''),
code('''if not OUTPUT.exists():
    metadata,metrics=train_model(ROOT,OUTPUT,CONFIG)
else:
    assert (OUTPUT/'metadata.json').exists(), 'Обучение ещё не завершено; дождитесь CLI'
    metadata=json.loads((OUTPUT/'metadata.json').read_text())
    assert json.loads(CONFIG.read_text())==json.loads((OUTPUT/'config.json').read_text()),'Конфигурация изменена: нужен новый выпуск'
    for name,digest in metadata['artifact_sha256'].items():assert sha256_file(OUTPUT/name)==digest,name
    for name,digest in metadata['input_split_sha256'].items():assert sha256_file(ROOT/f'data/annotations/{name}.csv')==digest,name
    metrics=json.loads((OUTPUT/'metrics.json').read_text())
selection=json.loads((OUTPUT/'selection.json').read_text())
display(pd.Series(selection))
display(pd.read_csv(OUTPUT/'training_history.csv'))
display(DisplayImage(filename=str(ROOT/'reports/mobilenetv3_finetuned/training_curves.png')))
'''),
md('''## 2. Отложенная оценка

Accuracy/macro-F1 ниже описывают выбор лидирующего POI без отказа. True accept дополнительно
учитывает прохождение порогов. External FAR считает любое подтверждение одного из 23 POI на постороннем фото.
Изменения по диагностическому test не используются для повторного выбора checkpoint или порогов.
'''),
code('''table=[]
for name in ['val','test']:
    table.append({'split':name, 'accuracy':metrics[name]['accuracy'],'macro_f1':metrics[name]['macro_f1'],
                  'true_accept':metrics[name]['true_accept_rate'],'hard_wrong_far':metrics[name]['hard_wrong_far'],
                  'external_any_poi_far':metrics[f'negative_{name}']['any_poi_far']})
display(pd.DataFrame(table))
display(pd.read_csv(OUTPUT/'test_per_poi.csv'))
display(pd.read_csv(OUTPUT/'negative_test_groups.csv'))
'''),
md('''## 3. Локальная верификация

Загрузчик использует дообученные encoder.pt, новую галерею и соответствующие пороги/preprocessing.
Автоматического переключения прежней версии приложения не выполняется.
'''),
code('''verifier=FinetunedPOIVerifier(OUTPUT)
test=pd.read_csv(OUTPUT/'splits/test.csv',keep_default_na=False)
example=test.iloc[0]
display(pd.Series(verifier.verify(ROOT/example.file_path,example.poi_id)))
# verifier.verify(Path('path/to/photo.jpg'),'moscow/ostankinskiy/ostankino_tower')
'''),
md('''## 4. Ограничения и отчёт

Малые классы и повторное использование val ограничивают уверенность в качестве.
Положительный test уже служил диагностике предыдущих экспериментов; нужен новый закрытый полевой набор.
Negative-test не покрывает все редкие типы. Источники, авторы и лицензии сохранены в splits и галерее.

[Transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
[Supervised contrastive learning](https://arxiv.org/abs/2004.11362).
'''),
code('''display(Markdown((ROOT/'reports/mobilenetv3_finetuned/summary.md').read_text(encoding='utf-8')))
''')
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
path=ROOT/'notebooks/04_mobilenetv3_finetuning.ipynb'
nbf.validate(book);nbf.write(book,path);print(path)
