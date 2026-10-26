"""Собрать ноутбук сравнения encoder и галерей без скрытого обучения."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
book=nbf.v4.new_notebook(cells=[
md('''# 07. Сравнение encoder и train-галерей после полного обогащения manual

MobileNetV3-Small (576D), MobileNetV3-Large V2 (960D), DINOv2 ViT-S/14 CLS (384D).
Frozen веса, L2 embeddings, одинаковый max-gallery-cosine score и глобальный отказ.
Размеры: 5/10/20/all независимых train-групп на POI; diverse/farthest-first и random SEED=42.
Все исходники остаются неизменными; source/author/licence сохранены.

Отдельные preprocessing encoder фиксируются в weights.json. DINOv2 использует 224 px для CPU:
это воспроизводимый стартовый вариант, а не максимально возможное качество DINOv2.
Пороги и галерея выбираются только по positive-val/negative-val при external FAR ≤1% на val.
Test после этого используется диагностически. Он уже участвовал в предыдущих экспериментах;
нужен новый закрытый полевой набор. Автоматической замены используемой модели нет.
'''),
code('''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
ROOT=Path.cwd().resolve()
if not (ROOT/'config/encoder_comparison.json').exists():ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from src.poi_baseline import sha256_file
OUTPUT=ROOT/'models/encoder_comparison_manual_all'
status=json.loads((OUTPUT/'status.json').read_text())
display(pd.Series(status,name='status'))
snapshot=json.loads((OUTPUT/'input_snapshot.json').read_text())
for name in ['train','val','test','negative_val','negative_test']:
    assert sha256_file(ROOT/f'data/annotations/{name}.csv')==snapshot[name],name
display(pd.read_csv(ROOT/'reports/manual_enrichment_plan.csv'))
'''),
md('''## 1. Полнота сравнения и проверка артефактов

Если веса отсутствуют, соответствующий encoder остаётся missing_weights. Его метрики не выдумываются.
После появления официальных весов в .runtime/torch/checkpoints повторите scripts/compare_encoders.py:
завершённые encoder не меняются, недостающие продолжают общий протокол.
'''),
code('''for name,state in status.items():
    if state!='complete':continue
    metadata=json.loads((OUTPUT/name/'metadata.json').read_text())
    for file,digest in metadata['artifact_sha256'].items():assert sha256_file(OUTPUT/name/file)==digest,file
    selection=json.loads((OUTPUT/name/'selection.json').read_text())
    assert not selection['test_used_for_selection']
    print(name,selection['variant'],'— hashes OK')
display(pd.read_csv(ROOT/'reports/encoder_comparison/comparison.csv'))
'''),
md('''## 2. Все варианты галерей: только validation

Оптимизация по нескольким вариантам делает val-результат оптимистичным. Не выбирайте вариант по test.
При all сохраняется один представитель независимой группы на POI; разные near-копии не считаются
дополнительными независимыми эталонами. Random детерминирован при SEED=42.
'''),
code('''candidates=pd.read_csv(ROOT/'reports/encoder_comparison/validation_candidates.csv')
display(candidates[['encoder','variant','gallery_images','accuracy','macro_f1','true_accept_rate','external_any_poi_far']])
fig,ax=plt.subplots(figsize=(9,5))
caps=['5','10','20','all']
for name,group in candidates.groupby('encoder'):
    for method in ['diverse','random']:
        local=group[group.variant.str.startswith(method+'_')].copy()
        local['cap']=local.variant.str.split('_').str[-1].map({v:i for i,v in enumerate(caps)})
        local=local.sort_values('cap')
        ax.plot(local.cap,local.true_accept_rate*100,marker='o',linestyle='-' if method=='diverse' else '--',label=f'{name}/{method}')
ax.axhline(85,color='gray',linestyle=':',label='TAR target 85%')
ax.set(xlabel='Gallery cap per POI (independent train groups)',ylabel='Validation true accept, %',ylim=(0,100))
ax.set_xticks(range(4),caps);ax.legend(fontsize=8);fig.tight_layout()
fig.savefig(ROOT/'reports/encoder_comparison/validation_tradeoff.png',dpi=140)
plt.show()
'''),
md('''## 3. Per-POI диагностика выбранных по validation вариантов

Accuracy и macro-F1 оценивают распознавание без отказа; true accept — подтверждение правильной цели.
Их нельзя подменять друг другом. Строки test служат для отчёта; пороги после них не меняются.
'''),
code('''for name,state in status.items():
    if state!='complete':continue
    print(name)
    display(pd.read_csv(OUTPUT/name/'test_per_poi.csv'))
display(Markdown((ROOT/'reports/encoder_comparison/summary.md').read_text(encoding='utf-8')))
'''),
md('''## 4. Повторение

Из корня проекта: `python scripts/compare_encoders.py`, затем этот Notebook → Run All.
Поддерживается `--extra-site .venv/Lib/site-packages` для рабочего Python.
Новый состав данных или конфигурация требуют нового --output. Чтобы открыть другой выпуск,
измените OUTPUT и путь reports в ноутбуке. Новые веса автоматически не скачиваются.

Источники архитектур и весов:
- https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html
- https://github.com/facebookresearch/dinov2

Eval-only DINOv2 реализован через стандартные PyTorch-операции с именами параметров upstream;
официальный checkpoint загружается strict=True. Дальнейшие оптимизации должны проверяться
на совпадение с официальной реализацией перед эксплуатацией.
''')
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
nbf.validate(book);nbf.write(book,ROOT/'notebooks/07_encoder_gallery_comparison.ipynb')
