"""Сборка выполненного ноутбука по сохранённым результатам H1."""
from pathlib import Path
import sys, os, re
import nbformat as nb
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path('I:/Невидаль')
sys.path.insert(0,str(ROOT/'scripts'))
import h1_research as h1
RUN=h1.RUN

def build():
    book=nb.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    cells=[]
    def md(s): cells.append(nb.v4.new_markdown_cell(s))
    def code(s): cells.append(nb.v4.new_code_cell(s))
    md('# H1. Ракурсы и качество MobileNetV3\n\nПроверяю, помогает ли разнообразие ракурсов сократить разницу между MobileNetV3 Small и Large. Обучаю новые модели на девяти объектах ВДНХ. Фото из h1_experiment просмотрены и подтверждены вручную.')
    code("""from pathlib import Path
import sys, json, csv, html
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Image
ROOT = Path.cwd()
if not (ROOT/'scripts').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT/'scripts'))
import h1_research as h1
DATA, REPORT, MODELS = h1.DATA, h1.REPORT, h1.MODELS
read, load = h1.read, h1.load
h1.verify()
def table(rows, columns=None):
    rows = list(rows)
    if not rows:
        print('Нет строк'); return
    columns = columns or list(rows[0])
    def fmt(v):
        if isinstance(v, float): return f'{v:.4f}'
        return str(v)
    head = ''.join('<th>'+html.escape(str(k))+'</th>' for k in columns)
    body = ''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'')))+'</td>' for k in columns)+'</tr>' for r in rows)
    display(HTML('<table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table>'))
print('Проверены хеши данных и изображений.')""")
    md('## Данные\n\nВ train-пуле 181 фото. На один запуск беру 108: по 12 на объект. В режиме О все 12 сняты спереди; в режиме Р беру 4 спереди, 4 слева и 4 справа.\n\nValidation содержит 112 фото, test — 260. Они собраны отдельно от h1_experiment. Для Кыргызстана добавлены 19 подходящих фотографий из интернета. Точные копии, найденные похожие кадры и известные общие съёмки объединены при разбиении.')
    code("""table([load(REPORT/'preflight.json')], ['classes','train_pool','val','test'])
mapping = load(DATA/'class_mapping.json')
table([{'poi_id':p, 'class_index':i} for p,i in mapping.items()])
table([r for r in read(REPORT/'train_coverage.csv') if r['seed']=='42'])
table(read(REPORT/'split_coverage.csv'))
display(Image(filename=str(REPORT/'figures/dataset_counts.png')))
display(Image(filename=str(REPORT/'figures/training_examples.jpg')))""")
    md('Освещение, масштаб и перекрытия не везде удалось уравнять между О и Р. Ниже показан остаточный дисбаланс TV: чем меньше значение, тем ближе условия. Поэтому результат относится к подборкам фото, а влияние одного ракурса полностью отделить нельзя.\n\nДля части контрольных снимков неизвестна история съёмки. Это внутренняя оценка: полную независимость съёмок подтвердить нельзя.')
    code("""balance = read(REPORT/'balance.csv')
table([{'poi_id':p, **{k:max(float(r[k]) for r in balance if r['poi_id']==p) for k in ['TV_illumination','TV_scale','TV_occlusion']}} for p in mapping])""")
    md('## Что проверяю\n\nОсновная метрика — Macro-F1 по девяти объектам. Сначала усредняю результаты трёх seed, затем считаю:\n\n- GО = F1(Large, О) − F1(Small, О).\n- GР = F1(Large, Р) − F1(Small, Р).\n- C = GО − GР.\n\nДля подтверждения нужны оба условия: нижняя граница 95% интервала C ≥ 3 п.п. и верхняя граница GР ≤ 3 п.п. Одних средних значений недостаточно.')
    md('## Обучение\n\nВсего 12 новых моделей: Small и Large, О и Р, seed 42, 43 и 44. Начальные веса — ImageNet V1. Старые обученные модели не использую.\n\nRGB 224 × 224, FP32, обучение всех слоёв и BatchNorm. AdamW: lr = 0.0001, weight decay = 0.0001, batch = 32. Бюджет — 20 эпох, выборка с балансировкой классов, из аугментаций только изменение цвета. Сохраняю эпоху с лучшим validation Macro-F1; при равенстве выбираю более раннюю.')
    code("""config = load(DATA/'config.json')
table([{'параметр':k, 'значение':v} for k,v in config.items()])
table([{'architecture':k, **v} for k,v in load(REPORT/'pretraining_weights.json').items()])
print(load(REPORT/'pilot_decision.json')['reason'])
table(read(REPORT/'pilot_review.csv'))""")
    md('Первые четыре обучения входят в итоговые 12. После пилота оставлен одинаковый бюджет 20 эпох. У части моделей validation ещё меняется к концу обучения, поэтому полную сходимость не заявляю. Test открыт после завершения всех 12 запусков.')
    for seed in [42,43,44]:
        md(f'### Seed {seed}\n\nПоказываю выбор checkpoint, изменение loss и validation Macro-F1. Полные значения всех 20 эпох сохранены в history.csv каждого запуска.')
        for key in h1.CONFIGS:
            code(f"""run = MODELS/'runs/{key}_seed{seed}'
print('{key}, seed {seed}')
table([load(run/'selection.json')])
history = read(run/'history.csv')
print('Поля лога:', ', '.join(history[0]))
fig, axes = plt.subplots(1, 2, figsize=(10, 2.7))
epochs = [int(r['epoch']) for r in history]
loss_key = next(k for k in history[0] if 'loss' in k and 'train' in k)
f1_key = 'macro_f1'
axes[0].plot(epochs, [float(r[loss_key]) for r in history]); axes[0].set_title('Train loss')
axes[1].plot(epochs, [float(r[f1_key]) for r in history]); axes[1].set_title('Validation Macro-F1')
for ax in axes: ax.set_xlabel('Эпоха'); ax.grid(alpha=.2)
fig.tight_layout(); plt.show(); plt.close(fig)
table(history[-5:])""")
    md('## Результаты на test\n\nСредние значения и SD рассчитаны по трём seed. SD рассчитано с ddof = 1, показывает разброс запусков и не является доверительным интервалом.')
    code("""summary = read(REPORT/'test_summary.csv')
metrics = read(REPORT/'test_metrics.csv')
table([{**{k:(r[k] if k=='config' else float(r[k])) for k in r}, 'accuracy_SD':float(np.std([float(t['accuracy']) for t in metrics if t['config']==r['config']], ddof=1))} for r in summary])
table(read(REPORT/'test_metrics.csv'))
table(read(REPORT/'per_seed_contrasts.csv'))
point = load(REPORT/'inference.json')['point']
table([{'показатель':k, 'п.п.':100*point[k]} for k in ['G_O','G_R','C','small_gain','large_gain']])""")
    md('### Ошибки по объектам и ракурсам\n\nМатрицы ошибок и F1 по объектам усреднены по трём seed. В срезах остаются все девять классов ответа. В проверке leave-one-POI-out исключаю один объект из оценки и усредняю F1 по восьми оставшимся; заново модели не обучаю.')
    code("""display(Image(filename=str(REPORT/'figures/confusion_matrices.png')))
per_poi = read(REPORT/'per_poi_metrics.csv')
table([{'poi_id':p, **{k:np.mean([float(r['f1']) for r in per_poi if r['poi_id']==p and r['config']==k]) for k in h1.CONFIGS}} for p in mapping])
print('Средний recall по объектам')
table([{'poi_id':p, **{k:float(np.mean([float(r['recall']) for r in per_poi if r['poi_id']==p and r['config']==k])) for k in h1.CONFIGS}} for p in mapping])
views = read(REPORT/'viewpoint_slices.csv')
groups = sorted(set((r['config'],r['viewpoint']) for r in views))
table([{'config':k, 'viewpoint':v, **{m:float(np.mean([float(r[m]) for r in views if r['config']==k and r['viewpoint']==v])) for m in ['n','macro_f1','accuracy']}} for k,v in groups])
table(read(REPORT/'leave_one_poi_out.csv'))""")
    md('### Устойчивость результата\n\nДополнительно сравниваю результаты трёх seed и проверяю, как меняется вывод при исключении одного объекта из оценки. Это описательный анализ, добавленный после основного расчёта. Он показывает разброс на имеющихся данных и не заменяет проверку по доверительным интервалам.')
    code("""table([{k:(v if k=='metric' else float(v)) for k,v in r.items()} for r in read(REPORT/'seed_stability.csv')])
print('Все значения таблицы — в процентных пунктах.')
print(load(REPORT/'stability_summary.json'))""")
    md('Сокращение разрыва C положительное при всех трёх seed. Но оба точечных условия выполняются вместе только в одном из трёх запусков. В проверке без одного объекта C положительное в восьми из девяти случаев. Без «Робостанции» C = −0.83 п.п., поэтому общий вывод чувствителен к составу объектов.\n\nИсходный план предусматривал 5000 парных bootstrap-повторов по сеансам съёмки. У 225 тестовых фото сеанс неизвестен, поэтому такие интервалы не рассчитаны. Результаты трёх seed и проверки по объектам показывают другие стороны устойчивости; статистического подтверждения исходной гипотезы они не дают.')
    md('## Экспорт и скорость\n\nДля сравнения заранее выбраны модели Р с seed 42. Обе экспортированы в FP32 TorchScript. Классы после экспорта совпали на всём test.\n\nЗамеры сделаны на CPU компьютера: 20 прогревов, затем 3 серии по 100 запусков с чередованием моделей. Отдельно показан проход модели и полный путь с чтением и подготовкой фото. На телефоне замеров нет.')
    code("""table(read(REPORT/'export_parity.csv'))
table(read(REPORT/'resources.csv'))
table([{'параметр':k,'значение':v} for k,v in load(REPORT/'mobile_status.json').items()])""")
    md('## Вывод\n\nВ режиме Р средний Macro-F1 Small — 0.5729, Large — 0.6064. Разрыв GР — 3.35 п.п., сокращение разрыва C — 4.24 п.п. Среднее отставание Small превышает заданные 3 п.п. Подтверждения составной гипотезы на этих данных не получено.\n\nSmall прибавила 2.97 п.п., а Large потеряла 1.28 п.п. Значит, разрыв уменьшился в том числе из-за ухудшения Large.\n\nВывод относится к этим девяти объектам и бюджету 20 эпох. Для выбора модели приложения нужны требования к качеству и p95, а также замеры на телефоне.')
    md('## Как повторить\n\nКод эксперимента — scripts/h1_research.py. Порядок команд: prepare с новым bundle и run_id, pilot, проверка пилота, main, evaluate, notebook. Конфигурация и манифесты этого запуска сохранены в data/annotations/h1_20261104_v1.\n\nПосле открытия test настройки этого запуска не меняю. Новый эксперимент запускается отдельно. Графики и таблицы этого ноутбука построены по сохранённым логам, checkpoint и предсказаниям.')
    book.cells=cells
    for c in cells:
        if c.cell_type=='markdown':
            assert len(c.source)<1000
            assert not re.search(r'https?://|[\U0001F000-\U0001FAFF\u2600-\u27BF]',c.source)
    path=ROOT/'notebooks/H1_mobilenetv3_viewpoint_research.ipynb'
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    manager.kernel_spec.env={**os.environ,'PYTHONIOENCODING':'utf-8','MPLCONFIGDIR':str(ROOT/'.runtime'/RUN/'matplotlib'),'IPYTHONDIR':str(ROOT/'.runtime'/RUN/'ipython')}
    try: NotebookClient(book,km=manager,timeout=240,resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        if manager.has_kernel: manager.shutdown_kernel(now=True)
    assert all(c.execution_count for c in cells if c.cell_type=='code')
    assert not [o for c in cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
    nb.validate(book);nb.write(book,path)
    h1.dump(h1.REPORT/'notebook_status.json',dict(executed=True,errors=0,path=str(path)))
    print('Notebook executed:',path,flush=True)
    return path

if __name__=='__main__': build()
