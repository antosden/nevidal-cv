"""Читаемые вычисленные результаты нового исследования; старые файлы не меняются."""
from collections import Counter
from pathlib import Path
import html
import json
import re

from src.viewpoint_audit import read,write,dump,folders,RUN_ID


def table(rows,columns=None):
    if not rows:return '_Данных пока нет._'
    columns=columns or list(rows[0])
    def cell(value):
        if isinstance(value,str) and re.fullmatch(r'-?\d+\.\d+(?:[eE][+-]?\d+)?',value):value=float(value)
        if isinstance(value,float):return f'{value:.4f}'
        return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(cell(r.get(c,'')) for c in columns)+' |' for r in rows])


def run_display(root,key,seed):
    from IPython.display import display,Markdown,Image
    _,_,models,_=folders(Path(root));run=models/'runs'/f'{key}_seed{seed}'
    if not (run/'selection.json').exists():
        display(Markdown('⏳ Ещё не завершено.'));return
    status=json.loads((run/'status.json').read_text())
    selection=json.loads((run/'selection.json').read_text())
    history=read(run/'history.csv')
    display(Markdown(table([dict(config=key,seed=seed,status=status['status'],epochs=len(history),
        best_epoch=selection['epoch'],val_macro_f1=selection['validation']['macro_f1'],
        val_accuracy=selection['validation']['accuracy'])])))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,2.3));ax.plot([int(r['epoch']) for r in history],[float(r['macro_f1']) for r in history],label='validation Macro-F1')
    ax2=ax.twinx();ax2.plot([int(r['epoch']) for r in history],[float(r['train_loss']) for r in history],color='gray',alpha=.5)
    ax.set_ylim(0,1);ax.set_xlabel('epoch');ax.set_ylabel('Macro-F1');ax2.set_ylabel('train CE');ax.grid(alpha=.2);plt.show()


def write_report(root):
    root=Path(root);data,report,models,_=folders(root)
    selection=json.loads((report/'selection_status.json').read_text(encoding='utf-8'))
    inference=json.loads((report/'inference.json').read_text(encoding='utf-8'))
    point=inference['point'];status=json.loads((report/'experiment_status.json').read_text())
    summary=read(report/'test_summary.csv')
    included=read(report/'included_pois.csv');audit=json.loads((report/'audit.json').read_text(encoding='utf-8')) if (report/'audit.json').exists() else {}
    mapping=json.loads((data/'class_mapping.json').read_text(encoding='utf-8'))
    included=[dict(class_index=mapping[r['poi_id']],**r) for r in included]
    direction='сократился' if point['C']>=0 else 'увеличился'
    small_change='выросло' if point['small_gain']>=0 else 'снизилось'
    large_change='выросло' if point['large_gain']>=0 else 'снизилось'
    numerical='выполнены' if point['C']>=.03 and point['G_R']<=.03 else 'не выполнены'
    interpretation=(f"В точечных оценках разрыв Large − Small {direction} на {100*abs(point['C']):.2f} п.п. "
        f"Качество Small {small_change} на {100*abs(point['small_gain']):.2f} п.п., "
        f"Large {large_change} на {100*abs(point['large_gain']):.2f} п.п. "
        f"На Р подписанный разрыв составляет {100*point['G_R']:.2f} п.п. "
        f"Два рабочих порога по одним точечным оценкам {numerical}; решение по протоколу требует интервалов.")
    text=f'''# 🔬 Ракурсы и достаточность MobileNetV3-Small

**Run:** `{RUN_ID}`. **Выполнено {status['completed_runs']}/12 обучений.**

## Что проверяли

При одинаковом числе фото каждого POI сравнили основной вид (О) и три вида поровну (Р).
Гипотеза: разнообразие сокращает подписанный разрыв Large − Small минимум на 3 п.п.,
а отставание Small на Р не превышает 3 п.п.

## Данные и EDA

Аудит: 23 POI, 1875 физических файлов, 19592 строки истории. Визуально просмотрены все
700 разрешённых train-фото, отдельно контрольные фотографии семи отобранных POI.
Проверка выполнена Codex (`codex_visual_review`), независимого второго человека нет.
Неоднозначный ракурс — `undetermined`; в управляемый train такие кадры не входят.
Папочные ошибки и нерелевантные кадры исключены из новых manifests.

Отобрано **{selection['K']} POI**, **84 train** в каждом режиме и seed,
**{selection['val']} validation / {selection['test']} внутренний test**.
Исходный полный каталог не использовался как заранее заданный K.

{table(included)}

Источники О/Р совпадают точно внутри POI; распределения света, масштаба и перекрытий
сопоставлены с TV≤0.20 для каждого признака. При возможности использовались точные
совместные страты. Число известных сеансов отличается не более чем на один,
доля кадров с неизвестным сеансом — не более чем на 0.20. Фактический баланс:
`selection_balance.csv`. Это сопоставление не гарантирует чистый причинный эффект ракурса.
Исключённые 16 POI и фактические числа main/left/right: `excluded_pois.csv`.

## Обучение

Small/Large × О/Р × seeds 42,43,44. Новые ImageNet V1 checkpoints, RGB224, FP32,
все слои и BatchNorm обучаются; AdamW lr=1e-4, wd=1e-4, batch32,20 эпох.
Class-balanced sampler с возвращением, ровно 84 выборки и 3 шага на эпоху.
Только умеренные цветовые изменения. Checkpoint — максимум validation Macro-F1,
при равенстве более ранняя эпоха. Seed меняет отбор и обучение.
Сначала четыре seed42 пилота; решение по development: `pilot_decision.json`.
Новый sampler и правило равенства отличаются от предыдущего исследования.

## Фактическое качество

Среднее трёх отдельных классификаторов; ансамбль не использовался.
SD — выборочное стандартное отклонение по трём seeds, ddof=1.

{table(summary)}

| Показатель | Значение, п.п. |
| --- | ---: |
| GО = LargeО − SmallО | {100*point['G_O']:.2f} |
| GР = LargeР − SmallР | {100*point['G_R']:.2f} |
| C = GО − GР | {100*point['C']:.2f} |
| Прирост Small: Р − О | {100*point['small_gain']:.2f} |
| Прирост Large: Р − О | {100*point['large_gain']:.2f} |

Все seeds: `test_metrics.csv`; F1/recall по POI: `per_poi_metrics.csv`;
срезы ракурсов с полным mapping из семи классов: `viewpoint_slices.csv`.
Все ответы по каждому кадру: `test_predictions.csv`, включая вероятности.

## Статистическая методика и статус

Общий test исторический, ранее использованный: исследование **поисковое внутреннее**.
Схема bootstrap зафиксирована до предсказаний: парные сеансы внутри POI,
общие outings совместно; запрошено 5000 повторов, seed2026, двусторонние 95% интервалы.
Выполнено {inference['performed_replicates']} корректных повторов.
Причины недоступности интервала: {'; '.join(inference['bootstrap_plan']['reasons']) or 'нет'}.
Независимый bootstrap отдельных кадров не подставлялся. CI C: {inference['C_ci95']};
CI GР: {inference['G_R_ci95']}. Неизвестные серии не объявлены независимыми.
Validation содержит все три вида в целом, но не для каждой комбинации POI/ракурс;
покрытие сохранено в `manifest_viewpoint_coverage.csv`.

## Вывод по гипотезе

{interpretation}

**Гипотеза не подтверждена:** без допустимых доверительных интервалов нельзя проверить
совместные условия lower95(C)≥0.03 и upper95(GР)≤0.03.
Точечные результаты выше описывают фактически обученные модели; отсутствие подтверждения
не означает доказанного отсутствия эффекта. Изменение качества Small и Large показано
раздельно: сокращение разрыва из-за падения Large не считается улучшением Small.

## Практический выбор

Сравнение ресурсов — заранее выбранная пара Р/seed42: одинаковый Windows CPU, FP32,
20 прогревов и 100 измерений ×3, p50/p95, размер и число параметров:
`resource_benchmark.csv`. Это не замеры Android. Экспорт новых моделей, требования
к абсолютному качеству и p95 телефона остаются отдельной проверкой. Отказ на чужих фото
в этом исследовании не оценивался. Внедрённая модель не менялась.

## Воспроизводимость

Manifests и их SHA-256: `data/annotations/{RUN_ID}/manifests_lock.json`.
Конфигурация: `final_config.json`; версии и исходные веса: `environment.json`.
Разметка и причины отбора: `selection_decisions.csv`; карточки: `orientation_cards.json`.
Проверены ID/SHA/группы копий/известные серии и outings между splits,
исторические val/test не перенесены в train. SHA-256 прежней работы сверены.
Фото, прежние splits, notebook09, предыдущие и мобильные модели сохранены.
'''
    (report/'research_summary.md').write_text(text,encoding='utf-8')


def create_notebook(root):
    root=Path(root);data,report,models,_=folders(root)
    import nbformat as nbf
    cells=[]
    md=lambda text:cells.append(nbf.v4.new_markdown_cell(text))
    code=lambda text:cells.append(nbf.v4.new_code_cell(text))
    md('''# 🔬 Ракурсы и достаточность Small

**Вопрос:** помогает ли разнообразие видов приблизить лёгкую MobileNetV3-Small к Large?

Для мобильного приложения важны качество и вычислительная стоимость. Проверяем их
раздельно. Модель приложения и прежний notebook09 не меняем.

---
## 🎯 Гипотеза и правила решения

**О** — только основной вид. **Р** — main/left/right поровну при том же N каждого POI.
Четыре конфигурации × seeds42,43,44 = **12 обучений**.

`GО = F(Large,О) − F(Small,О)`; `GР = F(Large,Р) − F(Small,Р)`; `C = GО − GР`.
F — среднее test Macro-F1 трёх классификаторов, не ансамбль.

✅ Поддержка: lower95(C)≥0.03 **и** upper95(GР)≤0.03.
❌ Порог не достигнут: upper95(C)<0.03 либо lower95(GР)>0.03.
❔ Пересечение границ или отсутствие допустимого интервала: неопределённый результат.
Отсутствие разницы не доказывает допустимое отставание.

Источник: [дизайн исследования](https://chatgpt.com/space/page_c6e1f387570c8191b2e0b23740308af7).
Снимок исходного дизайна сохранён рядом с manifests.''')
    code('''%matplotlib inline
from pathlib import Path
import sys, json
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert (ROOT / 'src/viewpoint_training.py').exists()
sys.path.insert(0, str(ROOT))
from IPython.display import display, Markdown, Image, HTML
import html
from src.viewpoint_audit import read, folders
from src.viewpoint_reporting import table, run_display
DATA, REPORT, MODELS, RUNTIME = folders(ROOT)
def show_csv(path, columns=None): display(Markdown(table(read(path), columns)))
def show_json(path): display(HTML('<details><summary>'+html.escape(path.name)+'</summary><pre>'+html.escape(path.read_text(encoding='utf-8'))+'</pre></details>'))
TRAIN = False  # По умолчанию показываем артефакты: повторного обучения нет.
''')
    md('''## 🗂️ EDA и визуальная разметка

Полный каталог проверен до метрик. Визуальный reviewer — `codex_visual_review`;
второй независимой человеческой проверки нет. Кроп/яркость/дистанция не создают новый
ракурс. Для неоднозначной стороны — `undetermined`.

Все исходные фото неизменны. Исторические val/test и связанная прогулка не входят в train.
Сведения source/author/licence сохранены; неизвестные значения не додумываются.''')
    code("show_csv(REPORT / 'reconciled_coverage.csv')\nshow_csv(REPORT / 'viewpoint_coverage.csv')\ndisplay(Image(filename=str(REPORT/'figures/eda.png')))")
    code("show_csv(REPORT / 'included_pois.csv')\nshow_csv(REPORT / 'excluded_pois.csv')")
    code("cards=json.loads((DATA/'orientation_cards.json').read_text(encoding='utf-8'))\nincluded={r['poi_id'] for r in read(REPORT/'included_pois.csv')}\ndisplay(Markdown(table([{'poi_id':r['poi_id'],'main':r['main'],'left':r['left'],'right':r['right']} for r in cards if r['poi_id'] in included])))")
    md('''### Сопоставимость О/Р

N выбирается из 24,21,18,15,12,9,6; в Р каждого вида N/3. Источники совпадают точно.
Для света/масштаба/перекрытий TV≤0.20 (сначала пробуем точные совместные страты).
Поэтому сравнение не объявляется чистым причинным эффектом ракурса.
Правило, каталог, manifests, bootstrap-схема и SHA зафиксированы до обучения.''')
    code("show_json(DATA / 'selection_rule.json')\nshow_csv(REPORT / 'selection_balance.csv')")
    code("show_csv(REPORT / 'manifest_viewpoint_coverage.csv')\nshow_json(REPORT / 'integrity_before_training.json')")
    md('''### Контактные листы и исходники

Все 700 train-кадров: `reports/viewpoint_20261102_v1/figures/train_final`.
Контрольные кадры: `figures/heldout_selected`; просмотренные оригиналы: `figures/original_checks`.
Ниже примеры выбранных и неопределённых видов. CSV индексы связывают плитки с image_id.''')
    code("for name in ['selected_O_seed42_01.jpg','selected_R_seed42_01.jpg','undetermined_01.jpg']:\n    path=REPORT/'figures'/name\n    if path.exists(): display(Image(filename=str(path)))")
    cells[-1].metadata['tags']=['local-photo-preview']
    md('Фотографии отображаются при локальном Run All; байты фото не сохраняются внутри notebook. Полные контактные листы и их индексы находятся в `figures/`.')
    md('''---
## ⚙️ Рецепт и четыре пилота

ImageNetV1, RGB224, FP32; все слои и BN обучаются. AdamW1e-4, wd1e-4,
batch32,20 эпох. Class-balanced sampler, CE; только цветовые аугментации.
Максимум validation Macro-F1; при равенстве ранняя эпоха.
После четырёх seed42 пилотов проверяем development-сходимость и ресурсы.
Один и тот же manifest режима используется Small и Large.''')
    code("show_json(DATA / 'final_config.json')\nshow_json(REPORT / 'environment.json')")
    for seed in [42,43,44]:
        md(f'## 🧪 Seed {seed}'+(' — пилот, входит в 12 при неизменном рецепте' if seed==42 else ' — основной план'))
        for key in ['small_O','large_O','small_R','large_R']:
            md(f'### {key.replace("_", " · ")} · seed {seed}')
            code(f'''if TRAIN:
    from src.viewpoint_training import fit
    fit(ROOT, '{key.split('_')[0]}', '{key.split('_')[1]}', {seed})
run_display(ROOT, '{key}', {seed})''')
        if seed==42:code("show_csv(REPORT / 'pilot_review.csv')\nshow_json(REPORT / 'pilot_decision.json')")
    md('''---
## 📊 Общая оценка после всех 12 обучений

Все модели оцениваются на одном внутреннем test. Ниже — фактические результаты из CSV.
Полный mapping сохраняется при расчёте метрик срезов. SD: ddof=1 по трём seeds.''')
    code("show_csv(REPORT / 'test_metrics.csv', ['config','seed','best_epoch','macro_f1','accuracy','images'])\nshow_csv(REPORT / 'test_summary.csv')")
    code("mapping=json.loads((DATA/'class_mapping.json').read_text(encoding='utf-8'))\ndisplay(Markdown(table([{'class_index':index,'poi_id':poi} for poi,index in mapping.items()])))\ndisplay(Image(filename=str(REPORT/'figures/confusion_matrices.png')))\nshow_csv(REPORT / 'per_poi_metrics.csv', ['config','seed','poi_id','support','f1','recall'])")
    code("show_csv(REPORT / 'viewpoint_slices.csv', ['config','seed','viewpoint','images','macro_f1','accuracy','mapping_classes'])")
    md('''### Статистическая методика

Предусмотрены 5000 парных кластерных повторов, seed2026, двусторонние 95% интервалы.
Общие outings учитываются совместно, сеансы — внутри фиксированного POI.
При неизвестной истории сеансов схема может быть недопустима: интервалы тогда
не рассчитываются и не заменяются bootstrap отдельных кадров. Test исторический;
статус исследования — поисковый внутренний.''')
    code("result=json.loads((REPORT/'inference.json').read_text(encoding='utf-8'))\npoint=result['point']\ndisplay(Markdown(table([{'metric':key,'value_pp':100*point[key]} for key in ['G_O','G_R','C','small_gain','large_gain']])))\nshow_csv(REPORT/'per_seed_contrasts.csv')\ndisplay(Markdown(table([{'bootstrap_requested':result['requested_replicates'],'performed':result['performed_replicates'],'C_ci95':result['C_ci95'],'G_R_ci95':result['G_R_ci95'],'decision':result['conclusion']}])))\nshow_json(DATA/'bootstrap_plan_before_predictions.json')\nshow_json(REPORT/'inference.json')")
    md('''### 💻 Ресурсы

Заранее выбранная пара Р/42: Windows CPU FP32,20 прогревов,100 замеров×3.
Это только forward; не Android, не экспорт и не полный путь обработки JPEG.''')
    code("show_csv(REPORT/'resource_benchmark.csv')")
    md('## 🧭 Итог по гипотезе')
    code("text=(REPORT/'research_summary.md').read_text(encoding='utf-8')\ndisplay(Markdown(text.split('## Вывод по гипотезе')[1].split('## Практический выбор')[0]))\nshow_json(REPORT/'experiment_status.json')")
    md('''## 🔁 Как воспроизвести

Из корня проекта: `python scripts/run_viewpoint_research.py pilot`, затем после
зафиксированного development-решения `main`, затем `evaluate`.
Повторное обучение после открытия test запрещено для этого run_id. Для нового рецепта
нужен новый run_id и новый снимок данных. По умолчанию ноутбук читает готовые артефакты.
Подробный отчёт: `reports/viewpoint_20261102_v1/research_summary.md`.
Отказ на чужих фото и мобильная готовность этим экспериментом не доказываются.''')
    md('''### 📎 Исходные таблицы

- [Manifests О/Р и общие val/test](../data/annotations/viewpoint_20261102_v1/manifests/)
- [Разметка и решения об отборе](../data/annotations/viewpoint_20261102_v1/selection_decisions.csv)
- [Карточки ракурсов](../data/annotations/viewpoint_20261102_v1/orientation_cards.json)
- [Все 12 результатов](../reports/viewpoint_20261102_v1/test_metrics.csv)
- [Предсказания по каждому кадру](../reports/viewpoint_20261102_v1/test_predictions.csv)
- [Краткий отчёт](../reports/viewpoint_20261102_v1/research_summary.md)
''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    path=root/'notebooks/10_mobilenetv3_viewpoint_research.ipynb';nbf.write(notebook,path)
    return path
