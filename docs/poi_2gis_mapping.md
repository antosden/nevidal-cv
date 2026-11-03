# Связь CV POI и объектов 2GIS

Метки обученной модели и индексы 0–22 остаются прежними. Изменять class_mapping.json
или переобучать модель из-за внешних ID не требуется. Единственный источник связей:
`config/poi_2gis_mapping.json`, перенесённый из страницы контракта 30 октября 2026 года.

Состояние: 23 класса, 22 связи с ID 2GIS, SDK-проверка всех связей ещё ожидается.
ID хранятся строками: многие из них больше точного целочисленного диапазона JavaScript.
Кавычки JSON не входят в сам ID. Числа, пробелы, кавычки внутри значения запрещены.

Арка главного входа: `class_index=7`, `cv_poi_id=moscow/ostankinskiy/main_entrance`.
В исходной странице кандидат `4504127908473840` отнесён ко всему парку ВДНХ.
Он сохранён в candidate_dgis_poi_id, но активный dgis_poi_id=null,
binding_status=scope_mismatch. Такой ID не разрешается в класс арки даже в тестовом режиме.
Если маршрут уже хранит правильный CV ID арки, распознавание класса от этого не блокируется.

## Python

```python
from src.poi_catalog import PoiCatalog

catalog = PoiCatalog.from_files(
    'config/poi_2gis_mapping.json',
    'models/mobilenetv3_small_classifier/class_mapping.json',
)
# Выход модели всегда декодируется по неизменному индексу.
binding = catalog.decode_class_index(14)
assert binding['cv_poi_id'] == 'moscow/ostankinskiy/friendship_of_the_peoples'

# Только для интеграционного теста с предварительными ID:
target_poi_id = catalog.target_for_dgis(
    '70030076661708120', allow_unverified=True,
)
```

По умолчанию resolve_dgis/target_for_dgis/dgis_for_cv не допускают предварительные
связи: ошибка dgis_mapping_unverified. После реальной SDK-проверки разработчик
меняет sdk_validation_status на verified в соответствующей записи. Флаг
allow_unverified=True не обходит scope_mismatch и не подтверждает фотографию.
Существующий HTTP-сервис автоматически к этому адаптеру не подключён.

## Flutter

`integration/flutter/poi_catalog.dart` — независимый Dart-адаптер без ML-зависимостей.
Flutter-проект находится вне этого репозитория, поэтому код ещё нужно перенести
в приложение и включить два JSON в pubspec.yaml как assets:

```yaml
flutter:
  assets:
    - assets/cv/poi_2gis_mapping.json
    - assets/cv/class_mapping.json
```

Первый файл берётся из config/, второй — из выпуска текущей модели.
Создание каталога (нужны импорты dart:convert, flutter/services.dart и poi_catalog.dart):

```dart
final modelMap = Map<String, int>.from(jsonDecode(
  await rootBundle.loadString('assets/cv/class_mapping.json'),
) as Map);
final catalog = PoiCatalog.fromJson(
  await rootBundle.loadString('assets/cv/poi_2gis_mapping.json'), modelMap,
);
// Только предварительный интеграционный тест. В production флаг не включать.
final targetPoiId = catalog.targetForDgis(
  '70030076661708120', allowUnverified: true,
);
final predictedPoiId = catalog.bindingForClassIndex(14).cvPoiId;
```

В API 1.1 target_poi_id остаётся CV-строкой, например
moscow/ostankinskiy/friendship_of_the_peoples. Поле poi_id — контекст точки маршрута,
а не ID 2GIS. Число class_index не отправляется вместо target_poi_id.
Соответствие POI карты классу само по себе не означает, что фотография принята:
решение об отказе и score обрабатываются отдельно. Flutter/Dart SDK в этой среде
не установлен; адаптер требует dart analyze и интеграционного теста в приложении.

## Проверки

```powershell
.\.venv\Scripts\python.exe scripts/check_poi_2gis_mapping.py
.\.venv\Scripts\python.exe -m unittest tests.test_poi_catalog -v
# Строгая проверка готовности SDK-связей: пока ожидается ненулевой exit code.
.\.venv\Scripts\python.exe scripts/check_poi_2gis_mapping.py --require-sdk-verified
```

Проверки локальные: целостность классов, точность строковых ID, отсутствие неоднозначных
ID и двустороннее преобразование 22 предоставленных связей. Ответы 2GIS SDK ими не проверяются.
В источнике также отмечены вопросы фасадов павильона №71 и особняка Морозова:
они сохранены в review_notes и требуют сверки разработчиком с реальными объектами.
