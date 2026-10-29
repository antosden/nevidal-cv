"""Связь ID карты с фиксированными выходами модели, без изменения весов."""
from __future__ import annotations

import json
from pathlib import Path
import re


class PoiMappingError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class PoiCatalog:
    def __init__(self, data, model_mapping):
        if data.get("schema_version") != 1 or not isinstance(data.get("mapping_version"), str):
            raise ValueError("Неподдерживаемый формат каталога")
        if not model_mapping or any(type(v) is not int for v in model_mapping.values()):
            raise ValueError("Некорректный mapping модели")
        if sorted(model_mapping.values()) != list(range(len(model_mapping))):
            raise ValueError("Индексы модели должны быть непрерывными и уникальными")
        self.version = data["mapping_version"]
        self._cv, self._index, self._dgis = {}, {}, {}
        for source in data["entries"]:
            row = dict(source)
            cv, index = row["cv_poi_id"], row["class_index"]
            if type(index) is not int or model_mapping.get(cv) != index:
                raise ValueError(f"Несовпадение класса модели: {cv}")
            if cv in self._cv or index in self._index:
                raise ValueError(f"Повтор CV-класса: {cv}")
            if row["binding_status"] not in {"mapped", "scope_mismatch", "unmapped"}:
                raise ValueError("Неизвестный binding_status")
            if row["sdk_validation_status"] not in {"pending", "verified"}:
                raise ValueError("Неизвестный sdk_validation_status")
            active, candidate = row.get("dgis_poi_id"), row.get("candidate_dgis_poi_id")
            for external in (active, candidate):
                if external is not None and (not isinstance(external, str) or not re.fullmatch(r"[0-9]+", external)):
                    raise ValueError("ID 2GIS должен быть строкой цифр без кавычек внутри")
            if (row["binding_status"] == "mapped") != (active is not None):
                raise ValueError("Активный ID допустим только для mapped")
            for external in {active, candidate} - {None}:
                if external in self._dgis:
                    raise ValueError(f"Неоднозначный ID 2GIS: {external}")
                self._dgis[external] = row
            self._cv[cv], self._index[index] = row, row
        if set(self._cv) != set(model_mapping):
            raise ValueError("Каталог не покрывает все выходы модели")

    @classmethod
    def from_files(cls, catalog_path, model_mapping_path):
        data = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        model = json.loads(Path(model_mapping_path).read_text(encoding="utf-8"))
        return cls(data, model)

    def decode_class_index(self, index):
        """Идентификация класса доступна даже без связи с картой; это не CV-успех."""
        if type(index) is not int or index not in self._index:
            raise PoiMappingError("unknown_class_index", f"Неизвестный class_index: {index}")
        return dict(self._index[index])

    def resolve_cv(self, cv_poi_id):
        if not isinstance(cv_poi_id, str) or cv_poi_id not in self._cv:
            raise PoiMappingError("unknown_cv_poi", f"Неизвестный CV POI: {cv_poi_id}")
        return dict(self._cv[cv_poi_id])

    def resolve_dgis(self, dgis_poi_id, *, allow_unverified=False):
        # Число из JS уже могло потерять точность; str(number) это не исправляет.
        if not isinstance(dgis_poi_id, str) or not re.fullmatch(r"[0-9]+", dgis_poi_id):
            raise PoiMappingError("invalid_dgis_id", "Передайте ID 2GIS строкой цифр")
        row = self._dgis.get(dgis_poi_id)
        if row is None:
            raise PoiMappingError("dgis_mapping_missing", "Для объекта карты нет CV-соответствия")
        self._require_binding(row, allow_unverified)
        return dict(row)

    @staticmethod
    def _require_binding(row, allow_unverified):
        if row["binding_status"] != "mapped":
            raise PoiMappingError("dgis_mapping_blocked", "ID относится к другому масштабу объекта; автоматическая связь запрещена")
        if row["sdk_validation_status"] != "verified" and not allow_unverified:
            raise PoiMappingError("dgis_mapping_unverified", "Соответствие ещё не проверено в SDK")

    def target_for_dgis(self, dgis_poi_id, *, allow_unverified=False):
        return self.resolve_dgis(dgis_poi_id, allow_unverified=allow_unverified)["cv_poi_id"]

    def dgis_for_cv(self, cv_poi_id, *, allow_unverified=False):
        row = self.resolve_cv(cv_poi_id)
        self._require_binding(row, allow_unverified)
        return row["dgis_poi_id"]
