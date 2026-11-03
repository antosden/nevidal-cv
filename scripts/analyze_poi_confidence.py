"""Диагностика POI по сохранённым предсказаниям без запуска encoder и обучения."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "models/encoder_comparison_manual_all/dinov2_small"
OUTPUT = ROOT / "reports/poi_confidence"
NAMES = {
    "arseny_morozov_house": "Особняк Арсения Морозова",
    "pashkov_house": "Дом Пашкова",
    "khrushchev_seleznev_house": "Усадьба Хрущёвых — Селезнёвых",
    "pushkin_museum": "ГМИИ им. Пушкина",
    "armenia": "Павильон «Армения»",
    "central_pavilion": "Центральный павильон",
    "cosmos": "Павильон «Космос»",
    "main_entrance": "Арка главного входа",
    "moscow_sun": "«Солнце Москвы»",
    "ostankino_tower": "Останкинская телебашня",
    "bolshoi_theatre": "Большой театр",
    "hotel_metropol": "Гостиница «Метрополь»",
    "hotel_national": "Гостиница «Националь»",
    "maly_theatre": "Малый театр",
    "friendship_of_the_peoples": "Фонтан «Дружба народов»",
    "kazakhstan_pavilion": "Павильон «Казахстан»",
    "moldova_pavilion": "Павильон «Молдова»",
    "moskvarium": "Москвариум",
    "pavilion_5": "Павильон №5",
    "pavilion_71": "Павильон №71",
    "polymer_house": "Дом полимеров",
    "robostation": "Робостанция",
    "stone_flower": "Фонтан «Каменный цветок»",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def yes(value):
    return str(value).lower() == "true"


def wilson(k, n):
    # Интервал описывает только биномиальную неопределённость; корреляцию серий не устраняет.
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (k/n + z*z/(2*n))/denominator
    radius = z*math.sqrt(k/n*(1-k/n)/n + z*z/(4*n*n))/denominator
    return max(0, center-radius), min(1, center+radius)


def run():
    OUTPUT.mkdir(exist_ok=True, parents=True)
    service = ROOT / "models/poi_service_dinov2"
    metadata = json.loads((EXPERIMENT / "metadata.json").read_text(encoding="utf-8"))
    bundle = json.loads((service / "metadata.json").read_text(encoding="utf-8"))
    policy = json.loads((service / "rejection_policy.json").read_text(encoding="utf-8"))
    assert bundle["encoder"] == metadata["encoder"] == "dinov2_small"
    assert bundle["artifact_sha256"]["weights.pth"] == metadata["weights_sha256"]
    assert json.loads((EXPERIMENT / "selection.json").read_text())["variant"] == "random_20"
    gallery = read_csv(service / "gallery.csv")
    assert {r["image_id"] for r in gallery} == {
        r["image_id"] for r in read_csv(EXPERIMENT / "random_20/gallery.csv")}
    counts = {}
    all_annotations = {}
    for split in ("train", "val", "test"):
        rows = read_csv(ROOT / f"data/annotations/{split}.csv")
        counts[split] = Counter(r["poi_id"] for r in rows)
        all_annotations.update({r["image_id"]: r for r in rows})
        saved_ids = {r["image_id"] for r in read_csv(EXPERIMENT / f"{split}_embedding_index.csv")}
        assert saved_ids == {r["image_id"] for r in rows}, f"Изменилась выборка {split}"
    queries = {}
    pairs = {}
    negatives = {}
    for split in ("val", "test"):
        pairs[split] = read_csv(EXPERIMENT / f"{split}_pairs.csv")
        own = [r.copy() for r in pairs[split] if yes(r["genuine"])]
        assert len(own) == len({r["image_id"] for r in own})
        for r in own:
            r["score"] = float(r["score"])
            r["margin"] = float(r["margin"])
            r["accepted"] = yes(r["accepted"])
            r["correct"] = r["true_poi_id"] == r["predicted_poi_id"]
            recomputed = (r["score"] >= policy["score_threshold"]
                          and r["margin"] >= policy["margin_threshold"] and r["margin"] > 0)
            assert recomputed == r["accepted"]
            r["split"] = split
            r["file_path"] = all_annotations[r["image_id"]]["file_path"]
        queries[split] = own
        negatives[split] = read_csv(EXPERIMENT / f"negative_{split}_predictions.csv")
        stored = read_csv(EXPERIMENT / f"{split}_per_poi.csv")
        for r in stored:
            own_poi = [q for q in own if q["true_poi_id"] == r["poi_id"]]
            assert sum(q["accepted"] for q in own_poi) == int(r["true_accepts"])
    result = []
    for poi in metadata["poi_ids"]:
        t = [r for r in queries["test"] if r["true_poi_id"] == poi]
        v = [r for r in queries["val"] if r["true_poi_id"] == poi]
        failures = [r for r in t if not r["accepted"]]
        confusions = Counter(r["predicted_poi_id"] for r in t if not r["correct"])
        incoming = [r for r in pairs["test"] if r["target_poi_id"] == poi
                    and not yes(r["genuine"]) and yes(r["accepted"])]
        external = [r for r in negatives["test"] if r["predicted_poi_id"] == poi
                    and yes(r["accepted_any_poi"])]
        accepted = sum(r["accepted"] for r in t)
        low, high = wilson(accepted, len(t))
        score_fails = sum(r["correct"] and r["score"] < policy["score_threshold"] for r in t)
        margin_fails = sum(r["correct"] and r["margin"] < policy["margin_threshold"] for r in t)
        failure_mode = ("мало тестовых примеров" if len(t) < 5 else
                        "есть ошибки выбора POI" if sum(r["correct"] for r in t)/len(t) < .8 else
                        "похож на конкурентов" if margin_fails > score_fails else
                        "низкое сходство с эталонами" if score_fails else "смешанная / редкие отказы")
        # A: много отказов или очень низкий TAR; C: хорошие результаты на обеих частях.
        priority = ("A" if len(t) >= 5 and (accepted/len(t) < .4 or len(failures) >= 8) else
                    "C" if len(t) >= 5 and accepted/len(t) >= .7
                    and sum(r["accepted"] for r in v)/len(v) >= .7 else "B")
        result.append({
            "poi_id": poi, "name": NAMES[poi.rsplit("/", 1)[-1]],
            "collection_priority": priority, "failure_mode": failure_mode,
            "train_images": counts["train"][poi], "gallery_images": sum(r["poi_id"] == poi for r in gallery),
            "val_images": len(v), "val_top1_correct": sum(r["correct"] for r in v),
            "val_true_accepts": sum(r["accepted"] for r in v),
            "test_images": len(t), "test_top1_correct": sum(r["correct"] for r in t),
            "test_top1_accuracy": sum(r["correct"] for r in t)/len(t),
            "test_true_accepts": accepted, "test_true_accept_rate": accepted/len(t),
            "test_tar_wilson95_low": low, "test_tar_wilson95_high": high,
            "test_false_rejects": len(failures),
            "correct_top1_below_score": score_fails, "correct_top1_below_margin": margin_fails,
            "correct_top1_below_both": sum(r["correct"] and r["score"] < policy["score_threshold"]
                                         and r["margin"] < policy["margin_threshold"] for r in t),
            "median_genuine_score": median(r["score"] for r in t),
            "median_genuine_margin": median(r["margin"] for r in t),
            "test_top1_confusions": json.dumps(confusions, ensure_ascii=False),
            "test_nearest_competitor_pois": json.dumps(Counter(
                max((p for p in pairs["test"] if p["image_id"] == r["image_id"]
                     and p["target_poi_id"] != poi), key=lambda p: float(p["score"]))["target_poi_id"]
                for r in t), ensure_ascii=False),
            "test_known_wrong_accepts_into_poi": len(incoming),
            "test_external_false_accepts_into_poi": len(external),
            "test_external_false_accept_sources": json.dumps(Counter(r["source_poi_id"] for r in external), ensure_ascii=False),
        })
    result.sort(key=lambda r: (r["collection_priority"], -r["test_false_rejects"], r["poi_id"]))
    write_csv(OUTPUT / "by_poi.csv", result)
    write_csv(OUTPUT / "query_diagnostics.csv", queries["test"] + queries["val"])
    wrong = [dict(r, split=split) for split in pairs for r in pairs[split]
             if not yes(r["genuine"]) and yes(r["accepted"])]
    if wrong:
        write_csv(OUTPUT / "wrong_accepts.csv", wrong)
    summary = {
        "model_version": bundle["model_version"], "encoder": bundle["encoder"],
        "score_threshold": policy["score_threshold"], "margin_threshold": policy["margin_threshold"],
        "test_images": len(queries["test"]),
        "test_correct_top1": sum(r["correct"] for r in queries["test"]),
        "test_true_accepts": sum(r["accepted"] for r in queries["test"]),
        "test_external_false_accepts": sum(yes(r["accepted_any_poi"]) for r in negatives["test"]),
        "negative_test_images": len(negatives["test"]), "by_poi": result,
        "method": "Диагностика сохранённых val/test; пороги и модель не изменены. Test уже использовался в исследовательских итерациях. Новые выводы проверять на новой съёмке. Приоритеты A/B/C — эвристика для планирования досъёмки, а не оценка достоверности POI.",
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "by_poi"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
