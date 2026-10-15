import json

from eval.eval import evaluate


def test_eval_scores_against_truth(tmp_path):
    (tmp_path / "books.csv").write_text(
        "title,author,legible,height_cm,thickness_cm,price_new,price_used,price_source\n"
        "Dune,Frank Herbert,yes,17.8,3.4,10,5,x\n"
        "Emma,Jane Austen,yes,19.8,2.0,8,4,x\n"
        "(unreadable),,no,21,1.2,,,\n"
    )
    (tmp_path / "items.csv").write_text("name,category\nlamp,lamp\nrug,rug\n")
    (tmp_path / "room.json").write_text(json.dumps({"floor_area_m2": 13.0, "wall_area_m2": 36.0}))
    rc = lambda a: {"amount": a}
    packet = {
        "totals": {"book_count": 3},
        "books": [
            {"id": "B1", "status": "identified", "title": "Dune", "id_confidence": 0.9, "spine_height_cm": 18.0, "spine_thickness_cm": 3.3, "replacement_cost": rc(11)},
            {"id": "B2", "status": "identified", "title": "Persuasion", "id_confidence": 0.8, "spine_height_cm": 19.0, "spine_thickness_cm": 2.0, "replacement_cost": rc(8)},
            {"id": "B3", "status": "unidentified", "title": "", "id_confidence": 0, "spine_height_cm": 21, "spine_thickness_cm": 1.2, "replacement_cost": rc(None)},
        ],
        "items": [{"category": "lamp"}, {"category": "rug"}],
        "room": {"floor_area_m2": 13.5, "wall_area_m2": 30.0},
        "pipeline_metrics": {"wall_clock_s": 120},
    }
    rows = {r["measure"]: r for r in evaluate(packet, tmp_path)}
    assert rows["Book count"]["pass"]
    assert not rows["Title identification"]["pass"]  # Persuasion is a confident wrong answer (1 of 2)
    assert "50%" in rows["Title identification"]["result"]
    assert rows["Non-book items"]["pass"] and rows["Floor area"]["pass"] and not rows["Wall area"]["pass"]
    assert rows["Time to packet"]["pass"]
