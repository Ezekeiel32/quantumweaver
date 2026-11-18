"""The recorded runs in training_jobs/ must be readable, and the numbers in the README must match them."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def load_jobs():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "training_jobs").glob("*.json"))]


def test_every_record_parses_and_has_the_core_fields():
    jobs = load_jobs()
    assert jobs
    for job in jobs:
        assert {"job_id", "status", "parameters"} <= set(job)


def test_accuracies_are_percentages():
    for job in load_jobs():
        accuracy = job.get("accuracy")
        if isinstance(accuracy, (int, float)):
            assert 0 <= accuracy <= 100, job["job_id"]


def test_readme_run_count_matches_the_records():
    claimed = int(re.search(r"(\d+) recorded training runs", README).group(1))
    assert claimed == len(load_jobs())


def test_readme_best_accuracy_matches_the_best_completed_run():
    completed = [j for j in load_jobs() if j["status"] == "completed" and isinstance(j.get("accuracy"), (int, float))]
    best = max(completed, key=lambda j: j["accuracy"])
    claimed = float(re.search(r"Best run: \*\*(\d+\.\d+)%\*\*", README).group(1))
    assert claimed == best["accuracy"]
    assert best["job_id"] in README
