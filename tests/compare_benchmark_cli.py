"""Exercise the real CPU model CLI; never claim device or performance qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
binary = args.binary.resolve(strict=True)
model = args.model_dir.resolve(strict=True)
out = args.out.resolve() / ("run-" + str(time.time_ns()))
out.mkdir(parents=True, exist_ok=False)
records = []
def run(name, argv, expected, timeout=300):
    command = [str(binary), *map(str, argv)]
    result = subprocess.run(command, cwd=root, capture_output=True, timeout=timeout)
    (out / (name + ".stdout")).write_bytes(result.stdout)
    (out / (name + ".stderr")).write_bytes(result.stderr)
    records.append({"name": name, "argv": command, "exit_code": result.returncode, "expected": expected,
                    "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                    "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()})
    (out / "commands.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    if result.returncode != expected:
        raise AssertionError((name, result.returncode, expected, result.stderr.decode(errors="replace")))
    return result
def spec(name, value):
    path = out / (name + ".json")
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path

w = h = 32
header = f"P6\n{w} {h}\n255\n".encode()
for name, shift in (("left", 0), ("right", 3)):
    pixels = bytes(v for y in range(h) for x in range(w)
                   for v in ((x + shift) * 7 % 256, y * 7 % 256, ((x + shift) ^ y) * 5 % 256))
    (out / (name + ".ppm")).write_bytes(header + pixels)
base = {"schema_version": 1, "model_directory": str(model), "left": str(out / "left.ppm"),
        "right": str(out / "right.ppm"), "alpha": "0.5", "precision_profile": "CPU_FP32_REFERENCE"}
for label in ("baseline", "candidate"):
    request = dict(base, output=str(out / (label + ".ppm")))
    run(label, ["interpolate", "--run-spec", spec(label + "-request", request),
                "--report", out / (label + "-report.json")], 0)
request = {"schema_version": 1, "baseline_report": str(out / "baseline-report.json"),
           "baseline_output": str(out / "baseline.ppm"), "candidate_report": str(out / "candidate-report.json"),
           "candidate_output": str(out / "candidate.ppm")}
run("compare_exact", ["compare", "--run-spec", spec("compare-request", request), "--report", out / "compare-report.json"], 0)
comparison = json.loads((out / "compare-report.json").read_text())
assert comparison["pixel_bytes_equal"] and comparison["quality_qualification"] == "INCONCLUSIVE"
altered = bytearray((out / "candidate.ppm").read_bytes()); altered[-1] ^= 1
(out / "altered.ppm").write_bytes(altered)
run("compare_tampered_output", ["compare", "--run-spec", spec("compare-bad", dict(request, candidate_output=str(out / "altered.ppm")))], 4)
candidate = json.loads((out / "candidate-report.json").read_text());candidate["requested_guard_flags"] += 1
spec("changed-report", candidate)
run("compare_guard_mismatch", ["compare", "--run-spec", spec("compare-guard", dict(request, candidate_report=str(out / "changed-report.json")))], 4)
candidate = json.loads((out / "candidate-report.json").read_text());candidate.pop("alpha")
spec("missing-alpha-report", candidate)
run("compare_missing_alpha", ["compare", "--run-spec", spec("compare-missing-alpha", dict(request, candidate_report=str(out / "missing-alpha-report.json")))], 4)
candidate = json.loads((out / "candidate-report.json").read_text());candidate["frame_kind"] = 2
spec("contradictory-frame-report", candidate)
run("compare_contradictory_frame", ["compare", "--run-spec", spec("compare-contradictory", dict(request, candidate_report=str(out / "contradictory-frame-report.json")))], 4)
candidate = json.loads((out / "candidate-report.json").read_text())
candidate["output_sha256"] = hashlib.sha256(altered).hexdigest()
candidate["pixel_sha256"] = hashlib.sha256(altered[len(header):]).hexdigest()
spec("changed-output-report", candidate)
run("compare_different", ["compare", "--run-spec", spec("compare-different", dict(request,
    candidate_report=str(out / "changed-output-report.json"), candidate_output=str(out / "altered.ppm"))),
    "--report", out / "compare-different-report.json"], 0)
different = json.loads((out / "compare-different-report.json").read_text())
assert different["different_samples"] == 1 and not different["pixel_bytes_equal"] and different["quality_qualification"] == "INCONCLUSIVE"
for name, extra in (("endpoint", {"alpha":"0", "warmup_seconds":30, "run_seconds":120}),
                    ("short", {"warmup_seconds":1, "run_seconds":120})):
    run("benchmark_reject_" + name, ["benchmark", "--run-spec", spec("invalid-"+name,
        dict(base, raw_events=str(out / (name + ".jsonl")), **extra)), "--report", out / (name + "-report.json")], 4)
collision = out / "protected-existing-events.jsonl"; collision.write_bytes(b"preserve-existing-raw\n")
run("benchmark_path_collision", ["benchmark", "--run-spec", spec("collision-request",
    dict(base, warmup_seconds=30, run_seconds=120, raw_events=str(collision))), "--report", collision], 9)
assert collision.read_bytes() == b"preserve-existing-raw\n"
invalid_alpha = dict(base, alpha="not-a-number", warmup_seconds=30, run_seconds=120,
                     raw_events=str(out / "invalid-alpha-events.jsonl"))
run("benchmark_exception_report", ["benchmark", "--run-spec", spec("invalid-alpha", invalid_alpha),
    "--report", out / "invalid-alpha-report.json"], 12)
assert json.loads((out / "invalid-alpha-report.json").read_text())["result"] == "FAIL"
benchmark = dict(base, warmup_seconds=30, run_seconds=120, raw_events=str(out / "events.jsonl"))
run("benchmark_actual", ["benchmark", "--run-spec", spec("benchmark-request", benchmark),
                         "--report", out / "benchmark-report.json"], 0, timeout=360)
result = json.loads((out / "benchmark-report.json").read_text())
raw = (out / "events.jsonl").read_bytes()
assert result["result"] == "OBSERVED" and result["performance_qualification"] == "INCONCLUSIVE"
assert hashlib.sha256(raw).hexdigest() == result["raw_events_sha256"]
events = [json.loads(line) for line in raw.splitlines()]
for phase, seconds, count in (("warmup",30,result["warmup_successful_inferred_jobs"]),
                              ("measure",120,result["successful_inferred_jobs"])):
    relevant = [e for e in events if e["phase"] == phase]
    start, end = relevant[0], relevant[-1]
    jobs = [e for e in relevant if e["kind"] == "job"]
    assert start["kind"] == "run_start" and end["kind"] == "run_end" and end["timestamp_ns"] - start["timestamp_ns"] >= seconds * 1_000_000_000
    assert len(jobs) == count and count > 0
    assert all(e["accepted"] and e["status"] == 0 and e["algorithm_accepted"] and e["frame_kind"] == 1 and e["submit_to_drain_ns"] > 0 for e in jobs)
summary = {"result":"PASS", "scope":"real fixed model Windows CPU CLI only; no GPU/paired/quality/performance qualification",
           "binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(), "checks":len(records),
           "warmup_jobs":result["warmup_successful_inferred_jobs"], "measured_jobs":result["successful_inferred_jobs"]}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"result":"PASS", "out":str(out), "checks":len(records)}))
