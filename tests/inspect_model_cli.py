"""Exercise inspect-model manifest validation without model weights or a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
binary = args.binary.resolve(strict=True)
out = args.out.resolve() / ("run-" + str(time.time_ns()))
out.mkdir(parents=True, exist_ok=False)
catalog = json.loads((root / "models/catalog/rife46.manifest.json").read_text(encoding="utf-8"))
inputs = {
    "formatted.json": json.dumps(catalog, indent=2).encode(),
    "mismatch.json": json.dumps({**catalog, "model_id": "other"}).encode(),
    "unknown.json": json.dumps({**catalog, "unexpected": True}).encode(),
    "duplicate.json": b'{"model_id":"one","model_id":"two"}',
    "malformed.json": b'{"model_id":',
    "oversize.json": b" " * 65537,
}
for name, data in inputs.items():
    (out / name).write_bytes(data)
records = []
def run(name, argv, expected):
    command = [str(binary), *map(str, argv)]
    result = subprocess.run(command, cwd=root, capture_output=True, timeout=30)
    stdout, stderr = out / (name + ".stdout"), out / (name + ".stderr")
    stdout.write_bytes(result.stdout); stderr.write_bytes(result.stderr)
    records.append({"name": name, "argv": command, "exit_code": result.returncode, "expected": expected,
                    "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                    "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()})
    (out / "commands.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    if result.returncode != expected:
        raise AssertionError((name, result.returncode, expected))
    return result

default = run("default", ["inspect-model"], 0)
explicit = run("explicit", ["inspect-model", "--manifest", root / "models/catalog/rife46.manifest.json"], 0)
formatted = run("formatted", ["inspect-model", "--manifest", out / "formatted.json"], 0)
assert json.loads(default.stdout) == catalog == json.loads(explicit.stdout) == json.loads(formatted.stdout)
for name in ("mismatch", "unknown", "duplicate", "malformed"):
    report = out / (name + "-report.json")
    run(name, ["inspect-model", "--manifest", out / (name + ".json"), "--report", report], 4)
    assert json.loads(report.read_text(encoding="utf-8"))["result"] == "FAIL"
run("oversize", ["inspect-model", "--manifest", out / "oversize.json"], 9)
run("missing", ["inspect-model", "--manifest", out / "missing.json"], 9)
run("duplicate_arg", ["inspect-model", "--manifest", out / "formatted.json", "--manifest", out / "formatted.json"], 2)
run("wrong_command", ["--version-json", "--manifest", out / "formatted.json"], 2)
summary = {"result": "PASS", "scope": "host CLI metadata contract only; model bytes and GPU not checked",
           "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "checks": len(records)}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"result": "PASS", "out": str(out), "checks": len(records)}))
