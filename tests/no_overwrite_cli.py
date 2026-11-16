"""Verify exclusive output creation and input aliases without model execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
binary = args.binary.resolve(strict=True)
out = args.out.resolve() / ("run-" + str(time.time_ns()))
out.mkdir(parents=True, exist_ok=False)
records = []
def run(name, argv, expected):
    command = [str(binary), *map(str, argv)]
    p = subprocess.run(command, capture_output=True, timeout=30)
    (out / (name + ".stdout")).write_bytes(p.stdout)
    (out / (name + ".stderr")).write_bytes(p.stderr)
    records.append({"name":name, "argv":command, "exit_code":p.returncode, "expected":expected,
                    "stdout_sha256":hashlib.sha256(p.stdout).hexdigest(), "stderr_sha256":hashlib.sha256(p.stderr).hexdigest()})
    (out / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
    assert p.returncode == expected, (name, p.returncode, expected, p.stderr)
    if p.returncode: assert json.loads(p.stderr)["result"] == "FAIL"
    return p
def save_spec(name, data):
    path = out / (name + ".json"); path.write_text(json.dumps(data), encoding="utf-8"); return path

existing = out / "existing.bin"; existing.write_bytes(b"KEEP EXISTING\n")
run("existing_report", ["--version-json", "--report", existing], 9)
assert existing.read_bytes() == b"KEEP EXISTING\n"
link = out / "hardlink.bin"; os.link(existing, link)
run("hardlink_report", ["--version-json", "--report", link], 9)
assert existing.read_bytes() == link.read_bytes() == b"KEEP EXISTING\n"
run("new_report", ["--version-json", "--report", out / "new-report.json"], 0)
assert json.loads((out / "new-report.json").read_text())["version"]

left = out / "left.ppm"; right = out / "right.ppm"
left.write_bytes(b"P6\n1 1\n255\n\x01\x02\x03"); right.write_bytes(b"P6\n1 1\n255\n\x04\x05\x06")
model = out / "model";model.mkdir();(model / "flownet.param").write_bytes(b"FAKE GRAPH\n");(model / "flownet.bin").write_bytes(b"FAKE WEIGHTS\n")
base = {"schema_version":1,"model_directory":str(model),"left":str(left),"right":str(right),
        "alpha":"0.5","precision_profile":"CPU_FP32_REFERENCE"}
for name,target in (("left",left),("right",right),("model",model/"flownet.bin"),("hardlink",link)):
    before=target.read_bytes(); report=out/("failure-"+name+".json")
    run("output_"+name,["interpolate","--run-spec",save_spec("spec-"+name,dict(base,output=str(target))),"--report",report],9)
    assert target.read_bytes()==before and json.loads(report.read_text())["result"]=="FAIL"

target=out/"shared-new.out"; request=save_spec("same-target",dict(base,output=str(target)))
run("report_equals_output",["interpolate","--run-spec",request,"--report",target],4)
assert not target.exists()
new=out/"case-new.out"; casepath=out/"CASE-NEW.OUT"
if os.name == "nt":
    run("case_insensitive_alias",["interpolate","--run-spec",save_spec("case-alias",dict(base,output=str(new))),"--report",casepath],4)
    assert not new.exists() and not casepath.exists()
specfile=save_spec("protected-spec",dict(base,output=str(out/"never-output.ppm")))
before=specfile.read_bytes(); run("report_equals_spec",["interpolate","--run-spec",specfile,"--report",specfile],9); assert specfile.read_bytes()==before
same_spec=out/"self-output.json";same_spec.write_text(json.dumps(dict(base,output=str(same_spec))),encoding="utf-8")
before=same_spec.read_bytes();run("output_equals_spec",["interpolate","--run-spec",same_spec,"--report",out/"self-output-failure.json"],9);assert same_spec.read_bytes()==before
missing_input=out/"missing-input.ppm"
run("report_equals_missing_input",["interpolate","--run-spec",save_spec("missing-input-spec",dict(base,left=str(missing_input),output=str(out/"unused.ppm"))),"--report",missing_input],4)
assert not missing_input.exists()
try:
    symbolic=out/"dangling-symlink";symbolic.symlink_to(out/"missing-target")
except OSError as error:
    symlink_scope="SKIPPED:"+str(error)
else:
    run("dangling_symlink_report",["--version-json","--report",symbolic],9)
    assert symbolic.is_symlink() and not (out/"missing-target").exists()
    symlink_scope="PASS"

junction_scope="NOT_APPLICABLE"
if os.name == "nt":
    # A junction needs no symlink privilege and is created only inside this
    # run's unique evidence directory. Preserve it for inspection.
    real=out/"junction-target";real.mkdir();junction=out/"junction-alias"
    script=out/"create-junction.ps1"
    script.write_text('param([string]$Link, [string]$Target)\n$ErrorActionPreference="Stop"\nNew-Item -ItemType Junction -Path $Link -Target $Target | Out-Null\n',encoding="utf-8")
    proc=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-File",str(script),str(junction),str(real)],capture_output=True,timeout=30)
    (out/"junction-create.stdout").write_bytes(proc.stdout);(out/"junction-create.stderr").write_bytes(proc.stderr)
    if proc.returncode:
        junction_scope="UNAVAILABLE:"+str(proc.returncode)
    else:
        junction_scope="PASS"
        same=real/"new-output.ppm"
        run("junction_alias",["interpolate","--run-spec",save_spec("junction-spec",dict(base,output=str(same))),
                              "--report",junction/"new-output.ppm"],4)
        assert not same.exists()

race=out/"race-report.json";argv=[str(binary),"--version-json","--report",str(race)]
processes=[subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(2)]
results=[p.communicate(timeout=30) for p in processes]
assert sorted(p.returncode for p in processes)==[0,9], ([p.returncode for p in processes],results)
assert json.loads(race.read_text())["version"]
for index,(stdout,stderr) in enumerate(results):
    (out/f"race-{index}.stdout").write_bytes(stdout);(out/f"race-{index}.stderr").write_bytes(stderr)
records.append({"name":"concurrent_exclusive_create","argv":argv,"exit_codes":[p.returncode for p in processes]})
(out/"commands.json").write_text(json.dumps(records,indent=2)+"\n")
summary={"result":"PASS","scope":"Windows/host CLI file protection only; no model or device execution",
         "checks":len(records),"symlink":symlink_scope,"junction":junction_scope,
         "binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest()}
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({"result":"PASS","out":str(out),"checks":len(records),"symlink":symlink_scope,"junction":junction_scope}))
