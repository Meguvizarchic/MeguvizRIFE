"""Compile RIFE's raw GPU resource kernels with the pinned host ShaderTools."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

project = Path(__file__).resolve().parents[1]
tool = project.parent / "MeguvizShaderTools/build/host-windows-release/bin/mgst.exe"
out = project / "artifacts/implementation" / ("shaders-" + str(time.time_ns()))
out.mkdir(parents=True)
version = subprocess.run([tool, "--version-json"], capture_output=True, check=True)
(out / "tool-version.json").write_bytes(version.stdout)
lock = json.loads(version.stdout)["toolchain_lock_hash"]
records = []
sources = [(source, source.stem, source.read_bytes(), {}) for source in sorted((project / "src/backends/ncnn/legacy").glob("frame_gpu_*.comp"))]
add = project / "src/backends/ncnn/legacy/frame_add_leaky.comp"
for pack in [0, 1]:
    for precision, options in [("fp32", {}), ("fp16", {"use_fp16_storage": True, "use_fp16_arithmetic": True, "use_fp16_packed": True}),
                              ("packed", {"use_fp16_storage": False, "use_fp16_arithmetic": False, "use_fp16_packed": True})]:
        payload = add.read_text().replace("#version 450", "#version 450\n#define MGR_PACK4 " + str(pack)).encode()
        sources.append((add, add.stem + "-pack" + str(4 if pack else 1) + "-" + precision, payload, options))
for source, name, payload, options in sources:
    case = out / name
    case.mkdir()
    (case / "shader.comp").write_bytes(payload)
    request = {"schema_version": 1, "logical_shader_id": "rife/" + source.stem,
        "owner": {"repository": "MeguvizRIFE", "revision": "working-tree"},
        "source": {"path": "shader.comp", "sha256": hashlib.sha256(payload).hexdigest()},
        "includes": [], "profile_id": "ncnn-compat", "profile_version": "20261020",
        "entry_point": "main", "target_env": "vulkan1.0", "effective_options": options,
        "numeric_profile": "rgb8-exact-integer-summary", "toolchain_lock_hash": lock,
        "required_flags": [], "optimization_passes": [], "resource_recipe": "ncnn-20261020",
        "ncnn_revision": "7e36d25cb153092ba87a60530455c7376fedf452",
        "target_facts": {"provenance": "synthetic", "maintenance4_enabled": False,
            "support_fp16_storage": bool(options.get("use_fp16_storage")), "shader_int64": False, "shader_int16": False,
            "moltenvk": False, "device_defines": []}}
    (case / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    p = subprocess.run([tool, "compile", "--request", case / "request.json", "--out-dir", case / "compiled"], capture_output=True)
    (case / "stdout.json").write_bytes(p.stdout)
    (case / "stderr.log").write_bytes(p.stderr)
    records.append({"shader": name, "sha256": hashlib.sha256(payload).hexdigest(), "exit_code": p.returncode})
    print(name, p.returncode, p.stdout.decode(errors="replace")[-1800:] if p.returncode else "", flush=True)
(out / "summary.json").write_text(json.dumps({"scope": "host shader compilation/ABI only; no GPU numerical qualification", "records": records}, indent=2) + "\n")
print(out, flush=True)
raise SystemExit(any(item["exit_code"] for item in records))
