"""Compile-only check against the current Compute source SDK; no device run."""
import json
from pathlib import Path
import subprocess
import time

project = Path(__file__).resolve().parents[1]
root = project.parent
sdk = Path(r"C:\Users\Administrator\AppData\Local\Android\Sdk")
ndk = sdk / "ndk/30.0.16248370"
cmake = sdk / "cmake/4.1.2/bin/cmake.exe"
build = project / "build/android-arm64-api27"
out = project / "artifacts/implementation" / ("compile-" + str(time.time_ns()))
out.mkdir(parents=True)
records = []


def run(name, args):
    result = subprocess.run(list(map(str, args)), cwd=project, capture_output=True)
    (out / (name + ".stdout.log")).write_bytes(result.stdout)
    (out / (name + ".stderr.log")).write_bytes(result.stderr)
    records.append({"name": name, "argv": list(map(str, args)), "exit_code": result.returncode})
    (out / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
    print(name, result.returncode, flush=True)
    if result.returncode:
        print(result.stderr.decode(errors="replace")[-12000:])
        raise SystemExit(result.returncode)


legacy = project / "src/backends/ncnn/legacy"
for source in legacy.rglob("*.comp"):
    run("embed-" + source.stem, [cmake, "-DSOURCE=" + str(source),
        "-DOUTPUT=" + str(build / "generated" / (source.name + ".hex.h")),
        "-P", legacy / "embed_shader.cmake"])
compiler = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
includes = [project / "include", project / "src", build / "generated",
    root / "MeguvizCompute/include", root / "MeguvizCompute/private",
    root / "MeguvizCompute/build/install/android-arm64-api27/include/ncnn"]
for source in [project / "src/core/rife.cpp", project / "src/cli/main.cpp", legacy / "frame_engine.cpp", legacy / "add_leaky.cpp"]:
    run("syntax-" + source.stem, [compiler, "--target=aarch64-none-linux-android27", "--sysroot=" + str(ndk / "toolchains/llvm/prebuilt/windows-x86_64/sysroot"),
        "-std=c++17", "-fsyntax-only", "-fexceptions", "-fno-rtti", "-Wall", "-Wextra"] +
        ["-I" + str(path) for path in includes] + [source])
print(out, flush=True)
