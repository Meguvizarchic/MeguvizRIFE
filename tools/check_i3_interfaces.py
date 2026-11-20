"""Necessary I3 frontend/configuration checks; no model, GPU or benchmark run."""
import json
from pathlib import Path
import subprocess
import time

project = Path(__file__).resolve().parents[1]
root = project.parent
sdk = Path(r"C:\Users\Administrator\AppData\Local\Android\Sdk")
ndk = sdk / "ndk/30.0.16248370"
cmake = sdk / "cmake/4.1.2/bin/cmake.exe"
javac = Path(r"C:\Program Files\Android\Android Studio\jbr\bin\javac.exe")
out = project / "artifacts/implementation" / ("i3-interfaces-" + str(time.time_ns()))
out.mkdir(parents=True)
records = []


def run(name, argv, expected=0):
    p = subprocess.run(list(map(str, argv)), cwd=root, capture_output=True)
    (out / (name + ".stdout.log")).write_bytes(p.stdout)
    (out / (name + ".stderr.log")).write_bytes(p.stderr)
    records.append({"name": name, "argv": list(map(str, argv)), "exit_code": p.returncode,
                    "expected_exit_code": expected, "device_executed": False})
    (out / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
    print(name, p.returncode, flush=True)
    if p.returncode != expected:
        print(p.stdout.decode(errors="replace")[-4000:])
        print(p.stderr.decode(errors="replace")[-10000:])
        raise SystemExit(1)


manifest = (project / "models/catalog/rife46.manifest.json").read_text()
(out / "model_catalog.h").write_text('#pragma once\ninline constexpr const char* mgm_model_catalog=R"mgm(' + manifest + ')mgm";\n')
compiler = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
for name, source in [("media-frame-jni", "distribution/android/frame_jni.cpp"),
                     ("media-cadence-core", "src/core/cadence.cpp"),
                     ("media-cadence-jni", "distribution/android/cadence_jni.cpp")]:
    run(name + "-syntax", [compiler, "--target=aarch64-none-linux-android27",
        "--sysroot=" + str(ndk / "toolchains/llvm/prebuilt/windows-x86_64/sysroot"), "-std=c++17",
        "-fsyntax-only", "-fexceptions", "-Wall", "-Wextra", "-I" + str(out),
        "-I" + str(project / "include"), "-I" + str(root / "MeguvizCompute/include"),
        "-I" + str(root / "MeguvizMedia/include"), root / "MeguvizMedia" / source])
sources = sorted((root / "MeguvizMedia/bindings/android/src/main/java").rglob("*.java"))
run("media-api-javac", [javac, "-source", "8", "-target", "8", "-encoding", "UTF-8",
    "-classpath", sdk / "platforms/android-37.2/android.jar", "-d", out / "java"] + sources)

# Isolated CMake contract fixture. A target-only fake dependency never executes
# Compute or RIFE and does not stand in for an inference/provider success gate.
for label, actual, expected in [("matching", "a" * 64, 0), ("mismatch", "b" * 64, 1)]:
    case = out / label
    dep = case / "compute"
    rife = case / "rife"
    dep.mkdir(parents=True)
    rife.mkdir()
    (dep / "MeguvizComputeConfig.cmake").write_text('add_library(MeguvizCompute::backend_ncnn_sdk INTERFACE IMPORTED)\nset_property(TARGET MeguvizCompute::backend_ncnn_sdk PROPERTY INTERFACE_MGC_PRIVATE_SDK_FINGERPRINT "' + actual + '")\n')
    (dep / "MeguvizComputeConfigVersion.cmake").write_text('set(PACKAGE_VERSION "1.2.2")\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\n')
    template = (project / "cmake/MeguvizRIFEConfig.cmake.in").read_text()
    configured = template.replace("@PACKAGE_INIT@", "macro(check_required_components name)\nendmacro()").replace("@MGR_PRIVATE_FP@", "a" * 64)
    (rife / "MeguvizRIFEConfig.cmake").write_text(configured)
    (rife / "MeguvizRIFETargets.cmake").write_text('file(WRITE "${CMAKE_BINARY_DIR}/targets-loaded.txt" "loaded")\n')
    (case / "CMakeLists.txt").write_text('cmake_minimum_required(VERSION 3.31)\nproject(fingerprint_contract NONE)\nset(MeguvizCompute_DIR "' + dep.as_posix() + '")\nfind_package(MeguvizRIFE REQUIRED CONFIG PATHS "' + rife.as_posix() + '" NO_DEFAULT_PATH)\n')
    run("cmake-" + label, [cmake, "-S", case, "-B", case / "build", "-G", "Ninja",
        "-DCMAKE_MAKE_PROGRAM=" + str(sdk / "cmake/4.1.2/bin/ninja.exe")], expected)
    assert (case / "build/targets-loaded.txt").exists() == (label == "matching")
print(out, flush=True)
