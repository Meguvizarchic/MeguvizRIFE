"""Build a partial compatibility pack from RIFE's real target capture.

No GPU facts are synthesized here. Unselected owner kernels and builtin ncnn
layers continue to use the runtime compiler. Output is a new generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def require(condition, cause):
    if not condition:
        raise ValueError(cause)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--kernels", default="5,6,7,8,9")
    args = parser.parse_args()
    capture = read(args.catalog)
    require(capture["schema_version"] == 1 and not capture["device_qualified"], "capture_schema_or_qualification")
    kernels = {int(value) for value in args.kernels.split(",")}
    require(kernels and kernels <= set(range(1, 10)), "kernel_selection")
    require(not args.out.exists() or (args.out.is_dir() and not any(args.out.iterdir())), "immutable_output_exists")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []

    def run(name, command):
        p = subprocess.run([str(args.tool)] + list(map(str, command)), capture_output=True)
        (args.out / (name + ".stdout.json")).write_bytes(p.stdout)
        (args.out / (name + ".stderr.log")).write_bytes(p.stderr)
        records.append({"operation": name, "argv": list(map(str, command)), "exit_code": p.returncode})
        (args.out / "commands.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        require(p.returncode == 0, name + "_failed")
        return json.loads(p.stdout)

    identity = run("version", ["--version-json"])
    lock = identity["toolchain_lock_hash"]
    bindings = []
    names = []
    found = set()
    for entry in capture["variants"]:
        kernel = entry["kernel"]
        if kernel not in kernels:
            continue
        require(kernel not in found, "duplicate_captured_kernel")
        found.add(kernel)
        request = entry["compile_request"]
        require(entry["target_facts"]["provenance"] == "captured", "synthetic_target_forbidden")
        require(request["target_facts"] == entry["target_facts"], "target_capture_mismatch")
        require(request["effective_options"] == entry["effective_options"], "effective_options_mismatch")
        require(request["toolchain_lock_hash"] == lock, "host_toolchain_capture_mismatch")
        directory = args.catalog.parent / ("kernel-" + str(kernel))
        source = (directory / "shader.comp").read_bytes()
        require(sha(source) == entry["source_sha256"] == request["source"]["sha256"], "captured_source_mismatch")
        require(read(directory / "request.json") == request, "request_file_mismatch")
        name = "kernel-" + str(kernel)
        target = args.out / "artifacts" / name
        run("compile-" + name, ["compile", "--request", directory / "request.json", "--out-dir", target])
        artifact = read(target / "artifact.json")
        receipt = read(target / "receipt.json")
        semantic = receipt["input_identity"]
        for key, value in request.items():
            expected = {"sha256": entry["source_sha256"]} if key == "source" else value
            require(semantic[key] == expected, "compile_receipt_identity_" + key)
        require(sha(canonical(semantic).encode()) == artifact["variant_key"], "variant_key_mismatch")
        require(sha(canonical(entry["effective_options"]).encode()) == artifact["effective_options_hash"], "artifact_options_mismatch")
        require(artifact.get("effective_options") == entry["effective_options"], "artifact_full_options_binding_missing_or_mismatched")
        require(artifact.get("target_facts") == entry["target_facts"], "artifact_target_binding_missing_or_mismatched")
        require(artifact.get("target_facts_hash") == entry["target_facts_hash"], "artifact_target_hash_mismatch")
        require(sha((target / "shader.spv").read_bytes()) == artifact["spirv_sha256"], "compiled_spirv_mismatch")
        require(artifact["layout_abi"]["local_size"] == entry["expected_local_size"], "local_size_mismatch")
        require(artifact["target_env"] == request["target_env"], "compiled_target_mismatch")
        # Actual VkDevice enabled capabilities are checked by Compute's pipeline
        # factory against the captured target binding, never inferred on host.
        names.append(name)
        bindings.append({"kernel": kernel, "variant_key": artifact["variant_key"],
            "effective_options_hash": artifact["effective_options_hash"], "target_facts_hash": entry["target_facts_hash"],
            "source_sha256": entry["source_sha256"], "spirv_sha256": artifact["spirv_sha256"]})
    require(found == kernels, "requested_kernel_not_active_in_plan")
    index = {"schema_version": 1, "profile_id": "ncnn-compat", "profile_version": "20261020", "artifacts": names}
    (args.out / "index.json").write_text(canonical(index) + "\n", encoding="utf-8")
    run("pack", ["pack", "--index", args.out / "index.json", "--artifacts", args.out / "artifacts", "--out-dir", args.out / "pack"])
    run("verify", ["verify", "--pack", args.out / "pack"])
    config = {"schema_version": 1, "private_sdk_fingerprint": capture["private_sdk_fingerprint"], "bindings": bindings}
    (args.out / "bindings.json").write_text(canonical(config) + "\n", encoding="utf-8")
    manifest = read(args.out / "pack/manifest.json")
    result = {"implementation": "partial RIFE compatibility shaderpack", "device_qualified": False,
        "manifest_sha256": manifest["manifest_hash"], "shaderpack_directory": str((args.out / "pack").resolve()),
        "shader_bindings_file": str((args.out / "bindings.json").resolve()), "kernels": sorted(found)}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(canonical(result))


if __name__ == "__main__":
    main()
