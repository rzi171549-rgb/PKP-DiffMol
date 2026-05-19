"""Unified training wrapper for final PKP-DiffMol configs.

This script reads a final YAML config, resolves the preserved legacy evidence
training entry, and forwards only argparse options that are statically declared
by that entry or its local helper modules. It does not rewrite the training
logic; runtime behavior remains in pkpdiffmol/runtime_legacy.
"""

from __future__ import annotations

import argparse
import ast
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


ENTRY_MAP = {
    "bace_stagec_residual_mix": "pkpdiffmol/runtime_legacy/evidence/bace/scripts/train_bace_chemprior_parallel_latentdiff_stagec_residual_mix.py",
    "bbbp_pooler_latentdiff": "pkpdiffmol/runtime_legacy/evidence/bbbp/scripts/train_bbbp_pool_pooler_base_tunev2_latentdiff.py",
    "muv_formal4fix_v2": "pkpdiffmol/runtime_legacy/evidence/muv/scripts/train_muv_chemprior_parallel_latentdiff_muvformal4fix_v2.py",
    "sider_formal4_s2": "pkpdiffmol/runtime_legacy/evidence/sider/scripts/train_sider_chemprior_parallel_latentdiff.py",
    "tox21_general_latentdiff": "pkpdiffmol/runtime_legacy/evidence/tox21/scripts/train_tox21_chemprior_parallel_latentdiff.py",
    "toxcast_old_g4_latentdiff": "pkpdiffmol/runtime_legacy/evidence/toxcast/scripts/train_toxcast_chemprior_parallel_latentdiff.py",
    "clintox_general_latentdiff": "pkpdiffmol/runtime_legacy/evidence/clintox/scripts/train_clintox_chemprior_parallel_latentdiff.py",
}

IMPLEMENTATION_BY_DATASET = {
    "bace": "bace_stagec_residual_mix",
    "bbbp": "bbbp_pooler_latentdiff",
    "muv": "muv_formal4fix_v2",
    "sider": "sider_formal4_s2",
    "tox21": "tox21_general_latentdiff",
    "toxcast": "toxcast_old_g4_latentdiff",
    "clintox": "clintox_general_latentdiff",
}

REQUIRED_CONFIG_KEYS = (
    "dataset",
    "seed",
    "training",
    "optimization",
    "dropout",
    "representation",
    "diffusion",
    "paths",
)

ARG_MAPPINGS = (
    ("seed", ("__root__", "seed"), ("--seed",)),
    ("epochs", ("optimization", "epochs"), ("--epochs",)),
    ("batch_size", ("optimization", "batch_size"), ("--batch-size", "--batch_size")),
    ("lr", ("optimization", "lr"), ("--lr",)),
    ("warmup_ratio", ("optimization", "warmup_ratio"), ("--warmup-ratio", "--warmup_ratio")),
    ("pooler_dropout", ("dropout", "pooler_dropout"), ("--pooler-dropout", "--pooler_dropout")),
    ("desc_dropout", ("dropout", "desc_dropout"), ("--desc-dropout", "--desc_dropout")),
    ("fusion_dropout", ("dropout", "fusion_dropout"), ("--fusion-dropout", "--fusion_dropout")),
    ("fusion_mode", ("representation", "fusion_mode"), ("--fusion-mode", "--fusion_mode")),
    ("fusion_alpha_init", ("representation", "fusion_alpha_init"), ("--fusion-alpha-init", "--fusion_alpha_init")),
    ("synthetic_rho", ("diffusion", "synthetic_rho"), ("--synthetic-rho", "--synthetic_rho")),
    ("qgate_quantile", ("diffusion", "qgate_quantile"), ("--qgate-quantile", "--qgate_quantile")),
    (
        "diffusion_num_timesteps",
        ("diffusion", "diffusion_num_timesteps"),
        ("--diffusion-num-timesteps", "--diffusion_num_timesteps"),
    ),
    ("stage_c_epochs", ("diffusion", "stage_c_epochs"), ("--stage-c-epochs", "--stage_c_epochs")),
    ("stage_c_lr", ("diffusion", "stage_c_lr"), ("--stage-c-lr", "--stage_c_lr")),
    (
        "stage_c_batch_size",
        ("diffusion", "stage_c_batch_size"),
        ("--stage-c-batch-size", "--stage_c_batch_size"),
    ),
    (
        "stage_c_weight_decay",
        ("diffusion", "stage_c_weight_decay"),
        ("--stage-c-weight-decay", "--stage_c_weight_decay"),
    ),
    (
        "stage_c_train_scope",
        ("diffusion", "stage_c_train_scope"),
        ("--stage-c-train-scope", "--stage_c_train_scope"),
    ),
    ("final_eval_mode", ("diffusion", "final_eval_mode"), ("--final-eval-mode", "--final_eval_mode")),
    ("checkpoint", ("paths", "checkpoint"), ("--checkpoint",)),
    ("dictionary", ("paths", "dictionary"), ("--dict", "--dictionary")),
    ("data_dir", ("paths", "data_dir"), ("--data-dir", "--data_dir")),
    (
        "text_encoder_name_or_path",
        ("representation", "text_encoder_name_or_path"),
        ("--text-encoder-name-or-path", "--text_encoder_name_or_path"),
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].strip()
    return value.strip()


def parse_scalar(text: str) -> Any:
    value = strip_inline_comment(text)
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_release_yaml(path: Path) -> dict[str, Any]:
    """Parse the simple config shape used by configs/final/*.yaml."""
    data: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and ":" in line and not line.startswith("-"):
            key, raw_value = line.split(":", 1)
            key = key.strip()
            if raw_value.strip() == "":
                data[key] = {}
                current_section = key
            else:
                data[key] = parse_scalar(raw_value)
                current_section = None
        elif indent == 2 and current_section and ":" in line and not line.startswith("-"):
            key, raw_value = line.split(":", 1)
            section = data.setdefault(current_section, {})
            if isinstance(section, dict):
                section[key.strip()] = parse_scalar(raw_value)
    return data


def normalize_dataset(value: Any) -> str:
    return str(value).strip().lower()


def get_nested(config: dict[str, Any], path: tuple[str, str]) -> Any:
    section, key = path
    if section == "__root__":
        return config.get(key)
    value = config.get(section)
    if isinstance(value, dict):
        return value.get(key)
    return None


def resolve_entry_path(root: Path, dataset: str, config: dict[str, Any]) -> Path:
    training = config.get("training")
    if isinstance(training, dict) and training.get("implementation"):
        implementation = str(training["implementation"])
        if implementation not in ENTRY_MAP:
            raise SystemExit(f"Unsupported training implementation: {implementation!r}")
        return root / ENTRY_MAP[implementation]

    if isinstance(training, dict) and training.get("entry"):
        configured = Path(str(training["entry"]))
        return configured if configured.is_absolute() else root / configured

    training_entry = config.get("training_entry")
    if isinstance(training_entry, dict):
        packaged_path = training_entry.get("packaged_path")
        if packaged_path:
            candidate = root / str(packaged_path)
            if candidate.exists():
                return candidate

    implementation = IMPLEMENTATION_BY_DATASET[dataset]
    return root / ENTRY_MAP[implementation]


def resolve_config_path(root: Path, config_path: Path) -> Path:
    path = config_path if config_path.is_absolute() else root / config_path
    return path.resolve()


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_local_imports(path: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    candidates: set[Path] = set()
    search_dirs = [
        path.parent,
        path.parent.parent / "modules",
        path.parent.parent / "scripts",
        path.parent.parent / "scripts_shell",
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module_name = node.module.split(".")[-1]
            if module_name.startswith("_"):
                continue
            for search_dir in search_dirs:
                candidate = search_dir / f"{module_name}.py"
                if candidate.exists():
                    candidates.add(candidate.resolve())
    return candidates


def collect_supported_options(path: Path, visited: set[Path] | None = None, depth: int = 0) -> set[str]:
    if visited is None:
        visited = set()
    path = path.resolve()
    if path in visited or depth > 4:
        return set()
    visited.add(path)

    options: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return options

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    options.add(arg.value)

    for imported in resolve_local_imports(path):
        options.update(collect_supported_options(imported, visited, depth + 1))
    return options


def choose_supported_flag(aliases: tuple[str, ...], supported_options: set[str]) -> str | None:
    for alias in aliases:
        if alias in supported_options:
            return alias
    return None


def build_command(
    root: Path,
    entry_path: Path,
    config: dict[str, Any],
    supported_options: set[str],
    extra_args: str | None,
) -> tuple[list[str], list[dict[str, str]]]:
    command = ["python", relative_path(root, entry_path)]
    skipped: list[dict[str, str]] = []

    for name, config_path, aliases in ARG_MAPPINGS:
        value = get_nested(config, config_path)
        if value is None:
            skipped.append({"name": name, "reason": "not set in config"})
            continue
        flag = choose_supported_flag(aliases, supported_options)
        if flag is None:
            skipped.append({"name": name, "reason": f"unsupported by entry argparse ({'/'.join(aliases)})"})
            continue
        command.extend([flag, str(value)])

    if extra_args:
        command.extend(shlex.split(extra_args))
    return command, skipped


def validate_config(config: dict[str, Any]) -> list[str]:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    return missing


def path_status(root: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    checks = [
        ("checkpoint", get_nested(config, ("paths", "checkpoint")), "file"),
        ("dictionary", get_nested(config, ("paths", "dictionary")), "file"),
        ("text_encoder_name_or_path", get_nested(config, ("representation", "text_encoder_name_or_path")), "dir"),
        ("data_dir", get_nested(config, ("paths", "data_dir")), "dir"),
    ]
    missing: list[dict[str, str]] = []
    for name, raw_path, kind in checks:
        if not raw_path:
            missing.append({"name": name, "path": "", "reason": "not set in config"})
            continue
        path = root / str(raw_path)
        exists = path.is_dir() if kind == "dir" else path.is_file()
        if not exists:
            missing.append({"name": name, "path": str(raw_path), "reason": f"missing {kind}"})

    data_dir = get_nested(config, ("paths", "data_dir"))
    if data_dir:
        for split in ("train.lmdb", "valid.lmdb", "test.lmdb"):
            split_path = root / str(data_dir) / split
            if not split_path.is_file():
                missing.append({"name": split, "path": f"{data_dir}/{split}", "reason": "missing file"})
    return missing


def print_dry_run(
    dataset: str,
    config_path: Path,
    entry_path: Path,
    command: list[str],
    supported_options: set[str],
    skipped: list[dict[str, str]],
    missing_paths: list[dict[str, str]],
) -> None:
    print(f"dataset: {dataset}")
    print(f"config: {config_path.as_posix()}")
    print(f"entry: {entry_path.as_posix()}")
    print(f"supported_arg_count: {len(supported_options)}")
    print("command:")
    print(f"  {shlex.join(command)}")
    print("skipped_args:")
    print(json.dumps(skipped, ensure_ascii=False, indent=2))
    print("missing_paths:")
    print(json.dumps(missing_paths, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a final PKP-DiffMol config via the legacy evidence entry.")
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/final/<dataset>.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved training command only.")
    parser.add_argument("--extra-args", default="", help="Additional raw CLI arguments appended after config args.")
    args = parser.parse_args()

    root = repo_root()
    config_path = resolve_config_path(root, args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_release_yaml(config_path)
    missing_config = validate_config(config)
    if missing_config:
        raise SystemExit(f"Config is missing required fields: {', '.join(missing_config)}")

    dataset = normalize_dataset(config.get("dataset"))
    if dataset not in IMPLEMENTATION_BY_DATASET:
        raise SystemExit(f"Unsupported dataset in config: {config.get('dataset')!r}")

    entry_path = resolve_entry_path(root, dataset, config)
    if not entry_path.is_file():
        raise FileNotFoundError(f"Legacy training entry not found: {entry_path}")

    supported_options = collect_supported_options(entry_path)
    command, skipped = build_command(root, entry_path, config, supported_options, args.extra_args)
    missing_paths = path_status(root, config)

    if args.dry_run:
        print_dry_run(dataset, config_path, entry_path, command, supported_options, skipped, missing_paths)
        return 0

    if missing_paths:
        print_dry_run(dataset, config_path, entry_path, command, supported_options, skipped, missing_paths)
        raise SystemExit("Required assets or data are missing; prepare them before running training.")

    completed = subprocess.run(command, cwd=root, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
