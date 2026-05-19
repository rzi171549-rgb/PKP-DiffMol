import os
import subprocess
from pathlib import Path

root = Path("/home/featurize/work/smi-editor-main")
os.chdir(root)

py = "/home/featurize/work/smi39/bin/python"
script = "tasks/muv/scripts/train_muv_chemprior_parallel_latentdiff_muvformal4fix_v2.py"

run_tag = os.environ.get("RUN_TAG", "MUV4_RUN")
start_seed = int(os.environ["START_SEED"])
end_seed = int(os.environ["END_SEED"])
machine_tag = os.environ["MACHINE_TAG"]

outroot = root / "tasks/muv/outputs" / run_tag / machine_tag
logdir = root / "tasks/muv/outputs/logs" / run_tag / machine_tag
outroot.mkdir(parents=True, exist_ok=True)
logdir.mkdir(parents=True, exist_ok=True)

configs = [
    ("P1", ["--synthetic-rho","0.03","--qgate-quantile","0.975","--stage-c-epochs","18","--stage-c-lr","2.2e-4","--stage-c-weight-decay","0.005"]),
    ("P2", ["--synthetic-rho","0.05","--qgate-quantile","0.99","--stage-c-epochs","20","--stage-c-lr","1.5e-4","--stage-c-weight-decay","0.005"]),
    ("P3", ["--synthetic-rho","0.02","--qgate-quantile","0.99","--stage-c-epochs","12","--stage-c-lr","1.0e-4","--stage-c-weight-decay","0.01"]),
]

env = os.environ.copy()
env["PYTHONPATH"] = "/home/featurize/work/Uni-Core:/home/featurize/work/smi-editor-main:/home/featurize/work/smi-editor-main/Uni-Mol-main/unimol:" + env.get("PYTHONPATH", "")
env["LD_LIBRARY_PATH"] = "/home/featurize/work/smi39/lib:" + env.get("LD_LIBRARY_PATH", "")
env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

print("RUN_TAG", run_tag, flush=True)
print("MACHINE_TAG", machine_tag, flush=True)
print("SEEDS", start_seed, end_seed, flush=True)

for seed in range(start_seed, end_seed + 1):
    for cfg, args in configs:
        sdir = outroot / cfg / ("seed_" + str(seed))
        log = logdir / (cfg + "_seed_" + str(seed) + ".log")
        sdir.mkdir(parents=True, exist_ok=True)

        if list(sdir.rglob("summary.json")):
            print("skip", cfg, seed, flush=True)
            continue

        cmd = [py, script, "--seed", str(seed), "--output-dir", str(sdir)] + args
        print("start", cfg, seed, flush=True)
        print("cmd", " ".join(cmd), flush=True)

        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            ret = p.wait()
            f.write("\nret=" + str(ret) + "\n")

        if ret == 0:
            deleted = 0
            for pt in sdir.rglob("*.pt"):
                pt.unlink()
                deleted += 1
            print("done", cfg, seed, "deleted_pt", deleted, flush=True)
        else:
            print("failed", cfg, seed, "keep_pt", flush=True)

print("all done", flush=True)
