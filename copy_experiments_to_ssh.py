from pathlib import Path
import shutil

logs_dir = Path("logs")
ssh_dir = Path("ssh")

ssh_dir.mkdir(exist_ok=True)

for experiment_dir in logs_dir.iterdir():
    if not experiment_dir.is_dir():
        continue

    experiment_name = experiment_dir.name

    i = 0
    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue

        timestamp = run_dir.name
        train_log = run_dir / "train.log"

        if train_log.exists():
            target_name = f"{experiment_name}_{i}"
            target_path = ssh_dir / target_name

            shutil.copy2(train_log, target_path)
            print(f"Copied {train_log} -> {target_path}")

            i += 1
