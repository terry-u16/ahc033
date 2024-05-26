import json
import math
import re
import shutil
import subprocess
import numpy as np

import optuna

TIME_RATIO = 1.5


class Objective:
    def __init__(self) -> None:
        pass

    def __call__(self, trial: optuna.trial.Trial) -> float:
        k2 = trial.suggest_float("k2", 0.2, 10.0, log=True)
        k3 = trial.suggest_float("k3", 0.2, 10.0, log=True)
        temp0 = trial.suggest_float("temp0", 1e-1, 1e1, log=True)
        temp1 = trial.suggest_float("temp1", 1e-2, 3e-1, log=True)
        neigh0 = trial.suggest_float("neigh0", 1e-3, 1e0, log=True)
        neigh1 = trial.suggest_float("neigh1", 1e-3, 1e0, log=True)
        neigh2 = trial.suggest_float("neigh2", 1e-3, 1e0, log=True)
        neigh3 = trial.suggest_float("neigh3", 1e-3, 1e0, log=True)
        neigh4 = trial.suggest_float("neigh4", 1e-3, 1e0, log=True)
        neigh5 = trial.suggest_float("neigh5", 1e-3, 1e0, log=True)
        neigh6 = trial.suggest_float("neigh6", 1e-3, 1e0, log=True)

        min_seed = 0
        max_seed = 511
        batch_size = 16
        score_sum = 0.0
        seed_sum = 0
        args = f"{k2} {k3} {temp0} {temp1} {neigh0} {neigh1} {neigh2} {neigh3} {neigh4} {neigh5} {neigh6}"
        local_execution = f"ahc033.exe {args}"
        cloud_execution = f"ahc033 {args}"

        # https://tech.preferred.jp/ja/blog/wilcoxonpruner/
        instances = []
        for begin in range(min_seed, max_seed + 1, batch_size):
            end = begin + batch_size
            instances.append((begin, end))

        instance_ids = np.random.permutation(len(instances))

        print(f">> {local_execution}")

        for instance_id in instance_ids:
            begin, end = instances[instance_id]
            seed_sum += batch_size

            with open("runner_config_original.json", "r") as f:
                config = json.load(f)

            config["RunnerOption"]["StartSeed"] = begin
            config["RunnerOption"]["EndSeed"] = end
            config["ExecutionOption"]["LocalExecutionSteps"][0][
                "ExecutionCommand"
            ] = local_execution
            config["ExecutionOption"]["CloudExecutionSteps"][0][
                "ExecutionCommand"
            ] = cloud_execution

            with open("runner_config.json", "w") as f:
                json.dump(config, f, indent=2)

            command = "dotnet marathon run-local"
            process = subprocess.run(
                command, stdout=subprocess.PIPE, encoding="utf-8", shell=True
            )

            lines = process.stdout.splitlines()
            score_pattern = r"rate:\s*(\d+.\d+)%"
            instance_score_sum = 0.0

            for line in lines:
                result = re.search(score_pattern, line)
                if result:
                    score = float(result.group(1))
                    # if score > 0.0:
                    #    score = math.log10(score)
                    # else:
                    #    score = math.log10(10.00)
                    instance_score_sum += score

            trial.report(instance_score_sum, instance_id)
            score_sum += instance_score_sum
            print(f"{score_sum / seed_sum:.3f}", end=" ", flush=True)

            if trial.should_prune():
                print(f"Trial {trial.number} pruned.")
                # raise optuna.TrialPruned()
                return score_sum / seed_sum

        print(f"{score_sum / seed_sum:.5f}")
        return score_sum / seed_sum


if __name__ == "__main__":
    STUDY_NAME = "ahc033-001"

    # subprocess.run("dotnet marathon compile-rust")
    subprocess.run("cargo build --release", shell=True)
    shutil.move("../target/release/ahc033.exe", "./ahc033.exe")

    objective = Objective()

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage="mysql+pymysql://default@localhost/optuna",
        load_if_exists=True,
        pruner=optuna.pruners.WilcoxonPruner(),
    )

    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "k2": 3.0,
                "k3": 1.0,
                "temp0": 1e0,
                "temp1": 1e-1,
                "neigh0": 1e0,
                "neigh1": 1e0,
                "neigh2": 1e0,
                "neigh3": 1e0,
                "neigh4": 1e0,
                "neigh5": 1e0,
                "neigh6": 1e0,
            }
        )

    study.optimize(objective, timeout=3000)
    print(study.best_trial)

    optuna.visualization.plot_param_importances(study).show()
    # optuna.visualization.plot_slice(study, params=["r", "k"]).show()
    # optuna.visualization.plot_contour(study).show()
