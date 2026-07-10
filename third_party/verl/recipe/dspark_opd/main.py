"""DSpark-OPD recipe entry point (verl 0.7.0).

Thin hydra entry. The actual TaskRunner lives in recipe/dspark_opd/task_runner.py so that
Ray can reconstruct it in worker processes by importing a real submodule (which runs the
package __init__ compat shim first) — NOT __main__. See task_runner.py for the rationale.
"""
from __future__ import annotations

import hydra
import ray

# Side-effecting: runs package __init__ (transformers compat shim + rollout registration)
# in the driver before verl imports.
import recipe.dspark_opd  # noqa: F401

from verl.trainer.main_ppo import run_ppo
from recipe.dspark_opd.task_runner import DSparkTaskRunner


@hydra.main(config_path="config", config_name="dspark_trainer", version_base=None)
def main(config):
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DSparkTaskRunner))


if __name__ == "__main__":
    main()
