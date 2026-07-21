"""DSpark-OPD TaskRunner (defined in a real submodule, NOT __main__).

WHY a separate module: when the entry is launched via `python -m recipe.dspark_opd.main`,
anything defined in that entry module has __module__ == "__main__". Ray reconstructs a
remote actor by IMPORTING its defining module in the worker process; it cannot cleanly
re-import "__main__", so our package __init__ (transformers compat shim + rollout
registration) would NOT run on the worker before verl imports transformers -> crash on
`AutoModelForVision2Seq`. Defining the actor here (module = recipe.dspark_opd.task_runner)
means Ray imports THIS submodule on the worker, which Python guarantees runs
recipe/dspark_opd/__init__.py (the shim) FIRST. See docs/opd/env-setup.md.
"""
from __future__ import annotations

# Side-effecting import: runs recipe/dspark_opd/__init__.py (compat shim + rollout
# registration) BEFORE any verl import below. Guaranteed to run on Ray workers too,
# because they import THIS module to reconstruct the actor.
import recipe.dspark_opd  # noqa: F401

import ray

from verl.trainer.main_ppo import TaskRunner as MainTaskRunner


class DSparkTaskRunner(MainTaskRunner):
    def add_actor_rollout_worker(self, config):
        # Reuse parent to get ray_worker_group_cls + strategy checks, then override class.
        actor_rollout_cls, ray_worker_group_cls = super().add_actor_rollout_worker(config)
        from verl.trainer.ppo.ray_trainer import Role
        from recipe.dspark_opd.worker import DSparkActorRolloutRefWorker

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(DSparkActorRolloutRefWorker)
        return DSparkActorRolloutRefWorker, ray_worker_group_cls

    def add_reward_model_worker(self, config):
        # FUSED design: teacher is built INSIDE the actor worker (DSparkActorRolloutRefWorker.
        # _build_teacher) and scored inside train_step — NOT a separate RewardModel role. So we
        # do NOT register a reward worker (skip super()); config.reward_model.enable stays false
        # (or is ignored). This removes a whole worker group + its dispatch/collect round-trips.
        return

    def run(self, config):
        import socket
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.reward import load_reward_manager

        from recipe.dspark_opd.dataset import dspark_collate_fn
        from recipe.dspark_opd.trainer import DSparkTrainer

        # DSpark samples are variable-length; verl's default stack-based collate can't
        # batch them. Use our right-padding collate (see dataset.dspark_collate_fn).
        collate_fn = dspark_collate_fn

        print(f"[DSparkTaskRunner] host={socket.gethostname()}")

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        # draft ckpt has no tokenizer files -> load from TARGET (draft shares vocab, §2.2)
        tok_path = config.actor_rollout_ref.model.get("override_config", {}).get(
            "dspark_tokenizer_path") or local_path
        tokenizer = hf_tokenizer(tok_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(tok_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor, is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor, is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = DSparkTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

        trainer.init_workers()
        trainer.fit()
