"""Our additions to the Cosmos-Policy experiment config, as an append-only excerpt.

The upstream file (cosmos_policy/configs/.../experiment_config.py) carries an NVIDIA
proprietary notice, so only the block we add is distributed here. To reproduce:

  1. obtain experiment_config.py from the Cosmos-Policy release you are using;
  2. append everything below the marker to it;
  3. add cosmos_predict2_2b_480p_libero__failsafe_ft_v1 to the list inside
     register_configs(), alongside the other LIBERO entries.

Nothing upstream is modified other than that one registration entry. Training knobs are
read from the environment: FAILSAFE_ROLLOUT_DIR selects the rollout mixture,
FAILSAFE_MAX_ITER sets both trainer.max_iter and scheduler.cycle_lengths[0],
FAILSAFE_SAVE_ITER the checkpoint interval, FAILSAFE_SEED the seed.
"""

# === FailSafe-WM injected (do not duplicate) ===
libero_failsafe_mixture_dataset = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=True,
    # The one difference from the upstream config: the rollout pool is the released
    # all_episodes plus our failure counterfactuals, merged as a directory of symlinks.
    rollout_data_dir=os.environ.get(
        "FAILSAFE_ROLLOUT_DIR", os.path.join(BASE_DATASETS_DIR, "rollout_mixed")
    ),
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
cosmos_predict2_2b_480p_libero__failsafe_ft_v1 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            {"override /callbacks": ["basic", "long", "cluster_speed"]},
            "_self_",
        ],
        checkpoint=dict(
            load_path=get_checkpoint_path(
                "hf://nvidia/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"
            ),
            load_training_state=False,
            strict_resume=False,
            save_iter=int(os.environ.get("FAILSAFE_SAVE_ITER", "2500")),
        ),
        optimizer=dict(lr=1e-5),
        scheduler=dict(
            cycle_lengths=[int(os.environ.get("FAILSAFE_MAX_ITER", "20000")), 100000000000000],
            warm_up_steps=[200, 0],
            f_start=[1e-6, 0.06], f_max=[1.0, 0.06], f_min=[0.3, 0.06],
        ),
        trainer=dict(max_iter=int(os.environ.get("FAILSAFE_MAX_ITER", "20000"))),
        dataloader_train=dict(
            dataset=libero_failsafe_mixture_dataset,
            sampler=dict(dataset=libero_failsafe_mixture_dataset),
            batch_size=int(os.environ.get("FAILSAFE_BATCH_SIZE", "30")),
            num_workers=int(os.environ.get("FAILSAFE_NUM_WORKERS", "12")),
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2_2b_480p_libero__failsafe_ft_v1"),
    )
)


def register_configs():
    cs = ConfigStore.instance()
    # Register the experiments
    for _item in [
        # LIBERO
        cosmos_predict2_2b_480p_libero,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_libero__inference_only,
        cosmos_predict2_2b_480p_libero__failsafe_ft_v1,  # FailSafe-WM
        # RoboCasa
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference,
        # ALOHA
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only,
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func,  # ALOHA planning model
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
