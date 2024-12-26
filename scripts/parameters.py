def default_args(test=False):
    if test:
        defaults = dict(       
            log_path = "./logs",
            training_log_folder = None, #"./logs/DSIFN_2024-08-16-02-18-13",
            test_dir = "../DSIFN/test", #"../sen12/opt/test", #
            test_length = 200,
            test_checkpoint = None,
            cycle_spinning = False,
            cycle_width = 100,
            batch_size = 1,
            seed = 123,
            sample_to_use = "SWEEP", #"MAX", #"LAST", #
            model_to_use = "MAX", #"BEST", #
            images_path = "test_images",

            use_ddim = False,
            timestep_respacing = "100", #[0,0,11,8,6], #[0,0,6,8,11], #[0,0,12,16,22], #[0,0,24,32,44], #[36,24,24,8,8],
        )
    else: # Train
        defaults = dict(
            log_path = "./logs",
            train_dir = "../DSIFN/train", #"../sen12/opt/train", #
            val_dir = "../DSIFN/val", #"../sen12/opt/val", #
            val_samples = 40,
            log_interval = 100,
            save_interval = 1000,
            in_channels = 3,
            batch_size = 8,
            seed = 123,
            ema_rate = "",
            lr = 1e-4,
            lr_anneal_steps = 8000,
            resume_checkpoint = "./weights/64_256_upsampler.pt",
            diffusion_steps = 1000,
            use_ddim = False,
            timestep_respacing = "100", #"ddim100",
            learn_sigma = True,
            predict_xstart = False,
            use_fp16 = True,
            noise_schedule = "linear",
        )
    return defaults