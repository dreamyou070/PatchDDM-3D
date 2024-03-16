import torch as th
import numpy as np


defaults_dict = dict(seed=0,
                    data_dir="",
                    schedule_sampler="uniform",
                    lr=1e-4,
                    weight_decay=0.0,
                    lr_anneal_steps=0,
                    batch_size=1,
                    microbatch=-1,
                    ema_rate="0.9999",
                    log_interval=100,
                    save_interval=5000,
                    resume_checkpoint='',
                    use_fp16=False,
                    fp16_scale_growth=1e-3,
                    dataset='brats3d',
                    use_tensorboard=True,
                    tensorboard_path='',  # set path to existing logdir for resuming
                    devices=[0],
                    dims=3,  # 2 for 2d images, 3 for 3d volumes
                    learn_sigma=False,
                    num_groups=29,
                    channel_mult="1,3,4,4,4,4,4",
                    in_channels=5,  # 4 MRI sequences + out_channels  (+ #dimensions if concat_coords==True)
                    out_channels=1,  # out_channels = number  of classes
                    bottleneck_attention=False,
                    num_workers=0,
                    resample_2d=False,
                    mode='segmentation',
                    renormalize=True,
                    additive_skips=True,
                    decoder_device_thresh=15,
                    half_res_crop=False,
                    concat_coords=False,)
res = dict(image_size=64,
            num_channels=128,
            num_res_blocks=2,
            num_heads=4,
            num_heads_upsample=-1,
            num_head_channels=-1,
            attention_resolutions="16,8",
            channel_mult="",
            dropout=0.0,
            class_cond=False,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_fp16=False,
            use_new_attention_order=False,
            dims=2,
            num_groups=32,
            in_channels=1,
            out_channels=0,  # automatically determine if 0
            bottleneck_attention=True,
            resample_2d=True,
            additive_skips=False,
            decoder_device_thresh=0,
            mode='default',)
defaults_dict.update(res)