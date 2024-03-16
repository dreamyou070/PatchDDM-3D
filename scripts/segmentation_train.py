""" Train a diffusion model on images """
import sys
import argparse
import torch as th
import torch.utils.tensorboard
import random
sys.path.append("../PatchDDM-3D")
sys.path.append("")
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults,create_model_and_diffusion,
                                          args_to_dict,add_dict_to_argparser,)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def main():

    print(f' step 1. argument checking')
    args = create_argparser().parse_args()
    print(f' (1.1) seed')
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f' (1.2) tensorboard')
    summary_writer = None
    if args.use_tensorboard: # True
        logdir = None # logdir = None
        if args.tensorboard_path: logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text('config',
                                '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()]))
        print(f'Using Tensorboard with logdir = {summary_writer.get_logdir()}')
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()
    dist_util.setup_dist(devices=args.devices)

    print(f' step 2. creating model and diffusion...')
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments) # unet, inferer
    print("number of parameters: {:_}".format(np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler,
                                                     diffusion,
                                                     maxt=1000)

    print(f' step 3. data loader')
    if args.dataset == 'brats':
        ds = BRATSDataset(args.data_dir, test_flag=False)
        datal = th.utils.data.DataLoader(ds,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=True)
    elif args.dataset == 'brats3d':
        print(f'args.dataset = {args.dataset}')
        assert args.image_size in [128, 256]
        ds = BRATSDataset(args.data_dir,
                          test_flag=False,
                          normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
                          mode='train',
                          half_resolution=(args.image_size == 128) and not args.half_res_crop, # No
                          random_half_crop=(args.image_size == 128) and args.half_res_crop,    # True
                          concat_coords=args.concat_coords,
                          num_classes=args.out_channels, )
        datal = th.utils.data.DataLoader(ds,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=True)
    print(f' step 4. training...')
    print(f'args.in_channels = 8 (means img + output)')
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels, # 7 ?
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler, # sampler --------------------------------------------------------
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode='segmentation', # only segmentation
    ).run_loop()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # basic argument
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
    res = dict(
        image_size=64,
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
    defaults_dict.undate(res)
    diffusion_dict = dict(learn_sigma=False,
                        diffusion_steps=1000,
                        noise_schedule="linear",
                        timestep_respacing="",
                        use_kl=False,
                        predict_xstart=False,
                        rescale_timesteps=False,
                        rescale_learned_sigmas=False,
                        dataset='brats',
                        dims=2,
                        num_groups=32,
                        in_channels=1,)
    defaults_dict.update(diffusion_dict)
    add_dict_to_argparser(parser, defaults_dict)
    main()
