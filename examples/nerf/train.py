import argparse
import numpy as np 
import time 
from torch.utils.data import DataLoader
import os 
import imageio
import torch 
import argparse

from simdltk.data import get_dataset_cls
from simdltk.model import get_model_cls, get_arch_arch
from simdltk.training import get_trainer_cls
from simdltk import logger, parse_args, set_random_state
from simdltk.model.nerf import to8b
from simdltk.training.callbacks import Callback, logger
from simdltk.utils import bool_flag

# TODO: set eval mode of model 


def render_path(model, render_poses, gt_imgs=None, savedir=None, render_factor=0):
    """ """
    # H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        # focal = focal / render_factor
    rgbs = []
    disps = []

    t = time.time()
    print('render_poses.shape', render_poses.shape)
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t, 'c2w.shape', c2w.shape)
        t = time.time()
        ret = model(rays=None, c2w=c2w)  # forward model
        # rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgb, disp = ret['rgb_map'], ret['disp_map']
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def render_video(model, render_poses, exp_dir, epoch_counter):
    with torch.no_grad():
        model.eval()
        render_poses = render_poses.to(next(model.parameters()).device)
        rgbs, disps = render_path(model, render_poses)
        # rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
    print('Done, saving', rgbs.shape, disps.shape)
    basename = os.path.basename(exp_dir)
    moviebase = os.path.join(exp_dir, '{}_spiral_{:06d}_'.format(basename, epoch_counter))
    imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
    imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
    print('basename', basename)


class RenderCallback(Callback):
    def __init__(self, exp_dir, render_every_epochs, render_poses):
        super().__init__(use_counter=True)
        self.exp_dir = exp_dir
        self.render_every_epochs = render_every_epochs
        self.render_poses = render_poses
    
    def on_evaluate_end(self, eval_logs=None):
        if self.epoch_counter % self.render_every_epochs == 0 and self.epoch_counter:  # skip initial
            logger.info('Render at epoch {}'.format(self.epoch_counter))
            render_video(self.model, self.render_poses, self.exp_dir, self.epoch_counter)
        return super().on_evaluate_end(eval_logs)


def main(args, evaluate_best_ckpt=True):
    set_random_state(args.seed)
    # dataset
    dataset_cls = get_dataset_cls(args.dataset)
    train_data = dataset_cls.build(args, 'train')
    valid_data = dataset_cls.build(args, 'valid')
    test_data = dataset_cls.build(args, 'test')

    # model
    model_cls = get_model_cls(args.model, args.arch)
    if args.arch:
        arch = get_arch_arch(args.arch)
        arch(args)
    model = model_cls.build(args, train_data)
    logger.info(f'Model:\n{model}')

    # device = 'cuda:{}'.format(args.devices)
    # print('Test run model')
    # dl = DataLoader(train_data, batch_size=13, shuffle=False)
    # model.to(device)
    # for batch in dl:
    #     rays = batch['rays'].to(device)
    #     print('forward model test')
    #     model(rays)
    #     break
    # print('callback test')
    # render_cb = RenderCallback(args.exp_dir, args.render_every_epochs, test_data.poses)  # Use test poses instead
    # render_cb.epoch_counter = args.render_every_epochs
    # render_cb.model = model 
    # render_cb.on_evaluate_end()
    render_cb = RenderCallback(args.exp_dir, args.render_every_epochs, test_data.render_poses)  # Use test poses instead
    # training
    trainer_cls = get_trainer_cls(args.trainer)
    trainer = trainer_cls.build(args, model, train_data, valid_data, [render_cb])
    logger.info(f'Args: {args}')
    if args.only_evaluate:
        print('Render videos')
        if args.load_ref_model:
            print('load weights from', args.load_ref_model)
            pt = torch.load(args.load_ref_model)
            print('keys:', list(pt.keys()))
            model.model.load_state_dict(pt['network_fn_state_dict'])
            model.model_fine.load_state_dict(pt['network_fine_state_dict'])
        render_video(trainer.model, test_data.render_poses, args.exp_dir, 999999)
        return 
    trainer.train()
    if evaluate_best_ckpt:
        logger.info('Reload best checkpoint and test on dataset ...')
        trainer.ckpt.restore(best=True)
        trainer.evaluate(test_data)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--render-every-epochs', type=int, default=1)
    parser.add_argument('--only-evaluate', type=bool_flag, default=False)
    parser.add_argument('--load-ref-model', type=str, default='')
    args = parse_args(parser)
    main(args)
