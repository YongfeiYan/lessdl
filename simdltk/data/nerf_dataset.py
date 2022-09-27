import os
import cv2
import json
import numpy as np
import imageio 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from simdltk.data.dataset import register_dataset, BaseDataset
from simdltk.utils import bool_flag


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, split=None):
    # splits = ['train', 'val', 'test']
    assert split and isinstance(split, str) and split in ('train', 'val', 'test'), 'split should be a non-empty string'
    splits = [split]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            if not os.path.exists(fname):
                print('[WARNING] {} is not found. Skip it.'.format(fname))
                continue
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]  # train/val/test ith
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        
    return imgs, poses, render_poses, [H, W, focal]


@register_dataset('nerf_dataset_blender')
class NeRFDataset(BaseDataset, Dataset):
    def __init__(self, data_dir, half_res, testskip, split, white_bkgd) -> None:
        super().__init__()
        images, poses, self.render_poses, img_stat, = load_blender_data(data_dir, half_res, testskip, split)
        self.H, self.W, self.focal = int(img_stat[0]), int(img_stat[1]), img_stat[2]
        self.img_stat = [self.H, self.W, self.focal]
        self.white_bkgd = white_bkgd
        if white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
        self.K = np.array([
            [self.focal, 0, 0.5 * self.W],
            [0, self.focal, 0.5 * self.H],
            [0, 0, 1]
        ])
        # get rays
        rays = np.stack([get_rays_np(self.H, self.W, self.K, p) for p in poses[:, :3, :4]], 0) # [N, ro+rd, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        # prepare for training
        self.images, self.poses, self.rays_rgb, self.render_poses = torch.Tensor(images), torch.Tensor(poses), torch.Tensor(rays_rgb), torch.Tensor(self.render_poses)
        self.rays, self.target = self.rays_rgb[..., :2, :], self.rays_rgb[..., 2, :]
        
        print('images', self.images.shape)
        print('poses', self.poses.shape)
        print('rays_rgb', rays_rgb.shape)
        print('render_poses', self.render_poses.shape)
        print('head of target', self.target[:10])
        print('head of rays', self.rays[:10])
        print('head of render_poses', self.render_poses[:10])
        print('None 1s target', (self.target != 1).float().sum().item(), '1s', (self.target == 1).float().sum().item())
        # # TODO: delete 
        # print('saving:')
        # torch.save({'rays_rgb': self.rays_rgb}, '/tmp/my.pt')
        # import sys 
        # sys.exit(1)

    def __getitem__(self, index):
        return {
            'index': torch.tensor(index),
            'rays': self.rays[index],  # ro+rd,3
            'target': self.target[index]  # 3
        }
    
    def __len__(self):
        return len(self.rays_rgb)
    
    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--data-dir', type=str, )
        parser.add_argument('--half-res', type=bool_flag, default=True)
        parser.add_argument('--testskip', type=int, default=1)
        parser.add_argument('--white-bkgd', type=bool_flag, default=False)
        # parser.add_argument('--near', type=float, default=2)
        # parser.add_argument('--far', type=float, default=6)
        parser.add_argument("--dataset-type", type=str, required=True, choices=['llff', 'blender', 'deepvoxels'],
            help='options: llff / blender / deepvoxels')
    
    @staticmethod
    def build(args, split=None):
        if split == 'valid': 
            split = 'val'
        return NeRFDataset(args.data_dir, args.half_res, args.testskip, split, args.white_bkgd)


if __name__ == '__main__':
    import argparse 
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True)
    NeRFDataset.add_args(parser)
    args = parser.parse_args()
    print('Args', args)
    ds = NeRFDataset.build(args, args.split)
    dl = DataLoader(ds, batch_size=13, shuffle=True, drop_last=False)
    for batch in dl:
        print('batch rays', batch['rays'].shape)
        print('batch target', batch['target'].shape)
        assert tuple(batch['rays'].shape[1:]) == (2, 3)
        assert tuple(batch['target'].shape[1:]) == (3,)
        break
"""
Run
PYTHONPATH=.:examples/nerf python examples/nerf/dataset.py --split train --data-dir local/nerf_synthetic/lego --dataset-type blender --white-bkgd True
"""
