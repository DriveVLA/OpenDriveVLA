#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import numpy as np
from skimage.draw import polygon
from pytorch_lightning.metrics.metric import Metric

DEBUG = False

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension

import matplotlib.pyplot as plt
class PlanningMetric(Metric):
    def __init__(
        self,
        n_future=6,
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)
        dx, bx, _ = gen_dx_bx([-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0])
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)

        _, _, self.bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )
        self.bev_dimension = self.bev_dimension.numpy()

        self.W = 1.85
        self.H = 4.084

        self.n_future = n_future

        self.curr_obj_box_col = 0

        self.add_state("obj_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("obj_box_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        self.add_state("L2", default=torch.zeros(self.n_future),dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.gt_collision = 0
        self.pred_collision = 0

    def evaluate_single_coll(self, traj, segmentation, token=None):
        '''
        gt_segmentation
        traj: torch.Tensor (n_future, 2) # in meters
        segmentation: torch.Tensor (n_future, 200, 200) # in pixels
        '''
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1) # all points inside the box (car)

        if DEBUG:
            plt.figure()
            plt.imshow(segmentation[0].cpu().numpy(), cmap='gray')
        #     plt.scatter(rc[:,1], rc[:,0], c='r', s=1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        trajs[:,:,[0,1]] = trajs[:,:,[1,0]] # can also change original tensor
        trajs = trajs / self.dx
        trajs = trajs.cpu().numpy() + rc # (n_future, 32, 2) # all points during the trajectory

        r = trajs[:,:,0].astype(np.int32) # (n_future, 32) decompose the points into row # TODO remove np.round
        r = np.clip(r, 0, self.bev_dimension[0] - 1) 

        c = trajs[:,:,1].astype(np.int32) # (n_future, 32) decompose the points into column
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        if DEBUG:
            for t in range(r.shape[0]):
                plt.scatter(c[t], r[t], c='b', s=1)
            plt.show()
            plt.savefig(f'vis_gt_traj_vad_stp3/{token}.png')
            plt.close()

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())

        # if collision.any() and len(collision)==6 and DEBUG and token:
        #     collision_time_steps = np.where(collision)[0]
        #     for ts in collision_time_steps:
        #         if 0 in segmentation[ts].cpu().numpy():
        #             plt.figure()
        #             plt.imshow(segmentation[ts].cpu().numpy(), cmap='gray')
        #             plt.scatter(c[ts], r[ts], c='b', s=1)
        #             plt.savefig(f'vis_vad_gt_collision_wP/{token}_{ts}.png')
        #             plt.close()

        if collision.any() and len(collision)==6 and token:
            collision_time_steps = np.where(collision)[0]
            flag = True
            for ts in collision_time_steps:
                if 0 in segmentation[ts].cpu().numpy():
                    if flag:
                        self.gt_collision += 1
                        # print(f'Collision at {token} timestep {ts}')
                        flag = False
                    
        # flag = False
        # l = len(collision)
        # for i in range(l):
        #     if flag:
        #         collision[i] =True
        #     if collision[i]:
        #         flag = True

        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(self, trajs, gt_trajs, segmentation, token=None):
        '''
        trajs: torch.Tensor (B, n_future, 2)
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)
        '''
        B, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i], token=token) # collision of trajectory points within the map

            xx, yy = trajs[i,:,0], trajs[i, :, 1]
            yi = ((yy - self.bx[0]) / self.dx[0]).long() # trajectory points in pixels
            xi = ((xx - self.bx[1]) / self.dx[1]).long() # trajectory points in pixels

            m1 = torch.logical_and(
                torch.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                torch.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            )
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll)) # timestep where gt box points are not collision and within the map

            ti = torch.arange(n_future, device=trajs.device)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long() # sum traj's collision points by timestep in batch, only sum collision points where gt box points are not collision

            m2 = torch.logical_not(gt_box_coll) # not_collision of gt box points
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i], token=None)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long() # sum box's collision points in timestep, only sum collision points where gt trajectory points are not collision
            # obj_box_coll_sum[ti] += (box_coll[ti]).long() 
        return obj_coll_sum, obj_box_coll_sum
   
    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask=None):
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''
        if gt_trajs_mask is None:
            gt_trajs_mask = torch.ones_like(gt_trajs)
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)) 

    def update(self, trajs, gt_trajs, segmentation, token=None, gt_trajs_mask=None):
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        segmentation: torch.Tensor (B, n_future, 200, 200)
        '''
        assert trajs.shape == gt_trajs.shape

        # trajs[..., 0] = - trajs[..., 0] # TODO comment this line
        # gt_trajs[..., 0] = - gt_trajs[..., 0]
        # segmentation = torch.flip(segmentation, [-1])     

        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], segmentation, token=token)

        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        self.total +=len(trajs)
        self.curr_obj_box_col = obj_box_coll_sum

    def compute(self):
        return {
            'obj_col': self.obj_col / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'L2' : self.L2 / self.total
        }