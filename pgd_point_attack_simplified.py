import torch
import argparse
import warnings
import os.path as osp

from tqdm import tqdm
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.structures import InstanceData

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Use PGD to attack pc.")
    argparser.add_argument(
        "--config",
        default="mmdet3d_projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
        # default="configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py",
        type=str,
    )
    argparser.add_argument(
        "--checkpoint",
        default="checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.pth",
        # default="checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth",
        type=str,
    )
    argparser.add_argument(
        "--point-eps-m", default=0.5, type=float, help="0, 0.1, 0.2, 0.5"
    )
    argparser.add_argument("--max-step", default=10, type=int, help="PGD10")
    args = argparser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.log_level = "ERROR"
    cfg.load_from = args.checkpoint
    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    dataloader = runner.val_dataloader
    model = runner.model

    model.eval()

    point_eps_m = args.point_eps_m
    point_lr = point_eps_m / (args.max_step - 2)

    iter = tqdm(dataloader)
    for i, data in enumerate(iter):
        # 数据预处理
        data = model.data_preprocessor(data, False)

        gt_instances_3d = InstanceData()
        eval_ann_info = data["data_samples"][0].eval_ann_info
        bboxes_3d = eval_ann_info["gt_bboxes_3d"]
        labels_3d = torch.LongTensor(eval_ann_info["gt_labels_3d"])
        bboxes_3d = bboxes_3d.to("cuda")
        labels_3d = labels_3d.to("cuda")
        gt_instances_3d.bboxes_3d = bboxes_3d
        gt_instances_3d.labels_3d = labels_3d
        data["data_samples"][0].gt_instances_3d = gt_instances_3d

        # TODO: 对于非bevfusion来说，此处取出的不止是points，还有在前述data_preprocessor过程中完成体素化的体素

        orig_pts_input = data["inputs"].pop("points", None)[0]

        # PGD random start
        pts_noisy = orig_pts_input.clone().detach()
        delta = torch.rand_like(orig_pts_input) * 2 * point_eps_m - point_eps_m
        pts_noisy = orig_pts_input + delta
        pts_noisy[:, 3:] = orig_pts_input[:, 3:]

        # 多轮迭代攻击
        for step in range(args.max_step):
            iter.set_description(f"step={step}")
            pts_noisy = pts_noisy.clone().detach().requires_grad_(True)
            data["inputs"]["points"] = [pts_noisy]

            output_loss = model(mode="loss", **data)
            loss = 0
            for key in output_loss:
                if "loss" in key:
                    loss = loss + output_loss[key]

            advloss = -loss
            advloss.backward()
            pts_noisy_grad = pts_noisy.grad.detach()
            pts_noisy = pts_noisy - point_lr * pts_noisy_grad.sign()
            diff = pts_noisy - orig_pts_input
            diff = torch.clamp(diff, -point_eps_m, point_eps_m)
            pts_noisy = diff + orig_pts_input
            pts_noisy[:, 3:] = orig_pts_input[:, 3:]
