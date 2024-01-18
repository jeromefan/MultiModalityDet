import os
import mmcv
import torch
import argparse
import mmengine
import warnings
import numpy as np
import os.path as osp
from tqdm import tqdm
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from torchvision.transforms import Compose, Normalize, ToPILImage, Pad, Resize
from met_multi_modality_visualizer import MetadverseMultiModalityVisualizer

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Use PGD to attack pc.")
    argparser.add_argument(
        "--config",
        # default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
        default="configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py",
        type=str,
    )
    argparser.add_argument(
        "--checkpoint",
        # default="checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.pth",
        default="checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth",
        type=str,
    )
    argparser.add_argument(
        "--point-eps-m", default=0.5, type=float, help="0, 0.1, 0.2, 0.5"
    )
    argparser.add_argument("--max-step", default=10, type=int, help="PGD10")
    argparser.add_argument("--pred-score-thr", default=0.3, type=float)
    argparser.add_argument("--work-dir")
    args = argparser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.log_level = "ERROR"
    cfg.load_from = args.checkpoint
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    image_aug_settings = None
    for settings in cfg.test_dataloader.dataset.pipeline:
        if settings["type"] == "ImageAug3D":
            image_aug_settings = settings

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    dataloader = runner.val_dataloader
    model = runner.model
    val_evaluator = runner.val_evaluator
    test_evaluator = runner.test_evaluator
    val_evaluator.dataset_meta["version"] = "v1.0-mini"
    for i in range(len(val_evaluator.metrics)):
        val_evaluator.metrics[i].jsonfile_prefix = f"{runner.log_dir}/clean"
    test_evaluator.dataset_meta["version"] = "v1.0-mini"
    for i in range(len(test_evaluator.metrics)):
        test_evaluator.metrics[i].jsonfile_prefix = f"{runner.log_dir}/adversarial"

    model.eval()

    save_path = f"{runner.log_dir}/vis_data"
    os.makedirs(save_path, exist_ok=True)
    point_eps_m = args.point_eps_m
    point_lr = point_eps_m / (args.max_step - 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visualizer = MetadverseMultiModalityVisualizer()
    iter = tqdm(dataloader)
    for i, ori_data in enumerate(iter):
        if ori_data["data_samples"][0].eval_ann_info["gt_labels_3d"].shape[0] == 0:
            print(f"第{i}组数据没有ground truth, 跳过攻击")
            continue

        orig_pts_input = ori_data["inputs"].pop("points", None)[0]
        # PGD random start
        pts_noisy = orig_pts_input.clone().detach()
        delta = torch.rand_like(orig_pts_input) * 2 * point_eps_m - point_eps_m
        pts_noisy = orig_pts_input + delta
        pts_noisy[:, 3:] = orig_pts_input[:, 3:]

        for step in range(args.max_step):
            iter.set_description(f"step={step}")
            pts_noisy = pts_noisy.clone().detach().requires_grad_(True)
            ori_data["inputs"]["points"] = [pts_noisy]

            data = model.data_preprocessor(ori_data, False)
            gt_instances_3d = InstanceData()
            eval_ann_info = data["data_samples"][0].eval_ann_info
            bboxes_3d = eval_ann_info["gt_bboxes_3d"]
            labels_3d = torch.LongTensor(eval_ann_info["gt_labels_3d"])
            bboxes_3d = bboxes_3d.to(device)
            labels_3d = labels_3d.to(device)
            gt_instances_3d.bboxes_3d = bboxes_3d
            gt_instances_3d.labels_3d = labels_3d
            data["data_samples"][0].gt_instances_3d = gt_instances_3d
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

        with torch.no_grad():
            ori_data["inputs"]["points"] = [pts_noisy]
            data = model.data_preprocessor(ori_data, False)
            visualizer.cache_points = (
                pts_noisy.clone().detach().cpu().numpy().reshape(-1, 5)[:, :3]
            )
            output_pred = model(mode="predict", **data)
            visualizer.vis_pc(
                name=f"{i}_adversarial_example",
                save_path=save_path,
                pred_result=output_pred[0],
                pred_score_thr=args.pred_score_thr,
            )
            test_evaluator.process(data_samples=output_pred, data_batch=data)
    clean_metrics = val_evaluator.evaluate(len(dataloader.dataset))
    adversarial_metrics = test_evaluator.evaluate(len(dataloader.dataset))
    mmengine.dump(
        dict(clean_metrics=clean_metrics, adversarial_metrics=adversarial_metrics),
        f"{runner.log_dir}/metrics.pkl",
    )
