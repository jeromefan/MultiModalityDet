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
    argparser = argparse.ArgumentParser(description="Use FGSM to attack img.")
    argparser.add_argument(
        "--config",
        default="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
        type=str,
    )
    argparser.add_argument(
        "--checkpoint",
        default="checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.pth",
        type=str,
    )
    argparser.add_argument("--eps255", default=8, type=float)
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

    mean = torch.Tensor([m / 255 for m in cfg.model.data_preprocessor.mean])
    std = torch.Tensor([s / 255 for s in cfg.model.data_preprocessor.std])
    de_mean = [-m / s for m, s in zip(mean, std)]
    de_std = [1 / s for s in std]

    image_aug_settings = None
    for settings in cfg.test_dataloader.dataset.pipeline:
        if settings["type"] == "ImageAug3D":
            image_aug_settings = settings

    H, W = 900, 1600  # todo 想办法不用写死的方式
    fH, fW = image_aug_settings["final_dim"]
    resize = np.mean(image_aug_settings["resize_lim"])
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(image_aug_settings["bot_pct_lim"])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = [crop_w, crop_h, newW - crop_w - fW, newH - crop_h - fH]
    de_trans = Compose(
        [
            Normalize(
                mean=de_mean,
                std=de_std,
            ),
            ToPILImage(),
            Pad(crop),
            Resize((H, W)),
        ]
    )

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
    eps01 = args.eps255 / 255.0
    lr = eps01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualizer = MetadverseMultiModalityVisualizer()
    iter = tqdm(dataloader)
    for i, data in enumerate(iter):
        if data["data_samples"][0].eval_ann_info["gt_labels_3d"].shape[0] == 0:
            print(f"第{i}组数据没有ground truth, 跳过攻击")
            continue
        imgs = []
        for img_path in data["data_samples"][0].img_path:
            single_img = mmcv.imread(img_path)
            single_img = mmcv.imconvert(single_img, "bgr", "rgb")
            imgs.append(single_img)
        visualizer.cache_imgs = imgs
        visualizer.vis_cam(
            name=f"{i}_ori",
            save_path=save_path,
            pred_result=None,
        )

        # 数据预处理
        data = model.data_preprocessor(data, False)

        gt_instances_3d = InstanceData()
        eval_ann_info = data["data_samples"][0].eval_ann_info
        bboxes_3d = eval_ann_info["gt_bboxes_3d"]
        labels_3d = torch.LongTensor(eval_ann_info["gt_labels_3d"])
        bboxes_3d = bboxes_3d.to(device)
        labels_3d = labels_3d.to(device)
        gt_instances_3d.bboxes_3d = bboxes_3d
        gt_instances_3d.labels_3d = labels_3d
        data["data_samples"][0].gt_instances_3d = gt_instances_3d

        # 攻击前检测
        with torch.no_grad():
            imgs = []
            for j in range(len(data["inputs"]["imgs"][0])):
                img = np.array(de_trans(data["inputs"]["imgs"][0][j]))
                img.dtype = np.uint8
                imgs.append(img)
            visualizer.cache_imgs = imgs
            output_pred = model(mode="predict", **data)
            visualizer.vis_cam(
                name=f"{i}_clean_example",
                save_path=save_path,
                pred_result=output_pred[0],
                pred_score_thr=args.pred_score_thr,
            )
            val_evaluator.process(data_samples=output_pred, data_batch=data)

        orig_imgs_input = data["inputs"].pop("imgs")
        imgs_noisy = orig_imgs_input.clone().detach()

        for step in range(1):
            iter.set_description(f"step={step}")
            imgs_noisy = imgs_noisy.clone().detach().requires_grad_(True)
            data["inputs"]["imgs"] = imgs_noisy

            output_loss = model(mode="loss", **data)
            loss = 0
            for key in output_loss:
                if "loss" in key:
                    loss = loss + output_loss[key]
            advloss = -loss
            advloss.backward()

            imgs_noisy_grad = imgs_noisy.grad.detach()
            imgs_noisy = imgs_noisy - lr * imgs_noisy_grad.sign()
            diff = imgs_noisy - orig_imgs_input
            diff = torch.clamp(diff, -eps01, eps01)
            imgs_noisy = torch.clamp(diff + orig_imgs_input, 0, 1)

        with torch.no_grad():
            data["inputs"]["imgs"] = imgs_noisy
            imgs = []
            for j in range(len(data["inputs"]["imgs"][0])):
                img = np.array(de_trans(data["inputs"]["imgs"][0][j]))
                img.dtype = np.uint8
                imgs.append(img)
            visualizer.cache_imgs = imgs
            output_pred = model(mode="predict", **data)
            visualizer.vis_cam(
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
