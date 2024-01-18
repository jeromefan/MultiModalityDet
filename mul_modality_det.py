import os
import mmcv
import torch
import argparse
import mmengine
import warnings
import numpy as np
from copy import deepcopy
from mmdet3d.apis import init_model
from mmdet3d.structures import get_box_type
from mmengine.dataset import Compose, pseudo_collate
from met_multi_modality_visualizer import MetadverseMultiModalityVisualizer

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="CARLA Sensor")
    argparser.add_argument(
        "--foldername",
        default="carla",
        type=str,
    )
    argparser.add_argument(
        "--filename",
        default="0095",
        type=str,
    )
    argparser.add_argument(
        "--pred-score-thr",
        default=0.3,
        type=float,
    )
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
    args = argparser.parse_args()
    model = init_model(
        config=args.config,
        checkpoint=args.checkpoint,
        device="cuda:0",
    )

    cfg = model.cfg

    foldername = args.foldername
    filename = args.filename
    pcd_path = f"data/{foldername}/{filename}/{filename}_LIDAR_TOP.bin"
    img_path = f"data/{foldername}/{filename}/"
    calib_path = f"data/{foldername}/{filename}/{filename}.pkl"

    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    calib_info = mmengine.load(calib_path)
    for _, img_info in calib_info.items():
        img_info["img_path"] = os.path.join(img_path, img_info["img_path"])

    data = dict(
        lidar_points=dict(lidar_path=pcd_path),
        images=calib_info,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        timestamp=0,
    )
    data = test_pipeline(data)
    collate_data = pseudo_collate([data])
    with torch.no_grad():
        # collate_data = model.data_preprocessor([data], False)
        result = model.test_step(collate_data)[0]

    if "axis_align_matrix" in result.metainfo:
        assert "_draw_instances_3d函数中if axis_align_matrix in input_meta这一句不能删！"

    imgs = []
    for img_path in result.img_path:
        single_img = mmcv.imread(img_path)
        single_img = mmcv.imconvert(single_img, "bgr", "rgb")
        imgs.append(single_img)
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)[:, :3]

    visualizer = MetadverseMultiModalityVisualizer(points=points, imgs=imgs)
    visualizer.vis_predresult(
        name="vis", pred_result=result, pred_score_thr=args.pred_score_thr
    )
