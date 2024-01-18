# todo: save_path 需要创建文件夹的功能

import math
import torch
import numpy as np
import open3d as o3d
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from typing import List, Optional, Sequence, Tuple, Union

from mmengine.dist import master_only
from mmengine.visualization.utils import (
    check_type,
    color_val_matplotlib,
    tensor2ndarray,
)
import mmcv
from mmdet3d.structures import (
    BaseInstance3DBoxes,
    Box3DMode,
    Coord3DMode,
    DepthInstance3DBoxes,
    Det3DDataSample,
    LiDARInstance3DBoxes,
)
from mmdet3d.visualization import (
    Det3DLocalVisualizer,
    to_depth_mode,
    proj_lidar_bbox3d_to_img,
)


class MetadverseMultiModalityVisualizer(Det3DLocalVisualizer):
    def __init__(
        self,
        points: Optional[np.ndarray] = None,
        imgs: Optional[list] = None,
        name: str = "visualizer",
    ) -> None:
        super().__init__(
            name=name,
            image=None,
            vis_backends=[{"type": "LocalVisBackend"}],
            save_dir="",
            bbox_color=None,
            text_color=(200, 200, 200),
            mask_color=None,
            line_width=3,
            alpha=0.8,
        )
        if points is not None:
            check_type("points", points, np.ndarray)
            self.cache_points = points

        if imgs is not None:
            check_type("imgs", imgs, list)
            self.cache_imgs = imgs

    @master_only
    def set_points(
        self,
        points: np.ndarray,
        frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
        points_color: Tuple[float] = (0.5, 0.5, 0.5),
        points_size: int = 2,
    ) -> None:
        if points.shape[1] != 3:
            assert "请保证输出的点云仅xyz三列"
        points = Coord3DMode.convert(points, 0, Coord3DMode.DEPTH)
        # PointCloud对象是Open3D中用于表示点云的类
        pcd = o3d.geometry.PointCloud()
        # 将numpy数组转换为Open3D中的Vector3dVector对象，并将其设置为点云的点。
        pcd.points = o3d.utility.Vector3dVector(points)
        # 获取渲染选项
        render_option = self.o3d_vis.get_render_option()
        if render_option is not None:
            # 设置点的大小
            render_option.point_size = points_size
            # 设置背景颜色为黑色
            render_option.background_color = np.asarray([0, 0, 0])

        # 创建一个坐标框架网格，坐标框架将以原点为中心
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(**frame_cfg)
        self.o3d_vis.add_geometry(mesh_frame)

        # 创建一个与点云数据同样大小的颜色数组并设置点云的颜色
        points_colors = np.tile(np.array(points_color), (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        self.o3d_vis.add_geometry(pcd)
        self.pcd = pcd
        self.points_colors = points_colors

    @master_only
    def draw_bboxes_3d(
        self,
        bboxes_3d: BaseInstance3DBoxes,
        bbox_colors: Union[Tuple[float], List[Tuple[float]]] = (0, 1, 0),
        rot_axis: int = 2,
        center_mode: str = "lidar_bottom",
    ) -> None:
        # Before visualizing the 3D Boxes in point cloud scene
        # we need to convert the boxes to Depth mode
        check_type("bboxes", bboxes_3d, BaseInstance3DBoxes)
        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            bboxes_3d = bboxes_3d.convert_to(Box3DMode.DEPTH)
        # convert bboxes to numpy dtype
        bboxes_3d = tensor2ndarray(bboxes_3d.tensor)

        if isinstance(bbox_colors, list):
            assert len(bbox_colors) == len(
                bboxes_3d
            ), "specify color for every bounding box"

        for i in range(len(bboxes_3d)):
            if isinstance(bbox_colors, tuple):
                bbox_color = bbox_colors
            else:
                bbox_color = bbox_colors[i]

            center = bboxes_3d[i, 0:3]
            dim = bboxes_3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = bboxes_3d[i, 6]
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)

            if center_mode == "lidar_bottom":
                # bottom center to gravity center
                center[rot_axis] += dim[rot_axis] / 2
            elif center_mode == "camera_bottom":
                # bottom center to gravity center
                center[rot_axis] -= dim[rot_axis] / 2
            box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)

            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
            line_set.paint_uniform_color(np.array(bbox_color, dtype=np.float64))
            # draw bboxes on visualizer
            self.o3d_vis.add_geometry(line_set)

            # 改变在框内的点的颜色
            if self.pcd is not None:
                indices = box3d.get_point_indices_within_bounding_box(self.pcd.points)
                self.points_colors[indices] = np.array(bbox_color, dtype=np.float64)

        # 更新点的颜色
        if self.pcd is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(self.points_colors)
            self.o3d_vis.update_geometry(self.pcd)

    @master_only
    def draw_proj_bboxes_3d(
        self,
        bboxes_3d: BaseInstance3DBoxes,
        input_meta: dict,
        edge_colors: Union[str, Tuple[int], List[Union[str, Tuple[int]]]] = "royalblue",
        line_styles: Union[str, List[str]] = "-",
        line_widths: Union[int, float, List[Union[int, float]]] = 2,
        alpha: Union[int, float] = 0.4,
        img_size: Optional[Tuple] = None,
    ):
        check_type("bboxes", bboxes_3d, BaseInstance3DBoxes)

        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            proj_bbox3d_to_img = proj_lidar_bbox3d_to_img
        else:
            assert "这里别删！"

        # 将edge_colors转换为matplotlib可识别的颜色格式
        edge_colors_norm = color_val_matplotlib(edge_colors)

        # 将3D边界框投影到图像上
        corners_2d = proj_bbox3d_to_img(bboxes_3d, input_meta)
        if img_size is not None:
            # 过滤掉一半在图像外部的边界框，这是为了多视图图像的可视化
            valid_point_idx = (
                (corners_2d[..., 0] >= 0)
                & (corners_2d[..., 0] <= img_size[0])
                & (corners_2d[..., 1] >= 0)
                & (corners_2d[..., 1] <= img_size[1])
            )
            # 只保留至少有4个点在图像内的边界框
            valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 4
            corners_2d = corners_2d[valid_bbox_idx]
            # 过滤边缘颜色
            filter_edge_colors = []
            filter_edge_colors_norm = []
            for i, color in enumerate(edge_colors):
                if valid_bbox_idx[i]:
                    filter_edge_colors.append(color)
                    filter_edge_colors_norm.append(edge_colors_norm[i])
            edge_colors = filter_edge_colors
            edge_colors_norm = filter_edge_colors_norm

        lines_verts_idx = [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 5, 1, 2, 6]
        lines_verts = corners_2d[:, lines_verts_idx, :]
        front_polys = corners_2d[:, 4:, :]
        codes = [Path.LINETO] * lines_verts.shape[1]
        codes[0] = Path.MOVETO
        pathpatches = []
        for i in range(len(corners_2d)):
            verts = lines_verts[i]
            pth = Path(verts, codes)
            pathpatches.append(PathPatch(pth))

        p = PatchCollection(
            pathpatches,
            facecolors="none",
            edgecolors=edge_colors_norm,
            linewidths=line_widths,
            linestyles=line_styles,
        )

        self.ax_save.add_collection(p)

        # draw a mask on the front of project bboxes
        front_polys = [front_poly for front_poly in front_polys]
        return self.draw_polygons(
            front_polys,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=edge_colors,
        )

    @master_only
    def vis_predresult(
        self,
        name: str,
        pred_result: Det3DDataSample,
        pred_score_thr: float = 0.3,
        save_path: Optional[str] = ".",
        multi_imgs_col: int = 3,
    ) -> None:
        # 创建一个Open3D可视化对象
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        self.o3d_vis = o3d.visualization.Visualizer()
        self.o3d_vis.create_window(window_name=name, width=1440, height=810)
        # 如果预测结果中没有3D实例，即没有检测结果，则抛出异常
        if "pred_instances_3d" not in pred_result:
            assert "无检测结果，请降低 pred_score_thr 值"
        # 获取预测结果并仅保留分数大于阈值的部分
        pred_instances_3d = pred_result.pred_instances_3d
        pred_instances_3d = pred_instances_3d[
            pred_instances_3d.scores_3d > pred_score_thr
        ].to("cpu")
        # 如果预测实例的数量为0，双重保险，但是应该没必要
        if not len(pred_instances_3d) > 0:
            assert "无检测结果，请降低 pred_score_thr 值"
        bboxes_3d = pred_instances_3d.bboxes_3d
        labels_3d = pred_instances_3d.labels_3d

        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            _, bboxes_3d_depth = to_depth_mode(self.cache_points, bboxes_3d)
        else:
            bboxes_3d_depth = bboxes_3d.clone()

        # bbox颜色 生成随机的调色板
        max_label = int(max(labels_3d) if len(labels_3d) > 0 else 0)
        state = np.random.get_state()
        np.random.seed(42)
        palettes = np.random.randint(0, 256, size=(max_label + 1, 3))
        np.random.set_state(state)

        # 点云
        self.set_points(points=self.cache_points)
        # 点云bbox颜色
        bbox_palette = [tuple(c / 255) for c in palettes]  # 将调色板的颜色值归一化
        colors = [bbox_palette[label] for label in labels_3d]  # 根据标签获取颜色
        # 绘制点云bbox
        self.draw_bboxes_3d(bboxes_3d_depth, bbox_colors=colors)

        # 相机
        img_size = self.cache_imgs[0].shape[:2]  # 获取图像的大小
        img_col = multi_imgs_col  # 设置图像的列数
        img_row = math.ceil(len(self.cache_imgs) / img_col)  # 计算图像的行数
        # 创建一个空的多图拼接图像
        composed_img = np.zeros(
            (img_size[0] * img_row, img_size[1] * img_col, 3), dtype=np.uint8
        )
        for i, single_img in enumerate(self.cache_imgs):
            self.set_image(single_img)
            single_img_meta = dict()
            for key, meta in pred_result.metainfo.items():
                if isinstance(meta, (Sequence, np.ndarray, torch.Tensor)) and len(
                    meta
                ) == len(self.cache_imgs):
                    single_img_meta[key] = meta[i]
                else:
                    single_img_meta[key] = meta
            # 相机bbox颜色
            bbox_palette = [tuple(c) for c in palettes]
            colors = [bbox_palette[label] for label in labels_3d]  # 根据标签获取颜色
            # 绘制相机bbox
            self.draw_proj_bboxes_3d(
                bboxes_3d,
                single_img_meta,
                img_size=single_img.shape[:2][::-1],
                edge_colors=colors,
            )
            # 将图像添加到拼接的图像中
            composed_img[
                (i // img_col) * img_size[0] : (i // img_col + 1) * img_size[0],
                (i % img_col) * img_size[1] : (i % img_col + 1) * img_size[1],
            ] = self.get_image()
        for _ in range(5):
            self.o3d_vis.update_renderer()
        # 保存结果
        img = mmcv.imconvert(composed_img, "rgb", "bgr")
        mmcv.imwrite(img, f"{save_path}/{name}_cam.png")
        self.o3d_vis.capture_screen_image(
            f"{save_path}/{name}_lidar.png", do_render=True
        )
        self.o3d_vis.destroy_window()

    @master_only
    def vis_cam(
        self,
        name: str,
        pred_result: Optional[Det3DDataSample] = None,
        pred_score_thr: float = 0.3,
        save_path: Optional[str] = ".",
        multi_imgs_col: int = 3,
    ) -> None:
        if pred_result is not None:
            # 如果预测结果中没有3D实例，即没有检测结果，则抛出异常
            if "pred_instances_3d" not in pred_result:
                assert "无检测结果，请降低 pred_score_thr 值"
            # 获取预测结果并仅保留分数大于阈值的部分
            pred_instances_3d = pred_result.pred_instances_3d
            pred_instances_3d = pred_instances_3d[
                pred_instances_3d.scores_3d > pred_score_thr
            ].to("cpu")
            # 如果预测实例的数量为0，双重保险，但是应该没必要
            if not len(pred_instances_3d) > 0:
                assert "无检测结果，请降低 pred_score_thr 值"
            bboxes_3d = pred_instances_3d.bboxes_3d
            labels_3d = pred_instances_3d.labels_3d

            # bbox颜色 生成随机的调色板
            max_label = int(max(labels_3d) if len(labels_3d) > 0 else 0)
            state = np.random.get_state()
            np.random.seed(42)
            palettes = np.random.randint(0, 256, size=(max_label + 1, 3))
            np.random.set_state(state)

        # 相机
        img_size = self.cache_imgs[0].shape[:2]  # 获取图像的大小
        img_col = multi_imgs_col  # 设置图像的列数
        img_row = math.ceil(len(self.cache_imgs) / img_col)  # 计算图像的行数
        # 创建一个空的多图拼接图像
        composed_img = np.zeros(
            (img_size[0] * img_row, img_size[1] * img_col, 3), dtype=np.uint8
        )
        for i, single_img in enumerate(self.cache_imgs):
            self.set_image(single_img)
            if pred_result is not None:
                single_img_meta = dict()
                for key, meta in pred_result.metainfo.items():
                    if isinstance(meta, (Sequence, np.ndarray, torch.Tensor)) and len(
                        meta
                    ) == len(self.cache_imgs):
                        single_img_meta[key] = meta[i]
                    else:
                        single_img_meta[key] = meta
                # 相机bbox颜色
                bbox_palette = [tuple(c) for c in palettes]
                colors = [bbox_palette[label] for label in labels_3d]  # 根据标签获取颜色
                # 绘制相机bbox
                self.draw_proj_bboxes_3d(
                    bboxes_3d,
                    single_img_meta,
                    img_size=single_img.shape[:2][::-1],
                    edge_colors=colors,
                )
            # 将图像添加到拼接的图像中
            composed_img[
                (i // img_col) * img_size[0] : (i // img_col + 1) * img_size[0],
                (i % img_col) * img_size[1] : (i % img_col + 1) * img_size[1],
            ] = self.get_image()
        # 保存结果
        img = mmcv.imconvert(composed_img, "rgb", "bgr")
        mmcv.imwrite(img, f"{save_path}/{name}_cam.png")

    @master_only
    def vis_pc(
        self,
        name: str,
        pred_result: Det3DDataSample,
        pred_score_thr: float = 0.3,
        save_path: Optional[str] = ".",
        multi_imgs_col: int = 3,
    ) -> None:
        # 创建一个Open3D可视化对象
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        self.o3d_vis = o3d.visualization.Visualizer()
        self.o3d_vis.create_window(window_name=name, width=1440, height=810)
        # 如果预测结果中没有3D实例，即没有检测结果，则抛出异常
        if "pred_instances_3d" not in pred_result:
            assert "无检测结果，请降低 pred_score_thr 值"
        # 获取预测结果并仅保留分数大于阈值的部分
        pred_instances_3d = pred_result.pred_instances_3d
        pred_instances_3d = pred_instances_3d[
            pred_instances_3d.scores_3d > pred_score_thr
        ].to("cpu")
        # 如果预测实例的数量为0，双重保险，但是应该没必要
        if not len(pred_instances_3d) > 0:
            assert "无检测结果，请降低 pred_score_thr 值"
        bboxes_3d = pred_instances_3d.bboxes_3d
        labels_3d = pred_instances_3d.labels_3d

        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            _, bboxes_3d_depth = to_depth_mode(self.cache_points, bboxes_3d)
        else:
            bboxes_3d_depth = bboxes_3d.clone()

        # bbox颜色 生成随机的调色板
        max_label = int(max(labels_3d) if len(labels_3d) > 0 else 0)
        state = np.random.get_state()
        np.random.seed(42)
        palettes = np.random.randint(0, 256, size=(max_label + 1, 3))
        np.random.set_state(state)

        # 点云
        self.set_points(points=self.cache_points)
        # 点云bbox颜色
        bbox_palette = [tuple(c / 255) for c in palettes]  # 将调色板的颜色值归一化
        colors = [bbox_palette[label] for label in labels_3d]  # 根据标签获取颜色
        # 绘制点云bbox
        self.draw_bboxes_3d(bboxes_3d_depth, bbox_colors=colors)

        for _ in range(5):
            self.o3d_vis.update_renderer()
        # 保存结果
        self.o3d_vis.capture_screen_image(
            f"{save_path}/{name}_lidar.png", do_render=True
        )
        self.o3d_vis.destroy_window()
