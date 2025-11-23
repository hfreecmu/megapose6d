# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union, Dict

# Third Party
import numpy as np
from PIL import Image

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData, transform_to_list
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

import imageio.v2 as imageio
import cv2
import torch
import open3d
from scipy.spatial.transform import Rotation as R
from vine_prune.utils.io import read_json, write_json
from vine_prune.utils.general_utils import create_pose
from vine_prune.utils.paths import OBJECT_DIR
from vine_prune.utils.cloud import write_pcd

logger = get_logger(__name__)


def load_observation(
    image_path: Path,
    load_depth: bool = False,
    depth_path: Path = None,
) -> Tuple[np.ndarray, Union[None, np.ndarray]]:
    
    rgb = np.array(Image.open(image_path), dtype=np.uint8)

    depth = None
    if load_depth:
        depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000

    return rgb, depth


def load_observation_tensor(
    image_path: Path,
    K: np.array,
    load_depth: bool = False,
    depth_path: Path = None,
) -> ObservationTensor:
    rgb, depth = load_observation(image_path, load_depth, depth_path)
    observation = ObservationTensor.from_numpy(rgb, depth, K)
    return observation


def load_object_data(mask_dir, label_identifiers, filename) -> List[ObjectData]: 
    object_data = []
    for li in label_identifiers:
        mask_path = os.path.join(mask_dir, li, 'mask_obj', filename)

        mask_im = cv2.imread(mask_path, -1)
        seg_inds = np.argwhere(mask_im > 0)

        y0, x0 = seg_inds.min(axis=0).tolist()
        y1, x1 = seg_inds.max(axis=0).tolist()

        label, seg_key = li.split('_')
        seg_key = int(seg_key)

        entry = {
            'label': label,
            'bbox_modal': [x0, y0, x1, y1],
            "instance_id": seg_key,
        }

        object_data.append(entry)

    object_data = [ObjectData.from_json(d) for d in object_data]

    return object_data


def load_detections(
    mask_dir, label_identifiers, filename
) -> DetectionsType:
    input_object_data = load_object_data(mask_dir, label_identifiers, filename)
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections, input_object_data

def make_object_dataset(label_identifiers: str, data_dir: str) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "m"

    objects = set()
    for li in label_identifiers:
        object_name = li.split('_')[0]
        label = object_name

        if label in objects:
            continue

        objects.add(label)

        # object_dir = os.path.join(OBJECT_DIR, object_name)
        # mesh_path = os.path.join(object_dir, 'generated_meshes', 'obj_mesh.ply')
        mesh_path = os.path.join(data_dir, 'meshes', li, 'object_mesh_pca_pre_opt.obj')
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))

    rigid_object_dataset = RigidObjectDataset(rigid_objects)

    return rigid_object_dataset

def save_predictions(
    output_path: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    scores = pose_estimates.infos['pose_score']
    instance_ids = pose_estimates.infos['instance_id']
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, score=score, TWO=Transform(pose),
                   instance_id=ii) for label, score, pose, ii in zip(labels, scores, 
                                                                     poses, instance_ids)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])

    output_fn = Path(output_path)
    output_fn.write_text(object_data_json)
    #logger.info(f"Wrote predictions: {output_fn}")
    return object_data


def run_inference(
    data_dir: Path,
    model_name: str,
    target_ind: int,
    vis: bool,
) -> None:

    model_info = NAMED_MODELS[model_name]

    image_dir = os.path.join(data_dir, 'undistorted')
    depth_dir = os.path.join(data_dir, 'depth')

    filenames = []
    for filename in os.listdir(image_dir):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue

        if target_ind is not None:
            file_ind = int(filename.split('.')[0])
            if file_ind != target_ind:
                continue

        filenames.append(filename)
    filenames = sorted(filenames)

    K_path = os.path.join(data_dir, 'cam_K.txt')
    K = np.loadtxt(K_path)

    dims_path = os.path.join(data_dir, 'cam_dims.txt')
    dims = np.loadtxt(dims_path).astype(int).tolist()

    contact_info_path = os.path.join(data_dir, 'contact_info.json')
    contact_res = read_json(contact_info_path)
    contact_info = contact_res['contact_info']

    label_identifiers = list(contact_info.keys())

    mesh_dir = os.path.join(data_dir, 'meshes')
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)

    output_dir = os.path.join(mesh_dir, 'megapose')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger.info(f"Loading model {model_name}.")
    object_dataset = make_object_dataset(label_identifiers, data_dir)
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    if vis:
        renderer = Panda3dSceneRenderer(object_dataset)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]

    logger.info(f"Running inference.")
    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        depth_path = os.path.join(depth_dir, filename.replace('.jpg', '.png'))

        observation = load_observation_tensor(
            image_path, K, 
            # load_depth=False, 
            load_depth=model_info["requires_depth"],
            depth_path=depth_path
        ).cuda()

        mask_dir = os.path.join(data_dir, 'masks', 'objects')

        detections, object_data = load_detections(mask_dir, label_identifiers, filename.replace('.jpg', '.png'))
        detections = detections.cuda()

        output, _ = pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"],
            top_k=5
        )
        
        output_path = os.path.join(output_dir, filename.split('.')[0] + '_raw.json')
        object_data_out = save_predictions(output_path, output)

        instance_id_inds = {}

        for odo_ind, odo in enumerate(object_data_out):
            label = odo.label
            score = odo.score
            instance_id = odo.instance_id

            if not instance_id in instance_id_inds:
                instance_id_inds[instance_id] = [None, None]

            saved_odo_ind, instance_id_score = instance_id_inds[instance_id]

            if (saved_odo_ind is None or
                score > instance_id_score):
                instance_id_inds[instance_id] = [odo_ind, score]

        is_vis_max = [False] * len(object_data_out)
        vis_custom_labels = [None] * len(object_data_out)
        res_dict = {}
        
        for instance_id in instance_id_inds:
            odo_ind, _ = instance_id_inds[instance_id]

            odo = object_data_out[odo_ind]
            two = odo.TWO
            score = odo.score
            label = odo.label

            # already asserted but doing again for safety
            assert odo.instance_id == instance_id

            label_identifier = '_'.join([label, str(instance_id)])

            entry = {
                "label": label,
                "TWO": transform_to_list(two),
                "score": score,

                'seg_key': instance_id
            }

            res_dict[label_identifier] = entry

            vis_custom_labels[odo_ind] = label_identifier
            is_vis_max[odo_ind] = True

        res_path = os.path.join(output_dir, filename.split('.')[0] + '.json')
        write_json(res_path, res_dict, pretty=True)

        if vis:
            object_data_out_orig = object_data_out

            camera_data = {
                "K": K.tolist(),
                "resolution": dims
            }
            
            camera_data = CameraData.from_json(json.dumps(camera_data))
            camera_data.TWC = Transform(np.eye(4))

            subdir = os.path.join(output_dir, filename.split('.')[0])
            if not os.path.exists(subdir):
                os.mkdir(subdir)

            for ind in range(len(object_data_out_orig)):
                object_data_out = [object_data_out_orig[ind]]

                camera_data, object_data_out = convert_scene_observation_to_panda3d(camera_data, object_data_out)
                
                renderings = renderer.render_scene(
                    object_data_out,
                    [camera_data],
                    light_datas,
                    render_depth=False,
                    render_binary_mask=False,
                    render_normals=False,
                    copy_arrays=True,
                )[0]

                orig_im = imageio.imread(image_path)
                comb_im = np.hstack((orig_im, renderings.rgb))

                if is_vis_max[ind]:
                    vis_path = os.path.join(output_dir, f'{vis_custom_labels[ind]}_{filename.split(".")[0]}.jpg')
                    imageio.imwrite(vis_path, comb_im)

                vis_path = os.path.join(subdir, f'{ind}.jpg')
                imageio.imwrite(vis_path, comb_im)

        # process result here
        for label_identifier in res_dict:
            two = res_dict[label_identifier]['TWO']
            pca_pcd_path = os.path.join(mesh_dir, label_identifier, 'mesh_pca_scale_pre_opt.pcd')
            pca_pcd = open3d.io.read_point_cloud(pca_pcd_path)
            pca_points = np.array(pca_pcd.points)

            R_mega =  R.from_quat(two[0]).as_matrix()
            t_mega = two[1]

            obj_points = pca_points @ R_mega.T + t_mega
            pca_pcd.points = open3d.utility.Vector3dVector(obj_points)

            rot_pcd_path = os.path.join(mesh_dir, label_identifier, 'mesh_rot_pre_opt.pcd')
            write_pcd(rot_pcd_path, pca_pcd)

            M_obj = create_pose(R_mega, t_mega)
            rot_path = os.path.join(mesh_dir, label_identifier, 'object_pose_rot_pre_opt.txt')
            np.savetxt(rot_path, M_obj)

# def make_output_visualization(
#     example_dir: Path,
# ) -> None:

#     rgb, _, camera_data = load_observation(example_dir, load_depth=False)
#     camera_data.TWC = Transform(np.eye(4))
#     object_datas = load_object_data(example_dir / "outputs" / "object_data.json")
#     object_dataset = make_object_dataset(example_dir)

#     renderer = Panda3dSceneRenderer(object_dataset)

#     camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
#     light_datas = [
#         Panda3dLightData(
#             light_type="ambient",
#             color=((1.0, 1.0, 1.0, 1)),
#         ),
#     ]
#     renderings = renderer.render_scene(
#         object_datas,
#         [camera_data],
#         light_datas,
#         render_depth=False,
#         render_binary_mask=False,
#         render_normals=False,
#         copy_arrays=True,
#     )[0]

#     # plotter = BokehPlotter()

#     # fig_rgb = plotter.plot_image(rgb)
#     # fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
#     contour_overlay = make_contour_overlay(
#         rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
#     )["img"]
#     # fig_contour_overlay = plotter.plot_image(contour_overlay)
#     # fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
#     vis_dir = example_dir / "visualizations"
#     vis_dir.mkdir(exist_ok=True)

#     imageio.imwrite(vis_dir / "contour_overlay.png", contour_overlay)
#     imageio.imwrite(vis_dir / "render.png", renderings.rgb)
#     imageio.imwrite(vis_dir / "orig.png", rgb)

#     # export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
#     # export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
#     # export_png(fig_all, filename=vis_dir / "all_results.png")
#     logger.info(f"Wrote visualizations to {vis_dir}.")
    return


# def make_mesh_visualization(RigidObject) -> List[Image]:
#     return


# def make_scene_visualization(CameraData, List[ObjectData]) -> List[Image]:
#     return


# def run_inference(example_dir, use_depth: bool = False):
#     return


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    # parser.add_argument("--model", type=str, default="megapose-1.0-RGBD")
    # parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis-icp")
    parser.add_argument("--target_ind", type=int, default=None)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()

    with torch.inference_mode():
        run_inference(args.data_dir, args.model, args.target_ind, 
                      args.vis)
    #make_output_visualization(args.data_dir)
