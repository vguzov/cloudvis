from dataclasses import dataclass
import dataclasses
import dacite
from argparse import ArgumentParser, Namespace
from pathlib import Path
import toml
from typing import Union, Optional, List, Dict, Tuple

@dataclass
class ScenesConfig:
    scenes_dir: Path
    pointclouds_dir: Path


@dataclass
class HumansConfig:
    smpl_motions_dir: Path
    smpl_shapes_dir: Path
    smpl_root: Path
    smpl_contacts_dir: Path
    smpl_template: Optional[Path] = None
    smpl_texture: Optional[Path] = None
    smpl_templates_dir: Optional[Path] = None
    no_smpl_template: bool = True
    load_with_height_corr: bool = False
    load_wo_tr: bool = False


@dataclass
class VideosConfig:
    videos_dir: Path
    kinrec_dir: Path
    kinects_dir: Path
    kinects_calibration_dir: Path


@dataclass
class ObjectsConfig:
    object_masks_dir: Path
    object_locations_dir: Path
    objloc_fps: float


@dataclass
class CapturingConfig:
    screen_capture_dir: Path
    trajectory_dir: Path
    take_screenshot: bool = False
    exit_on_videorec_stop: bool = False
    rectraj: Optional[Path] = None
    recvid: Optional[float] = None


@dataclass
class PreloadConfig:
    loadpath: Optional[str] = None
    nobkg: bool = False
    nosmpl: bool = False
    noobj: bool = False


@dataclass
class MiscConfig:
    continuous_redraw: bool = False
    obj_mask_override: Optional[Path] = None
    smpl_shape_override: Optional[Path] = None


@dataclass
class CloudvisConfig:
    resolution: List[int]
    viewport_resolution: List[int]
    fps_limit: float
    camera_fov: float
    scenes: ScenesConfig
    humans: HumansConfig
    videos: VideosConfig
    objects: ObjectsConfig
    capturing: CapturingConfig
    preload: PreloadConfig
    misc: MiscConfig


def config_to_dict(config: CloudvisConfig) -> dict:
    res_dict = {}
    config_dict = dataclasses.asdict(config)
    for param_name, param_val in config_dict.items():
        if isinstance(param_val, Path):
            param_val = str(param_val)
        res_dict[param_name] = param_val
    return res_dict


def dict_to_config(config_args_dict: dict) -> CloudvisConfig:
    for param_group in ["misc", "preload"]:
        if param_group not in config_args_dict:
            config_args_dict[param_group] = {}
    config = dacite.from_dict(data_class=CloudvisConfig, data=config_args_dict, config=dacite.Config(cast=[Path]))
    return config

def map_all_params(config, parent_key):
    mapdict = {}
    for config_key in dataclasses.asdict(config).keys():
        config_val = getattr(config, config_key)
        if dataclasses.is_dataclass(config_val):
            mapdict.update(map_all_params(config_val, parent_key + (config_key,)))
        else:
            mapdict[config_key] = parent_key
    return mapdict


def fuse_with_args(config: CloudvisConfig, args: Namespace) -> CloudvisConfig:
    key_mapping = map_all_params(config, ())
    for config_key, key_path in key_mapping.items():
        if hasattr(args, config_key) and getattr(args, config_key) is not None:
            target_dataclass = config
            for key in key_path:
                target_dataclass = getattr(target_dataclass, key)
            setattr(target_dataclass, config_key, getattr(args, config_key))
    return config


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # Static scene
    parser.add_argument("--scenes_dir", type=Path)
    parser.add_argument("--pointclouds_dir", type=Path)

    # Humans
    parser.add_argument("--smpl_poses_dir", type=Path)
    parser.add_argument("--smpl_shapes_dir", type=Path)
    parser.add_argument("--smpl_root", type=Path)
    parser.add_argument("--smpl_contacts_dir", type=Path)
    parser.add_argument("--smpl_template", type=Path, default=None)
    parser.add_argument("--smpl_texture", type=Path, default=None)
    parser.add_argument("--smpl_templates_dir", type=Path)
    parser.add_argument("--no_smpl_template", action="store_true", default=None)
    parser.add_argument("--load_with_height_corr", action="store_true", default=None)
    parser.add_argument("--load_wo_tr", action="store_true", default=None)

    # Third person videos (RGB and RGBD)
    parser.add_argument("--videos_dir", type=Path)
    parser.add_argument("--kinrec_dir", type=Path)
    # Legacy kinects
    parser.add_argument("--kinects_dir", type=Path)
    parser.add_argument("--kinects_calibration_dir", type=Path)

    # Object motion
    parser.add_argument("--object_masks_dir", type=Path)
    parser.add_argument("--object_locations_dir", type=Path)
    parser.add_argument("--objloc_fps", type=float)

    # Screenshots and video recording
    parser.add_argument("--screen_capture_dir", type=Path)
    parser.add_argument("--trajectory_dir", type=Path)
    parser.add_argument("--rectraj", type=Path, default=None)
    parser.add_argument("--recvid", type=float, default=None)
    parser.add_argument("--take_screenshot", action="store_true", default=None)
    parser.add_argument("-evs", "--exit_on_videorec_stop", action="store_true", default=None)
    parser.add_argument("-fov", "--camera_fov", type=float)

    # Preload options
    parser.add_argument("-l", "--loadpath", default=None)
    parser.add_argument("--nobkg", action="store_true", default=None)
    parser.add_argument("--nosmpl", action="store_true", default=None)
    parser.add_argument("--noobj", action="store_true", default=None)

    # Misc
    parser.add_argument("--continuous_redraw", action="store_true", default=None)
    parser.add_argument("--obj_mask_override", default=None)
    parser.add_argument("--smpl_shape_override", default=None)

    parser.add_argument("-c", "--config", type=Path, default="./configs/default.toml")

    return parser


def get_config():
    parser = create_parser()
    args = parser.parse_args()
    config_dict = toml.load(open(args.config))
    config = dict_to_config(config_dict)
    config = fuse_with_args(config, args)
    return config

