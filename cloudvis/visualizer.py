import pygame as pg
import imgui
from typing import List, Union, Optional

from .imgui_pygame import PygameRenderer
from .gui import GuiController, StatusTextLine, TrajectoryEditDialog
from . import __version__ as cloudvis_version

import OpenGL.GL as gl
import pickle as pkl

import os
import sys
import time
from loguru import logger
import trimesh
import json
import zipjson
from scipy.spatial.transform import Rotation
from pathlib import Path
from glob import glob
import numpy as np
import ctypes
from skimage.io import imsave, imread
from videoio import VideoWriter
from queue import SimpleQueue

import palettable

import cloudrender
from cloudrender.utils import list_zip, trimesh_load_from_zip, get_closest_ind_before, get_closest_ind_after, ObjectTrajectory, ObjectLocation
from cloudrender.scene import Scene
from cloudrender.render import SimplePointcloud, SimpleMesh, SimplePointcloudWithNormals, AvgcolorPointcloudWithNormals
from cloudrender.camera import PerspectiveCameraModel, OpenCVCameraModel
from cloudrender.utils import trimesh_load_from_zip
from cloudrender.render.smpl import SMPLXColoredModel, SMPLXTexturedModel, SMPLXModelBase
from cloudrender.render import DirectionalLight
from cloudrender.render.depthmap import DepthVideo
from cloudrender.render.rigid import RigidObject, RigidObjectSimpleMesh, RigidObjectSimplePointcloud

from .kinect import KinectMulticameraSystem, KinrecSequenceController
from .config import CloudvisConfig


class InteractiveViewer:
    CAMERA_MODES = ['free', 'relative', 'first_person', 'trajectory']
    TIME_MODES = ['constant', 'real']

    def __init__(self, config: CloudvisConfig):
        self.bkg_color = (1.0, 1.0, 1.0, 0.0)
        self.gui_resolution = np.asarray(config.resolution)
        self.viewport_resolution = np.asarray(config.viewport_resolution)
        self.fps_clock = pg.time.Clock()
        self.fps_limit = config.fps_limit
        self.sliding_fps_window_size = 10
        self.camera_mode = 'free'  # free, relative, first_person
        self.camera_modes_available = {"free"}
        self.camera_pos = np.zeros(3)
        self.camera_rot = Rotation.from_euler('x', 90, degrees=True)
        self.camera_yaw_pitch = np.zeros(2)
        self.camera_roll = 0.
        self.camera_relative_window_size = 5
        self.is_active = True
        self.mouse_tracking = False
        self.mouse_capture = False
        self.mouse_capture_pos = self.gui_resolution / 2
        self.mouse_sensitivity = 1 / 300.
        self.movement_speed = 3.0
        self.animation_active = False
        self.last_timediff = 0
        self.last_realtimediff = 0
        self.far = 100.
        self.config = config
        self.last_smpl_poses_dir = config.humans.smpl_motions_dir
        self.screen_capture_dir = config.capturing.screen_capture_dir
        self.smpl_root = config.humans.smpl_root
        self.needs_redraw = True
        self.contacts_db = {}
        self.draw_contacts_flag = False
        self.template_height_correction = True

        self.scene_path = None
        self.render_with_norms = False
        self.trajectory = None
        self.raw_trajectory = None
        self.kinect_prefix = None
        self.kinects_multicamera_info = None
        self.icp_pc_cache = None
        self.time_mode = "real"  # constant, real
        self.global_time = 0
        self.last_timestamp = time.time()
        self.global_time_step = 1 / 30.  # 1 / 30.
        self.time_scrolling_speed = 1 / 30.
        self.time_scrolling_last_change = 0
        self.screen_capture_prefix = 'screenshot'
        self.screen_video_prefix = 'video'
        self.video_end_time = None

        self.trajectory_folder_rel = "trajectories"
        self.screen_capture_active = False
        self.draw_shadows = True
        self.screen_capture_get_delay = 50
        self.screen_capture_queue = []
        self.screen_capture_mode = "video"  # video, screenshot
        self.screen_capture_serialize = True
        self.screen_capture_frames_count = 0
        self.screen_capture_video_fps = 1. / self.global_time_step
        self.global_frame_counter = 0
        self.video_writer = None
        self.require_complete_rendering = False
        self.loop_active = False
        self.obj_locations = None
        self.kinect_renderers: List[DepthVideo] = []
        self.template_path = config.humans.smpl_template
        self.model_texture_path = config.humans.smpl_texture
        self.time_scale = 1.

        self.smpl_type = "smplh_compat"

        self.initialize_graphics_pipeline()

        self.task_queue = SimpleQueue()
        self.scene = Scene()
        self.camera_fov = config.camera_fov

        self.camera = PerspectiveCameraModel()
        self.camera.init_intrinsics(self.viewport_resolution, fov=self.camera_fov, far=self.far)

        self.main_pc = None
        self.main_mesh = None
        self.main_renderable_pc: Optional[SimplePointcloud] = None
        self.main_renderable_mesh = None
        self.timed_renderables = []
        self.smpl_model_shadowmap = None
        self.gui_focused, self.gui_hovered = False, False
        self.renderable_smpl = None
        self.object_pc = None
        self.smpl_models = []
        self.objloc_path = None
        self.obj_frame_inds = None
        self.draw_pause = False
        self.main_pc_path = None
        self.last_obj_location = None
        self.draw_queue = []

        self.video_capture_start_time = None
        self.video_capture_end_time = None
        self.video_capture_save_description = True

        self._seqdraw_fbo_texture = False

        self.max_seq_time = 0

        if config.preload.loadpath is not None:
            preload_vid = False
            try:
                preload_arg = config.preload.loadpath
                if preload_arg.startswith("v"):
                    preload_arg = preload_arg[1:]
                    preload_vid = True
                val = int(preload_arg)
            except ValueError:
                self.preload_path = config.preload.loadpath
            else:
                if val < 0:
                    maxind = self.get_last_fileindex(self.screen_capture_dir,
                                                     (self.screen_video_prefix if preload_vid else self.screen_capture_prefix) + "_", "????",
                                                     ".png" if preload_vid else ".mp4")
                    val = maxind + 1 + val
                logger.info(f"'{config.preload.loadpath}' is supplied, will load screenshot {val:04d}")
                self.preload_path = os.path.join(self.screen_capture_dir,
                                                 f"{(self.screen_video_prefix if preload_vid else self.screen_capture_prefix)}_{val:04d}.json")
        else:
            self.preload_path = None

        self.scene_setup()

    def scene_setup(self):
        light = DirectionalLight(np.array([0., -1., -1.]), np.array([0.8, 0.8, 0.8]))
        self.directional_light = light
        self.smpl_model_shadowmap_offset = -light.direction * 3
        self.smpl_model_shadowmap = self.scene.add_dirlight_with_shadow(light=light, shadowmap_texsize=(1024, 1024),
                                                                        shadowmap_worldsize=(4., 4., 10.),
                                                                        shadowmap_center=np.zeros(3) + self.smpl_model_shadowmap_offset)
        self.init_contact_sphere()

    def create_seqdraw_fbo(self, width, height):
        binded_fb = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)
        self._seqdraw_cb, self._seqdraw_db = gl.glGenRenderbuffers(2)

        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._seqdraw_cb)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_RGBA,
            width, height
        )

        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._seqdraw_db)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT32,
            width, height
        )
        self._seqdraw_fb = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._seqdraw_fb)
        gl.glFramebufferRenderbuffer(
            gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_RENDERBUFFER, self._seqdraw_cb
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_DRAW_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER, self._seqdraw_db
        )
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, binded_fb)

    def create_seqdraw_fbo_texture(self, width, height):
        self._seqdraw_fbo_texture = True
        binded_fb = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)

        self._seqdraw_fb = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._seqdraw_fb)

        self._seqdraw_cb = gl.glGenTextures(1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._seqdraw_cb)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glFramebufferTexture(gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, self._seqdraw_cb, 0)

        self._seqdraw_db = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._seqdraw_db)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT32,
            width, height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_DRAW_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER, self._seqdraw_db
        )
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, binded_fb)

    def initialize_graphics_pipeline(self):
        pg.init()
        pg.display.set_caption(f'CloudVis imgui {cloudvis_version} - powered by cloudrender {cloudrender.__version__}')
        icon_surface = pg.image.load(os.path.join(os.path.dirname(__file__), "resources/icon.png"))
        pg.display.set_icon(icon_surface)
        image_size = window_size = self.gui_resolution

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 4)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 1)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

        pg.display.set_mode(window_size, pg.DOUBLEBUF | pg.OPENGL)  # | pg.RESIZABLE

        self.init_opengl(*image_size)

        self.gui_controller = GuiController(self, self.gui_resolution)

        binded_draw_fb = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)
        binded_read_fb = gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING)
        self._main_fb = {'draw': binded_draw_fb, "read": binded_read_fb}
        self.create_seqdraw_fbo_texture(*self.viewport_resolution)
        self.switch_to_mainfbo()

    def init_opengl(self, width, height):
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(*self.bkg_color)
        gl.glViewport(0, 0, width, height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDepthRange(0.0, 1.0)

    def find_closest_camera_in_traj(self, control=True):
        times = np.array([x['time'] for x in self.trajectory])
        times_diff = times - self.global_time
        times_mask = times_diff > 0
        if times_mask.sum() == 0:
            if control:
                self.animation_active = False
                self.switch_camera_mode("free")
                if self.screen_capture_active:
                    self.screen_capture_active = False
                    self.require_complete_rendering = False
                    self.time_mode = 'real'
            return self.trajectory[-1]
        times_inds = np.flatnonzero(times_mask)
        curr_ind = times_inds[0]
        return self.trajectory[curr_ind]

    def switch_camera_mode(self, mode):
        if mode == self.camera_mode:
            return
        self.camera_mode = mode
        if self.camera_mode == "first_person":
            self.smpl_models[0].visible = False
        if self.camera_mode == 'free':
            self.camera_yaw_pitch = (self.camera_rot * Rotation.from_euler('x',-90, degrees=True)).as_euler('xyz')[[2, 0]]
            self.control_camera_rot(np.zeros(2))
        elif self.camera_mode == 'relative':
            self.camera_relative_offset = None
            self.camera_relative_center = None
            self.camera_relative_window = 0
        self.static_update = True

    def load_multi_objloc(self, objloc_info_path):
        self.objloc_path = objloc_info_path
        self.objloc_type = "multi"
        if hasattr(self, "curr_dynamic_rigid_objects"):
            for renderable_objpc in self.curr_dynamic_rigid_objects:
                self.timed_renderables.remove(renderable_objpc)
                self.scene.objects.remove(renderable_objpc)
        objloc_info_path = Path(objloc_info_path)
        objloc_dir = os.path.dirname(objloc_info_path)
        objmask_dir = self.config.objects.object_masks_dir

        status_line = StatusTextLine(f"Loading objloc {os.path.basename(objloc_dir)}")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()

        self.config.objects.object_locations_dir = objloc_dir
        self.request_redraw()
        objloc_info = json.load(objloc_info_path.open())
        objects_dict = objloc_info["objects"]
        self.overall_object_mask = np.zeros(len(self.main_pc.vertices), dtype=bool)
        self.curr_dynamic_rigid_objects = []
        for ind, (objname, object_info) in enumerate(objects_dict.items()):
            status_line.text = f"Loading object {ind + 1}/{len(objects_dict)}"
            self.redraw_gui()
            objloc_relpath = object_info["relpath"]
            objloc_path = Path(objloc_dir) / objloc_relpath
            objmask_path = objmask_dir / f"{objname}.npz"
            objmask_npz = np.load(objmask_path)
            if "type" in objmask_npz:
                objmask_type = objmask_npz["type"]
                objmask_vistype = objmask_npz["visual_type"] if "visual_type" in objmask_npz else "pc"
            else:
                objmask_type = "internal"
                objmask_vistype = "pc"
            if "present" in object_info:
                obj_present = object_info["present"]
            else:
                obj_present = True
            if objmask_type == "internal":
                obj_vertex_mask_inds = objmask_npz['indices']
                obj_vertex_mask = np.zeros(len(self.main_pc.vertices), dtype=bool)
                obj_vertex_mask[obj_vertex_mask_inds] = True
                self.overall_object_mask[obj_vertex_mask_inds] = True
                obj_vertices = self.main_pc.vertices[obj_vertex_mask]
                obj_colors = self.main_pc.colors[obj_vertex_mask]
                obj_pc = trimesh.PointCloud(obj_vertices, colors=obj_colors)
            else:
                # External mask
                obj_data = trimesh.load(objmask_path.with_suffix(".ply"), process=False)
                if not isinstance(obj_data, trimesh.PointCloud):
                    obj_mesh = obj_data
                    obj_pc = trimesh.PointCloud(np.asarray(obj_data.vertices), colors=np.asarray(obj_data.visual.vertex_colors))
                else:
                    obj_pc = obj_data

            if obj_present:
                if objmask_vistype == 'pc':
                    renderable_obj = RigidObjectSimplePointcloud(self.camera, generate_shadows=False)
                    renderable_obj.init_context()
                    renderable_obj.set_buffers(obj_pc)
                elif objmask_vistype == 'mesh':
                    renderable_obj = RigidObjectSimpleMesh(self.camera, generate_shadows=False)
                    renderable_obj.init_context()
                    renderable_obj.set_buffers(obj_mesh)
                    renderable_obj.set_material(ambient=0.3, diffuse=0.5, specular=0.7)
                else:
                    raise ValueError(f"Unknown visual type {objmask_vistype}")
                objloc = np.load(objloc_path)
                renderable_obj.set_sequence(ObjectTrajectory(objloc["translations"], objloc["quaternions"], objloc["timestamps"]),
                                              times=objloc["timestamps"])
                renderable_obj.set_time(self.global_time)
                self.timed_renderables.append(renderable_obj)
                self.scene.add_object(renderable_obj)
                self.curr_dynamic_rigid_objects.append(renderable_obj)
        status_line.text = f"Reloading main PC"
        self.redraw_gui()
        rest_pc = trimesh.PointCloud(self.main_pc.vertices[~self.overall_object_mask], colors=self.main_pc.colors[~self.overall_object_mask])
        self.main_renderable_pc.set_buffers(rest_pc)
        self.gui_controller.status_text.remove(status_line)

    def load_objloc(self, objloc_path):
        self.config.objects.object_locations_dir = os.path.dirname(objloc_path)
        self.request_redraw()
        self.objloc_path = objloc_path
        self.objloc_type = "single"
        objloc_prefix = os.path.splitext(os.path.basename(objloc_path))[0]
        if self.config.misc.obj_mask_override is not None:
            objmask_path = self.config.misc.obj_mask_override
        else:
            objmask_prefix = "_".join(objloc_prefix.split(".")[0].split("_")[1:4])
            objmask_paths = [x for x in glob(os.path.join(self.config.objects.object_masks_dir, "*.npz"))
                             if os.path.basename(x).startswith(objmask_prefix)]
            if len(objmask_paths) > 1:
                logger.warning(f"More than 1 mask for {objloc_prefix}")
            objmask_path = objmask_paths[0]
            if 'kinect' in os.path.basename(self.main_pc_path) and os.path.exists(os.path.splitext(objmask_path)[0] + ".kinect.npz"):
                objmask_path = os.path.splitext(objmask_path)[0] + ".kinect.npz"
        obj_vertex_mask_inds = np.load(objmask_path)['indices']
        obj_vertex_mask = np.zeros(len(self.main_pc.vertices), dtype=bool)
        obj_vertex_mask[obj_vertex_mask_inds] = True
        with_norms = self.main_pc.normals is not None and self.render_with_norms
        obj_pc = SimplePointcloud.PointcloudContainer(vertices=self.main_pc.vertices[obj_vertex_mask], colors=self.main_pc.colors[obj_vertex_mask],
                                                      normals=self.main_pc.normals[obj_vertex_mask] if with_norms else None)
        rest_pc = SimplePointcloud.PointcloudContainer(self.main_pc.vertices[~obj_vertex_mask], colors=self.main_pc.colors[~obj_vertex_mask],
                                                       normals=self.main_pc.normals[~obj_vertex_mask] if with_norms else None)
        self.main_renderable_obj_pc.set_buffers(obj_pc)
        self.main_renderable_pc.set_buffers(rest_pc)
        self.rest_pc = rest_pc
        self.object_pc = obj_pc

        if (objloc_path_ext := os.path.splitext(objloc_path)[1]) == ".json":
            obj_locations = self.obj_locations = json.load(open(objloc_path))
            self.obj_frame_inds = sorted([int(x) for x in obj_locations.keys()])
            self.obj_locations_timestamps = np.array([float(x) / self.config.objects.objloc_fps for x in self.obj_frame_inds])
        elif objloc_path_ext == ".npz":
            obj_locations_npz = np.load(objloc_path)
            self.obj_locations_timestamps = obj_locations_npz["timestamps"]
            self.obj_frame_inds = np.arange(len(self.obj_locations_timestamps))
            obj_translations = obj_locations_npz["translations"]
            obj_quaternions = obj_locations_npz["quaternions"]
            self.obj_locations = {f"{frame_ind:06d}": {"position": tr, "quaternion": quat} for frame_ind, (tr, quat) in
                                  enumerate(zip(obj_translations, obj_quaternions))}
        else:
            raise NotImplementedError(f"Unrecognized object motion format: {objloc_path_ext}")

    def load_kinects(self, kinect_path: str):
        # if os.path.isfile(os.path.join(kinect_path, "metadata.json")):
        #     return self.load_kinrec_seq(kinect_path)
        self.request_redraw()
        kinect_pcs_count = 3

        kinect_bname = os.path.basename(kinect_path)
        kinect_name = os.path.splitext(kinect_bname)[0]
        kinect_prefix_name = ".".join(kinect_name.split('.')[:-2])
        kinect_prefix = os.path.join(os.path.dirname(kinect_path), kinect_prefix_name)
        self.kinect_prefix = kinect_prefix
        config_path = kinect_prefix + ".config.json"
        logger.info(f"Loading {kinect_prefix}")
        status_line = StatusTextLine(f"Reading kinect system config")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()
        if not os.path.isfile(config_path):
            config_path = os.path.join(os.path.dirname(kinect_prefix), "config.json")
            if not os.path.isfile(config_path):
                config_path = sorted(glob(os.path.join(os.path.dirname(kinect_prefix), "*.config.json")))
                if len(config_path) > 0:
                    config_path = config_path[0]
                else:
                    config_path = None
        self.kinects_multicamera_info = KinectMulticameraSystem(camera_count=kinect_pcs_count, config_path=config_path)
        if config_path is None:
            location_paths = [kinect_prefix + f".{kinect_ind}.location.json" \
                              for kinect_ind in range(kinect_pcs_count)]
            if all([os.path.isfile(x) for x in location_paths]):
                self.kinects_multicamera_info.apply_loc_files(location_paths)

        if len(self.kinect_renderers) == 0:
            self.kinect_renderers = [None for _ in range(kinect_pcs_count)]

        for kinect_ind in range(kinect_pcs_count):
            status_line.text = f"Loading kinect {kinect_ind}.."
            self.redraw_gui()
            colorpath = kinect_prefix + f".{kinect_ind}.depthcolor.mp4"
            if not os.path.isfile(colorpath):
                colorpath = None
            depthpath = kinect_prefix + f".{kinect_ind}.depth.mp4"
            timepath = kinect_prefix + f".{kinect_ind}.time.json"
            calibration_pc_path = os.path.join(self.config.videos.kinects_calibration_dir, f"{kinect_ind}/pointcloud_table.npy")
            pc_table = np.load(calibration_pc_path)
            time_offset = self.kinects_multicamera_info.get_time_offset()
            timestamps = np.array(json.load(open(timepath))["color"]) / 1e6 - time_offset

            prev_renderable_dvideo = self.kinect_renderers[kinect_ind]
            if prev_renderable_dvideo is not None:
                self.timed_renderables.remove(prev_renderable_dvideo)
                self.scene.objects.remove(prev_renderable_dvideo)
                self.kinect_renderers[kinect_ind] = None

            renderable_dvideo = DepthVideo(pc_table, camera=self.camera,
                                           color=np.array([255 if i == kinect_ind else 0 for i in range(3)] + [255], dtype=np.uint8))
            renderable_dvideo.draw_shadows = False
            renderable_dvideo.generate_shadows = False
            renderable_dvideo.init_context()
            renderable_dvideo.set_sequence(depthpath, colorpath, times=timestamps)
            renderable_dvideo.init_model_extrinsics(*self.kinects_multicamera_info.get_location(kinect_ind))
            renderable_dvideo.set_time(self.global_time)
            renderable_dvideo.set_splat_size(0.4)  # 0.2
            self.kinect_renderers[kinect_ind] = renderable_dvideo
            self.timed_renderables.append(renderable_dvideo)
            self.scene.add_object(renderable_dvideo)
        self.gui_controller.status_text.remove(status_line)

    def load_kinrec_seq(self, seqpath: str):
        color_palette = np.array(palettable.colorbrewer.qualitative.Accent_8.colors).astype(np.uint8)
        color_palette = np.hstack([color_palette, np.ones((color_palette.shape[0], 1), dtype=np.uint8) * 255])

        if os.path.isfile(seqpath):
            seqpath = os.path.dirname(seqpath)

        seqpath = Path(seqpath)
        seqname = os.path.splitext(os.path.basename(seqpath))[0]
        prefix_name = ".".join(seqname.split(".")[:-1])
        prefix_path = os.path.join(os.path.dirname(seqpath), prefix_name)
        self.kinect_prefix = str(seqpath)
        logger.info(f"Loading {prefix_name}")
        extrinsics_path = seqpath / "extrinsics.json"
        if not os.path.isfile(extrinsics_path):
            extrinsics_path = None

        depth_dir = seqpath / "depth"
        depthcolor_dir = seqpath / "depthcolor"
        times_dir = seqpath / "times"
        depth2pc_dir = seqpath / "depth2pc_maps"
        kinect_fnames = [os.path.splitext(os.path.basename(x))[0] for x in depth_dir.glob("*.mp4")]
        kinect_fnames_ids = [x.split("_")[1] for x in kinect_fnames]

        metadata_path = seqpath / "metadata.json"
        status_line = StatusTextLine(f"Reading kinect system config")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()
        self.kinects_multicamera_info = KinrecSequenceController(metadata_path, extrinsics_path)
        kinect_pcs_count = self.kinects_multicamera_info.camera_count
        if len(self.kinect_renderers) == 0:
            self.kinect_renderers = [None for _ in range(kinect_pcs_count)]
        kinect_ids = self.kinects_multicamera_info.participating_kinects
        kinect_fnames = [kinect_fnames[kinect_fnames_ids.index(x)] for x in kinect_ids]
        for ind, (kinect_id, kinect_fname) in enumerate(zip(kinect_ids, kinect_fnames)):
            status_line.text = f"Loading kinect {kinect_id} ({ind + 1}/{len(kinect_ids)})..."
            self.redraw_gui()
            depthcolorpath = depthcolor_dir / f"{kinect_fname}.mp4"
            if not os.path.isfile(depthcolorpath):
                depthcolorpath = None
            depthpath = depth_dir / f"{kinect_fname}.mp4"
            timepath = times_dir / f"{kinect_fname}.json"
            calibration_pc_path = depth2pc_dir / f"{kinect_fname}.npz"
            pc_table = np.load(calibration_pc_path)[kinect_id]
            time_offset = self.kinects_multicamera_info.get_time_offset()
            timestamps = np.array(json.load(open(timepath))["depth"]) / 1e6 - time_offset

            prev_renderable_dvideo = self.kinect_renderers[ind]
            if prev_renderable_dvideo is not None:
                self.timed_renderables.remove(prev_renderable_dvideo)
                self.scene.objects.remove(prev_renderable_dvideo)
                self.kinect_renderers[ind] = None

            renderable_dvideo = DepthVideo(pc_table, camera=self.camera,
                                           color=np.array(color_palette[ind], dtype=np.uint8))
            renderable_dvideo.draw_shadows = False
            renderable_dvideo.generate_shadows = False
            renderable_dvideo.init_context()
            renderable_dvideo.set_sequence(depthpath, depthcolorpath, times=timestamps)
            renderable_dvideo.init_model_extrinsics(*self.kinects_multicamera_info.get_location(kinect_id))
            renderable_dvideo.set_time(self.global_time)
            renderable_dvideo.set_splat_size(0.4)  # 0.2
            self.kinect_renderers[ind] = renderable_dvideo
            self.timed_renderables.append(renderable_dvideo)
            self.scene.add_object(renderable_dvideo)
        self.gui_controller.status_text.remove(status_line)

    def find_closest_interval_before(self, intervals):
        start_diff = self.global_time - intervals[:, 0]
        starts_happened_before = start_diff >= 0
        if np.count_nonzero(starts_happened_before) == 0:
            return None
        closest_start_ind = np.argmin(start_diff[starts_happened_before])
        return closest_start_ind

    def find_closest_interval_after(self, intervals):
        start_diff = self.global_time - intervals[:, 0]
        starts_happened_after = start_diff < 0
        if np.count_nonzero(starts_happened_after) == 0:
            return None
        closest_start_ind = np.argmax(start_diff[starts_happened_after]) + np.count_nonzero(~starts_happened_after)
        return closest_start_ind

    def get_contacts(self):
        return self.contacts_db

    def set_contacts(self, contacts_db, save=True):
        self.contacts_db = contacts_db
        if save:
            for seqname, seq_contacts_info in self.contacts_db.items():
                serializable_contacts_info = {surf_name: intervals.tolist() for surf_name, intervals in seq_contacts_info["contacts"].items()}
                serializable_info = {k: v if k != "contacts" else serializable_contacts_info for k, v in seq_contacts_info.items()}
                json.dump(serializable_info, open(os.path.join(self.config.humans.smpl_contacts_dir, seqname + ".json"), "w"), indent=1)

    def get_current_contacts(self):
        res = {}
        curr_time = self.global_time
        for seqname, seq_contacts_info in self.contacts_db.items():
            res[seqname] = {}
            if seq_contacts_info["type"] == "intervals":
                seq_contacts = seq_contacts_info["contacts"]
                for surface_name, intervals in seq_contacts.items():
                    if len(intervals) == 0:
                        res[seqname][surface_name] = False
                    else:
                        closest_start_ind = self.find_closest_interval_before(intervals)
                        if closest_start_ind is None:
                            res[seqname][surface_name] = False
                        else:
                            contact_state = ((intervals[closest_start_ind, 1] - intervals[closest_start_ind, 0]) < 0) or (
                                    intervals[closest_start_ind, 1] > curr_time)
                            res[seqname][surface_name] = contact_state
            elif seq_contacts_info["type"] == "sequential":
                # TODO: intergate "find_nearest_in_sorted" from utils
                seq_contacts = seq_contacts_info["sequence"]
                seq_timestamps = seq_contacts_info["timestamps"]
                closest_ts_ind = self.find_closest_interval_before(seq_timestamps[:, np.newaxis])
                for surface_name, seq_contacts_labels in seq_contacts.items():
                    if closest_ts_ind is None:
                        res[seqname][surface_name] = False
                    else:
                        res[seqname][surface_name] = seq_contacts_labels[closest_ts_ind]
            else:
                logger.error(f"Unknown contact DB type for {seqname}")
        return res

    def draw_contacts(self, override=False):
        if not self.draw_contacts_flag:
            return
        curr_contacts = self.get_current_contacts()
        for smpl_model in self.smpl_models:
            model_contacts = curr_contacts[smpl_model.sequence_prefix]
            joints = smpl_model.get_joints()
            hands_joints = joints[21:23]
            if model_contacts["left_hand"]:
                self.draw_sphere(hands_joints[1])
            if model_contacts["right_hand"]:
                self.draw_sphere(hands_joints[0])

    def draw_redblue_hands(self):
        for smpl_model in self.smpl_models:
            joints = smpl_model.get_joints()
            hands_joints = joints[21:23]
            self.draw_sphere(hands_joints[1], color=(0, 0, 255, 70))
            self.draw_sphere(hands_joints[0], color=(255, 0, 0, 70))

    def draw_contact_sphere(self, sequence_prefix, surface_name, color=(255, 165, 0, 70)):
        if surface_name in ["left_hand", "right_hand"]:
            for smpl_model in self.smpl_models:
                if sequence_prefix == smpl_model.sequence_prefix:
                    joints = smpl_model.get_joints()
                    hands_joints = joints[21:23]
                    if surface_name == "left_hand":
                        self.draw_sphere(hands_joints[1], color=color)
                    elif surface_name == "right_hand":
                        self.draw_sphere(hands_joints[0], color=color)

    def init_contact_sphere(self, radius=0.15, color=(0, 0, 0, 0)):
        mesh = trimesh.primitives.Sphere(radius=radius)
        colors = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 4), (mesh.vertices.shape[0], 1))
        mesh.visual.vertex_colors = colors
        renderable_mesh = SimpleMesh(camera=self.camera, draw_shadows=False, generate_shadows=False)
        renderable_mesh.init_context()
        renderable_mesh.set_buffers(mesh)
        self.renderable_sphere = renderable_mesh

    def draw_sphere(self, coords, color=(40, 210, 30, 70)):
        self.renderable_sphere.set_overlay_color(color)
        self.renderable_sphere.init_model_extrinsics(np.array([1, 0, 0, 0]), pose=np.asarray(coords))
        # gl.glDisable(gl.GL_CULL_FACE)
        self.renderable_sphere.draw()

    def replace_smpl_template(self, template_path, smpl_model: Union[SMPLXColoredModel, SMPLXTexturedModel], texture_path=None,
            height_correction_override=None):
        status_line = StatusTextLine(f"Loading template {os.path.basename(template_path)}")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()
        new_pickle = pkl.load(open(template_path, "rb"), encoding='latin1')
        template = new_pickle["v"]
        corr_vct = self.get_template_correction_vct(smpl_model.default_gender, smpl_model.default_betas, template, self.smpl_type,
                                                    height_correction_override=height_correction_override)
        smpl_model.set_body_template(template)
        smpl_model.set_global_offset(corr_vct)
        smpl_model.template_path = template_path
        if isinstance(smpl_model, SMPLXTexturedModel):
            if texture_path is None:
                template_name = os.path.basename(template_path)[:-len("_unposed.pkl")]
                texture_path = os.path.join(os.path.dirname(template_path), template_name + ".jpg")
                if not os.path.isfile(texture_path):
                    logger.warning(f"Texture {texture_path} does not exist, attempting to infer a path from parent folder name...")
                    template_name = os.path.basename(os.path.dirname(template_path))
                    texture_path = os.path.join(os.path.dirname(template_path), template_name + ".jpg")
            texture_data = imread(texture_path)
            tex_faces = new_pickle["ft"]
            uv_map = new_pickle["vt"][tex_faces.reshape(-1)]
            smpl_model.set_texture(texture_data, uv_map)
            smpl_model.texture_path = texture_path
        smpl_model.reload_current_frame()
        self.request_redraw()
        self.gui_controller.status_text.remove(status_line)

    def get_template_correction_vct(self, smpl_gender, betas, template, smpl_type, height_correction_override=None):
        height_correction_flag = self.template_height_correction if height_correction_override is None else height_correction_override
        if not height_correction_flag:
            return np.zeros(3)
        smpl_notemplate = SMPLXModelBase(camera=self.camera, gender=smpl_gender,
                                         smpl_root=self.smpl_root, template=None,  # global_offset=offset,
                                         model_type=smpl_type, center_root_joint=True,
                                         flat_hand_mean=True)
        smpl_with_template = SMPLXModelBase(camera=self.camera, gender=smpl_gender,
                                            smpl_root=self.smpl_root, template=template,  # global_offset=offset,
                                            model_type=smpl_type, center_root_joint=True,
                                            flat_hand_mean=True)
        verts_notemplate = smpl_notemplate.get_vertices(betas=betas, return_normals=False)
        verts_with_template = smpl_with_template.get_vertices(betas=betas, return_normals=False)
        min_height_verts_diff = verts_notemplate[:, 1].min() - verts_with_template[:, 1].min()
        # min_height_verts_diff -= 0.0
        corr_vct = np.array([0., 0., min_height_verts_diff])
        logger.info(f"Computed correction vct for template is {corr_vct}")
        return corr_vct

    def load_smpl_model(self, poses_path, prev_renderable_smpl=None, keep_template=True, height_correction_override=None):
        smpl_type = self.smpl_type
        self.config.humans.smpl_motions_dir = os.path.dirname(poses_path)
        self.request_redraw()
        # color_palette = (np.array(((0.8,0.2,0.2,1), (0.2,0.8,0.2,1)))*255).astype(np.uint8)
        color_palette = np.array(palettable.colorbrewer.qualitative.Accent_8.colors).astype(np.uint8)
        color_palette = np.hstack([color_palette, np.ones((color_palette.shape[0], 1), dtype=np.uint8) * 255])
        poses_bname = os.path.basename(poses_path)
        seq_prefix, poses_path_ext = os.path.splitext(poses_bname)
        if poses_path_ext == ".zip" and seq_prefix.endswith(".json"):
            seq_prefix = seq_prefix[:-len(".json")]
            poses_path_ext = ".json.zip"
        sub_prefix = seq_prefix.split("_")[0]
        if self.config.misc.smpl_shape_override is None:
            betas_path = os.path.join(self.config.humans.smpl_shapes_dir, sub_prefix + ".json")
        else:
            betas_path = self.config.misc.smpl_shape_override
        contacts_path = os.path.join(self.config.humans.smpl_contacts_dir, seq_prefix + ".json")
        if os.path.isfile(contacts_path):
            self.contacts_db[seq_prefix] = json.load(open(contacts_path))
            for surface_name in self.contacts_db[seq_prefix]["contacts"]:
                arr = np.asarray(self.contacts_db[seq_prefix]["contacts"][surface_name])
                if len(arr) == 0:
                    arr = np.zeros((0, 2))
                self.contacts_db[seq_prefix]["contacts"][surface_name] = arr
            self.contacts_db[seq_prefix]["type"] = "intervals"
        elif os.path.isfile(contacts_path := os.path.join(self.config.humans.smpl_contacts_dir, seq_prefix + ".json.zip")):
            logger.info(f"Loading 'sequential' style contacts for {seq_prefix}")
            self.contacts_db[seq_prefix] = zipjson.load(open(contacts_path, "rb"))
            for surface_name in self.contacts_db[seq_prefix]["sequence"]:
                arr = np.asarray(self.contacts_db[seq_prefix]["sequence"][surface_name])
                self.contacts_db[seq_prefix]["sequence"][surface_name] = arr
            self.contacts_db[seq_prefix]["timestamps"] = np.asarray(self.contacts_db[seq_prefix]["timestamps"])
        else:
            self.contacts_db[seq_prefix] = {"contacts": {"left_hand": np.zeros((0, 2)), "right_hand": np.zeros((0, 2))}}
            self.contacts_db[seq_prefix]["type"] = "intervals"

        status_line = StatusTextLine(f"Loading sequence {seq_prefix}")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()

        smpl_shape_params = json.load(open(betas_path))
        smpl_gender = smpl_shape_params['gender']
        smpl_betas = np.array(smpl_shape_params["betas"])

        template_path = self.template_path
        model_texture_path = self.model_texture_path

        if keep_template and prev_renderable_smpl is not None:
            if hasattr(prev_renderable_smpl, "template_path"):
                template_path = prev_renderable_smpl.template_path
            if hasattr(prev_renderable_smpl, "texture_path"):
                model_texture_path = prev_renderable_smpl.texture_path

        if self.config.humans.no_smpl_template:
            template_path = None
            model_texture_path = None

        if template_path is not None:
            new_pickle = pkl.load(open(template_path, "rb"), encoding='latin1')
            template = new_pickle["v"]
            corr_vct = self.get_template_correction_vct(smpl_gender, smpl_betas, template, smpl_type,
                                                        height_correction_override=height_correction_override)
        else:
            template = None
            corr_vct = np.zeros(3)

        if model_texture_path is not None:
            texture_data = imread(model_texture_path)
            tex_faces = new_pickle["ft"]
            uv_map = new_pickle["vt"][tex_faces.reshape(-1)]
        else:
            texture_data = None
            uv_map = None

        if texture_data is None:
            model_class = SMPLXColoredModel
        else:
            model_class = SMPLXTexturedModel

        flat_hand_mean = False
        if poses_path_ext in [".json", ".json.zip"]:
            if poses_path_ext == ".json.zip":
                fbx_motion_seq = zipjson.load(open(poses_path, "rb"))
            else:
                fbx_motion_seq = json.load(open(poses_path))
            if isinstance(fbx_motion_seq, list):
                old_style_fbx_seq = True
                fbx_timestamps = np.array([x['time'] for x in fbx_motion_seq])
                fbx_motion_seq = [{k: np.array(v) for k, v in x.items()} for x in fbx_motion_seq]
            else:
                old_style_fbx_seq = False
                betas = np.array(fbx_motion_seq["global"]["betas"])
                gender = fbx_motion_seq["global"]["gender"]
                fbx_motion_seq = [{k: np.array(v) for k, v in x.items()} for x in fbx_motion_seq["sequence"]]
                fbx_timestamps = np.array([x['time'] for x in fbx_motion_seq])
                flat_hand_mean = "right_hand_pose" in fbx_motion_seq[0]
                logger.debug(f"flat_hand_mean is {flat_hand_mean}")
            if self.config.humans.load_wo_tr:
                # Make transl zero:
                for i in range(len(fbx_motion_seq)):
                    fbx_motion_seq[i]["transl"] = np.zeros(3)
            renderable_smpl = model_class(camera=self.camera, gender=smpl_gender,
                                          smpl_root=self.smpl_root, template=template,  # global_offset=offset,
                                          model_type=smpl_type, center_root_joint=True,
                                          flat_hand_mean=flat_hand_mean, global_offset=corr_vct)
        else:
            raise ValueError(f"Unknown extension {poses_path_ext}")

        if template_path is not None:
            renderable_smpl.template_path = template_path

        renderable_smpl.default_gender = smpl_gender
        renderable_smpl.default_betas = smpl_betas

        logger.info("Uploading to GPU")
        status_line.text = "Uploading to GPU"
        self.redraw_gui()
        if prev_renderable_smpl is not None:
            self.timed_renderables.remove(prev_renderable_smpl)
            self.scene.objects.remove(prev_renderable_smpl)
            self.smpl_models.remove(prev_renderable_smpl)

        renderable_smpl.draw_shadows = False
        # renderable_smpl.generate_shadows = False
        renderable_smpl.init_context()
        renderable_smpl.set_material(0.65, 1, 0, 0)
        renderable_smpl.color_ind = -1
        if not old_style_fbx_seq:
            # Set the shape
            renderable_smpl.update_params(betas=betas)
        if model_class == SMPLXColoredModel:
            if len(self.smpl_models) > 0:
                occup_inds = []
                for mdl in self.smpl_models:
                    occup_inds.append(mdl.color_ind)
                free_inds = [i for i in range(len(color_palette)) if i not in occup_inds]
                if len(free_inds) > 0:
                    color_ind = free_inds[0]
                    logger.info(f"There are {len(self.smpl_models)} SMPL models, "
                                f"changing color to {color_palette[color_ind]}")
                    renderable_smpl.set_uniform_color(color_palette[color_ind])
                    renderable_smpl.color_ind = color_ind
                else:
                    logger.info(f"There are {len(self.smpl_models)} SMPL models, "
                                f"but all colors are occupied, no color change")
        else:
            renderable_smpl.set_texture(texture_data, uv_map)
            renderable_smpl.texture_path = model_texture_path

        if len(fbx_motion_seq) > 0 and ('pose' in fbx_motion_seq[0] or 'shape' in fbx_motion_seq[0]):
            # Converting to the new format
            rebuilt_fbx_motion_seq = []
            for old_style_params in fbx_motion_seq:
                new_params = {}
                if 'pose' in old_style_params:
                    new_params["global_orient"] = old_style_params["pose"][:3]
                    new_params["body_pose"] = old_style_params["pose"][3:]
                if 'translation' in old_style_params:
                    new_params["transl"] = old_style_params["translation"]
                if 'shape' in old_style_params:
                    new_params["betas"] = old_style_params["shape"]
                rebuilt_fbx_motion_seq.append(new_params)
            fbx_motion_seq = rebuilt_fbx_motion_seq

        if renderable_smpl.model_type in ["smplh", "smplx"] and len(fbx_motion_seq[0]["body_pose"]) == 69:
            fbx_motion_seq = [{k: v if k != "body_pose" else v[:63] for k, v in x.items()} for x in fbx_motion_seq]
        elif renderable_smpl.model_type == "smpl" and len(fbx_motion_seq[0]["body_pose"]) == 63:
            fbx_motion_seq = [{k: v if k != "body_pose" else np.concatenate([np.array(v), np.zeros(6)]) for k, v in x.items()} for x in
                              fbx_motion_seq]

        self.contacts_db[seq_prefix]["sequence_maxtime"] = float(fbx_timestamps[-1])
        self.contacts_db[seq_prefix]["sequence_mintime"] = float(fbx_timestamps[0])
        renderable_smpl.set_sequence(fbx_motion_seq, times=fbx_timestamps)
        self.max_seq_time = max(self.max_seq_time, fbx_timestamps[-1])
        renderable_smpl.set_time(self.global_time)
        renderable_smpl.tag = f"SMPL: {seq_prefix}"
        renderable_smpl.sequence_prefix = seq_prefix
        renderable_smpl.sequence_path = poses_path
        self.renderable_smpl = renderable_smpl

        self.scene.add_object(renderable_smpl)
        self.timed_renderables.append(renderable_smpl)
        self.smpl_models.append(renderable_smpl)
        self.camera_modes_available += {"relative", "first_person"}
        self.gui_controller.status_text.remove(status_line)

    def apply_current_camera(self):
        pos = self.camera_pos
        quat = np.roll(self.camera_rot.as_quat(), 1)
        self.camera.init_extrinsics(quat, pos)

    def request_color_async(self, pbo=None):
        width, height = self.viewport_resolution
        if pbo is None:
            pbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo)
            gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, (3 * width * height), None, gl.GL_STREAM_READ)
        else:
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo)
        # buffer = (GLubyte * (3 * width * height))(0)
        # gl.glReadBuffer(gl.GL_BACK)
        gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, 0)
        return pbo

    def get_requested_color(self, pbo, delete_pbo=True):
        width, height = self.viewport_resolution
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo)
        # data = np.ones((height, width, 3), dtype=np.uint8)*255
        # glGetBufferSubData(GL_PIXEL_PACK_BUFFER, 0, width*height*3, data)
        bufferdata = gl.glMapBuffer(gl.GL_PIXEL_PACK_BUFFER, gl.GL_READ_ONLY)
        data = np.frombuffer(ctypes.string_at(bufferdata, (3 * width * height)), np.uint8).reshape(height, width, 3)
        gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        if delete_pbo:
            gl.glDeleteBuffers(1, [pbo])
        return data[::-1]

    def get_image(self):
        width, height = self.viewport_resolution
        # gl.glReadBuffer(gl.GL_BACK)
        color_buf = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        color = np.frombuffer(color_buf, np.uint8).reshape(height, width, 4)[::-1]
        return color

    def save_frame_description(self, outpath, override_start_time=None, end_time=None):
        frame_descripton = {}
        if self.renderable_smpl is not None:
            frame_descripton["model"] = {k: v.tolist() for k, v in self.renderable_smpl.current_params.items()}
        frame_descripton["camera"] = {
            "c2w": np.hstack([self.camera_rot.as_matrix(), self.camera_pos[:, np.newaxis]]).tolist(),
            "fov": self.camera_fov, "far": self.far, "viewport_resolution": self.viewport_resolution.tolist(),
            "camera_yaw_pitch": self.camera_yaw_pitch.tolist(),
            "camera_roll": self.camera_roll,
            "camera_pos": self.camera_pos.tolist(),
            "camera_rot": np.roll(self.camera_rot.as_quat(), 1).tolist()}
        frame_descripton["scene_path"] = self.scene_path
        frame_descripton["template_height_correction"] = self.template_height_correction
        if self.main_renderable_pc is not None:
            frame_descripton["scene_pc_splat_size"] = self.main_renderable_pc.get_splat_size()
        frame_descripton["global_time"] = self.global_time if override_start_time is None else override_start_time
        if end_time:
            frame_descripton["video_start_time"] = frame_descripton["global_time"]
            frame_descripton["video_end_time"] = end_time
        frame_descripton["dir_light"] = {"direction": self.directional_light.direction.tolist(),
                                         "intensity": self.directional_light.intensity.tolist()}
        frame_descripton["smpl_models"] = [{"poses_path": str(model.sequence_path),
                                            "color": np.array(model.color).tolist(),
                                            "time_offset": float(model.time_offset),
                                            "texture_path": str(model.texture_path) if hasattr(model, "texture_path") else None,
                                            "template_path": str(model.template_path) if hasattr(model, "template_path") else None}
                                           for model in self.smpl_models]
        if self.objloc_path is not None:
            frame_descripton["object"] = {"objloc_path": self.objloc_path, "objloc_type": self.objloc_type}
        else:
            frame_descripton["object"] = None
        if self.kinect_prefix is not None:
            frame_descripton["kinects"] = {"prefix": self.kinect_prefix, "count": len(self.kinect_renderers)}
        if self.camera_mode == "trajectory" and self.raw_trajectory is not None:
            frame_descripton["trajectory"] = self.raw_trajectory
        json.dump(frame_descripton, open(outpath, "w"), indent=2)

    @staticmethod
    def get_last_fileindex(folder, file_prefix, index_mask, file_suffix):
        files = glob(os.path.join(folder, file_prefix + index_mask + file_suffix))
        if len(files) == 0:
            maxind = -1
        else:
            if len(file_suffix) > 0:
                maxind = max(
                    [int(os.path.basename(f)[len(file_prefix):-len(file_suffix)]) for f in files])
            else:
                maxind = max(
                    [int(os.path.basename(f)[len(file_prefix):]) for f in files])
        return maxind

    def save_screenshot(self):
        logger.info("Saving screenshot...")
        frame = self.get_image()
        maxind = self.get_last_fileindex(self.screen_capture_dir, self.screen_capture_prefix + "_", "????", ".png")
        imsave(os.path.join(self.screen_capture_dir, self.screen_capture_prefix + f"_{maxind + 1:04d}.png"), frame)
        if self.screen_capture_serialize:
            self.save_frame_description(os.path.join(self.screen_capture_dir, self.screen_capture_prefix + f"_{maxind + 1:04d}.json"))
        logtext = f"Saved as {self.screen_capture_prefix + f'_{maxind + 1:04d}'}"
        self.gui_controller.status_text.append(StatusTextLine(logtext, time_limit=2))
        logger.info(logtext)

    def load_from_description(self, frame_descripton):
        logger.info("Loading from frame description...")
        logger.info("Processing scene")
        if frame_descripton["scene_path"] != self.scene_path:
            if os.path.splitext(frame_descripton["scene_path"])[1] == ".ply":
                self.load_pointcloud(frame_descripton["scene_path"])
            else:
                self.load_scene(frame_descripton["scene_path"])
        logger.info("Setting camera")
        cam_desc = frame_descripton["camera"]
        self.camera_fov = cam_desc['fov']
        self.far = cam_desc['far']
        self.camera.init_intrinsics(self.viewport_resolution, fov=cam_desc['fov'], far=cam_desc['far'])
        self.camera_yaw_pitch = np.array(cam_desc['camera_yaw_pitch'])
        self.camera_roll = cam_desc['camera_roll'] if 'camera_roll' in cam_desc else 0.
        self.camera_pos = np.array(cam_desc['camera_pos'])
        self.camera_rot = Rotation.from_quat(np.roll(cam_desc['camera_rot'], -1))
        self.locate_camera()
        logger.info("Setting lights")
        dirlight_desc = frame_descripton["dir_light"]
        self.directional_light.direction = np.array(dirlight_desc['direction'])
        self.directional_light.intensity = np.array(dirlight_desc['intensity'])
        self.smpl_model_shadowmap_offset = -self.directional_light.direction * 3
        logger.info("Loading models")
        height_correction_on_templates = frame_descripton.get("template_height_correction", False)
        if self.config.humans.load_with_height_corr:
            height_correction_on_templates = True
        self.template_height_correction = height_correction_on_templates
        for model_desc in frame_descripton['smpl_models']:
            self.load_smpl_model(model_desc['poses_path'], height_correction_override=height_correction_on_templates)
            if 'color' in model_desc and model_desc["color"] is not None:
                self.smpl_models[-1].set_uniform_color(model_desc['color'])
            if "template_path" in model_desc and not self.config.humans.no_smpl_template:
                self.replace_smpl_template(model_desc["template_path"], self.smpl_models[-1],
                                           model_desc["texture_path"] if "texture_path" in model_desc else None,
                                           height_correction_override=height_correction_on_templates)
            if 'time_offset' in model_desc:
                self.smpl_models[-1].set_time_offset(model_desc["time_offset"])
        if frame_descripton['object'] is not None:
            logger.info("Loading object")
            if frame_descripton['object'].get("objloc_type", "single") == "single":
                self.load_objloc(frame_descripton['object']['objloc_path'])
            else:
                self.load_multi_objloc(frame_descripton['object']['objloc_path'])
        if 'kinects' in frame_descripton and frame_descripton['kinects'] is not None:
            if os.path.isdir(frame_descripton['kinects']["prefix"]):
                self.load_kinrec_seq(frame_descripton['kinects']["prefix"])
            else:
                self.load_kinects(frame_descripton['kinects']["prefix"] + ".0.color.mp4")
        self.reset_global_time(current_time=frame_descripton["global_time"])
        if "scene_pc_splat_size" in frame_descripton:
            self.main_renderable_pc.set_splat_size(frame_descripton["scene_pc_splat_size"])
        if "video_end_time" in frame_descripton:
            self.video_end_time = frame_descripton["video_end_time"]

    def handle_videocapture(self):
        if self.video_writer is None:
            files = glob(os.path.join(self.screen_capture_dir, self.screen_video_prefix + "_*.mp4"))
            if len(files) == 0:
                maxind = -1
            else:
                # maxind = max([int(f.split("_")[-1].split(".")[0]) for f in files])
                maxind = max([int(os.path.basename(f)[len(self.screen_video_prefix) + 1:len(self.screen_video_prefix) + 5]) for f in files])
            video_name = os.path.join(self.screen_capture_dir, self.screen_video_prefix + f"_{maxind + 1:04d}.mp4")
            self.video_capture_description_outpath = os.path.join(self.screen_capture_dir, self.screen_video_prefix + f"_{maxind + 1:04d}.json")

            self.video_writer = VideoWriter(video_name, resolution=self.viewport_resolution, fps=self.screen_capture_video_fps)
            self.screen_capture_frames_count = 0
        # Save ready frame
        pbo = None
        if len(self.screen_capture_queue) > 0:
            pbo, frame_ind = self.screen_capture_queue[0]
            if frame_ind + self.screen_capture_get_delay > self.global_frame_counter:
                pbo = None
            else:
                self.screen_capture_queue = self.screen_capture_queue[1:]
                self.video_writer.write(self.get_requested_color(pbo, delete_pbo=not self.screen_capture_active))
                if len(self.screen_capture_queue) == 0:
                    self.video_writer.close()
                    self.video_writer = None
                    logtext = f"Saved as {os.path.splitext(os.path.basename(self.video_capture_description_outpath))[0]}"
                    self.gui_controller.status_text.append(
                        StatusTextLine(logtext, time_limit=2))
                    logger.info(logtext)
                    if self.config.capturing.exit_on_videorec_stop:
                        sys.exit(0)

        # Request new frame
        if self.screen_capture_active:
            if self.video_capture_start_time is None:
                self.video_capture_start_time = self.global_time
                self.video_capture_end_time = None
            pbo = self.request_color_async(pbo)
            frame_ind = self.global_frame_counter
            self.screen_capture_queue.append((pbo, frame_ind))
            self.screen_capture_frames_count += 1
        else:
            if self.video_capture_end_time is None:
                self.video_capture_end_time = self.global_time
                if self.video_capture_save_description:
                    self.save_frame_description(self.video_capture_description_outpath,
                                                override_start_time=self.video_capture_start_time,
                                                end_time=self.video_capture_end_time)
                self.video_capture_start_time = None

    def handle_screen_capture(self):
        if self.screen_capture_mode == 'screenshot':
            if self.screen_capture_active:
                self.save_screenshot()
                self.screen_capture_active = False
        else:
            if self.screen_capture_active or len(self.screen_capture_queue) > 0:
                self.handle_videocapture()

    def locate_camera(self):
        if self.camera_mode == 'trajectory':
            impos = self.find_closest_camera_in_traj()
            pos = np.array(impos['position'])
            quat = np.array(impos['quaternion'])
            rot = Rotation.from_quat(np.roll(quat, -1))
            self.camera_pos = pos
            self.camera_rot = rot
            self.static_update = True
            self.request_redraw()
        elif self.camera_mode == "relative":
            pass
        self.apply_current_camera()

    @property
    def dynamic_objects(self):
        return self.timed_renderables

    def control_camera_pos(self, move):
        move = move * self.last_realtimediff
        if self.camera_mode == 'free':
            self.camera_pos += self.camera_rot.apply(np.array(move))
        elif self.camera_mode == 'relative':
            self.camera_relative_offset += self.camera_rot.apply(np.array(move))

    def get_elapsed_time(self):
        curr_time = time.time()
        if self.time_mode == "real":
            time_diff = curr_time - self.last_timestamp
        else:
            time_diff = self.global_time_step  # 1./self.fps_limit if self.fps_limit is not None else
        self.last_timestamp = curr_time
        self.last_realtimediff = time_diff
        time_diff = time_diff * self.time_scale
        self.last_timediff = time_diff
        return time_diff

    def control_camera_rot(self, mouse_diff, roll=None):
        if self.camera_mode != 'first_person':
            self.camera_yaw_pitch += mouse_diff * self.mouse_sensitivity * np.array([-1, 1])
            self.camera_yaw_pitch[1] = np.clip(self.camera_yaw_pitch[1], -np.pi / 2, np.pi / 2)
            if roll is not None:
                self.camera_roll += roll
            else:
                self.camera_roll = 0
            rot = Rotation.from_euler('YZX', (self.camera_roll, self.camera_yaw_pitch[0], self.camera_yaw_pitch[1]))
            self.camera_rot = rot * Rotation.from_euler('x', 90, degrees=True)

    def process_events(self):
        # logger.debug(f"{self.gui_focused}, {self.gui_hovered}")
        self.gui_controller.process_ingame_controls(self.gui_focused, self.gui_hovered)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

            self.gui_controller.process_event(event, self.gui_focused, self.gui_hovered)

    def draw(self):
        self.switch_to_seqfbo()
        if not self.draw_pause and (self.config.misc.continuous_redraw or self.needs_redraw):
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self.scene.draw()
            self.draw_contacts()
            # self.draw_redblue_hands()
            while len(self.draw_queue) > 0:
                draw_task = self.draw_queue[0]
                draw_task()
                self.draw_queue = self.draw_queue[1:]
            self.needs_redraw = False
        self.handle_screen_capture()
        self.restore_from_seq_fbo()
        self.gui_controller.draw_gui()
        pg.display.flip()

    def add_task(self, func, *args, **kwargs):
        self.task_queue.put((func, (args, kwargs)))

    def execute_tasks(self):
        while not self.task_queue.empty():
            func, (args, kwargs) = self.task_queue.get()
            func(*args, **kwargs)

    def restore_from_seq_fbo(self):
        width, height = self.viewport_resolution
        gui_width, gui_height = self.gui_resolution
        off_x, off_y = (self.gui_resolution - self.viewport_resolution) // 2
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._seqdraw_fb)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._main_fb['draw'])
        gl.glClearColor(0.0, 0.0, 0.2, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(*self.bkg_color)
        gl.glBlitFramebuffer(0, 0, width, height, off_x, off_y, off_x + width, off_y + height,
                             gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT, gl.GL_NEAREST)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._main_fb['read'])
        gl.glViewport(0, 0, *self.gui_resolution)

    def switch_to_seqfbo(self):
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._seqdraw_fb)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._seqdraw_fb)
        gl.glViewport(0, 0, *self.viewport_resolution)

    def switch_to_mainfbo(self):
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._main_fb['draw'])
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._main_fb['read'])
        gl.glViewport(0, 0, *self.gui_resolution)

    def change_precision(self, obj_type, internal_path):
        status_line = StatusTextLine(f"Loading {internal_path}")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()
        obj = trimesh_load_from_zip(self.scene_path, internal_path)
        status_line.text = "Uploading to GPU"
        self.redraw_gui()
        if obj_type == "mesh":
            self.main_renderable_mesh.set_buffers(obj)
            self.main_mesh = obj
        else:
            self.main_renderable_pc.set_buffers(obj)
            self.main_pc = obj
        self.gui_controller.status_text.remove(status_line)

    def load_trimesh_object(self, path):
        self.request_redraw()
        status_line = StatusTextLine(f"Loading {self.scene_path}")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()
        trimesh_object = trimesh.load(path, process=False, validate=False)
        self.gui_controller.status_text.remove(status_line)
        if isinstance(trimesh_object, trimesh.PointCloud):
            self.load_pointcloud(path, trimesh_object)
        else:
            self.load_mesh(path, trimesh_object)

    def load_pointcloud(self, pointcloud_path, pc = None):
        self.request_redraw()
        self.scene_path = pointcloud_path
        status_line = StatusTextLine(f"Loading {self.scene_path}")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()
        if pc is None:
            pc = trimesh.load(pointcloud_path, process=False, validate=False)
        if self.render_with_norms and "nx" in pc.metadata['ply_raw']['vertex']['properties']:
            logger.debug("Detected normals in the pointcloud, loading with normals")
            with_norms = True
        else:
            logger.debug("Loading PC without normals")
            with_norms = False
        pc = SimplePointcloud.PointcloudContainer(vertices=np.asarray(pc.vertices), colors=np.asarray(pc.colors),
                                                  normals=np.stack([pc.metadata['ply_raw']['vertex']['data'][x] for x in ["nx", "ny", "nz"]],
                                                                   axis=1) if with_norms else None)
        status_line.text = f"Uploading to GPU"
        self.redraw_gui()
        empty_pc = SimplePointcloud.PointcloudContainer(vertices=pc.vertices[:1], colors=pc.colors[:1], normals=np.array([[0, 0, 1.]]))
        if self.main_renderable_pc is not None and (
                with_norms != isinstance(self.main_renderable_pc, (SimplePointcloudWithNormals, AvgcolorPointcloudWithNormals))):
            logger.debug("Wrong PC format, reloading shaders...")
            self.main_renderable_pc.delete_buffers()
            self.main_renderable_obj_pc.delete_buffers()
            self.scene.objects.remove(self.main_renderable_pc)
            self.scene.objects.remove(self.main_renderable_obj_pc)
            self.main_renderable_pc = None
            self.main_renderable_obj_pc = None
        if self.main_renderable_pc is None:
            if with_norms:
                # Replace the two line below with the two commented lines to turn on splat color averaging
                renderable_obj_pc = SimplePointcloudWithNormals(self.camera)
                renderable_pc = SimplePointcloudWithNormals(self.camera)
                # renderable_obj_pc = AvgcolorPointcloudWithNormals(self.camera, resolution=self.viewport_resolution)
                # renderable_pc = AvgcolorPointcloudWithNormals(self.camera, resolution=self.viewport_resolution)
            else:
                renderable_obj_pc = SimplePointcloud(self.camera)
                renderable_pc = SimplePointcloud(self.camera)
            renderable_obj_pc.generate_shadows = False
            renderable_pc.generate_shadows = False
            renderable_obj_pc.init_context()
            renderable_pc.init_context()
            self.scene.add_object(renderable_pc)
            self.scene.add_object(renderable_obj_pc)
            self.main_renderable_pc = renderable_pc
            self.main_renderable_obj_pc = renderable_obj_pc
        self.main_renderable_pc.set_buffers(pc)
        self.main_renderable_pc.shading = 0.
        self.main_renderable_obj_pc.set_buffers(empty_pc)
        if self.main_renderable_mesh is not None:
            self.main_renderable_mesh.visible = False
        self.gui_controller.status_text.remove(status_line)
        self.main_pc = pc
        self.main_pc_path = pointcloud_path
        self.object_pc = None
        self.obj_locations = None
        self.main_mesh = None

    def load_mesh(self, mesh_path, mesh = None):
        self.request_redraw()
        self.scene_path = mesh_path
        status_line = StatusTextLine(f"Loading {self.scene_path}")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()
        if mesh is None:
            mesh = trimesh.load(mesh_path, process=False, validate=False)
        status_line.text = "Uploading to GPU"
        self.redraw_gui()
        if self.main_renderable_pc is not None:
            self.main_renderable_pc.delete_buffers()
            self.main_renderable_obj_pc.delete_buffers()
            self.scene.objects.remove(self.main_renderable_pc)
            self.scene.objects.remove(self.main_renderable_obj_pc)
            self.main_renderable_pc = None
            self.main_renderable_obj_pc = None
        if self.main_renderable_mesh is None:
            renderable_mesh = SimpleMesh(self.camera)
            renderable_mesh.generate_shadows = False
            renderable_mesh.init_context()
            self.scene.add_object(renderable_mesh)
            self.main_renderable_mesh = renderable_mesh
        self.main_renderable_mesh.set_buffers(mesh)
        self.main_renderable_mesh.set_material(ambient=0.4, diffuse=0.7, specular=0.1)
        self.gui_controller.status_text.remove(status_line)
        self.main_pc = None
        self.main_pc_path = None
        self.object_pc = None
        self.obj_locations = None
        self.main_mesh = mesh

    def load_scene(self, scene_zip_path):
        self.request_redraw()
        self.scene_path = scene_zip_path
        mesh_in_zip = \
            sorted(list_zip(self.scene_path, "mesh"), key=lambda x: int(os.path.basename(x).split("M")[0]))[0]
        pc_in_zip = sorted(list_zip(self.scene_path, "pointcloud"),
                           key=lambda x: int(os.path.basename(x).split("M")[0]))[0]
        status_line = StatusTextLine(f"Loading from {self.scene_path}")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()
        mesh = trimesh_load_from_zip(self.scene_path, mesh_in_zip)
        pc = trimesh_load_from_zip(self.scene_path, pc_in_zip)
        if self.render_with_norms and "nx" in pc.metadata['ply_raw']['vertex']['properties']:
            logger.debug("Detected normals in the pointcloud, loading with normals")
            with_norms = True
        else:
            logger.debug("Loading PC without normals")
            with_norms = False
        pc = SimplePointcloud.PointcloudContainer(vertices=np.asarray(pc.vertices), colors=np.asarray(pc.colors),
                                                  normals=np.stack([pc.metadata['ply_raw']['vertex']['data'][x] for x in ["nx", "ny", "nz"]],
                                                                   axis=1) if with_norms else None)
        self.gui_controller.status_text.remove(status_line)
        status_line = StatusTextLine(f"Uploading to GPU")
        self.gui_controller.status_text.append(status_line)
        self.redraw_gui()
        empty_pc = trimesh.PointCloud(vertices=pc.vertices[:1], colors=pc.colors[:1], normals=np.array([[0, 0, 1.]]))

        if self.main_renderable_pc is not None and (
                with_norms != isinstance(self.main_renderable_pc, (SimplePointcloudWithNormals, AvgcolorPointcloudWithNormals))):
            logger.debug("Wrong PC format, reloading shaders...")
            self.main_renderable_pc.delete_buffers()
            self.main_renderable_obj_pc.delete_buffers()
            self.scene.objects.remove(self.main_renderable_pc)
            self.scene.objects.remove(self.main_renderable_obj_pc)
            self.main_renderable_pc = None
            self.main_renderable_obj_pc = None
            if self.main_renderable_mesh is not None:
                self.main_renderable_mesh.delete_buffers()
                self.scene.objects.remove(self.main_renderable_mesh)
                self.main_renderable_mesh = None

        if self.main_renderable_pc is None:
            renderable_mesh = SimpleMesh(self.camera, generate_shadows=False)
            if with_norms:
                renderable_obj_pc = SimplePointcloudWithNormals(self.camera)
                renderable_pc = SimplePointcloudWithNormals(self.camera)
                # renderable_obj_pc = AvgcolorPointcloudWithNormals(self.camera, resolution=self.viewport_resolution)
                # renderable_pc = AvgcolorPointcloudWithNormals(self.camera, resolution=self.viewport_resolution)
            else:
                renderable_obj_pc = SimplePointcloud(self.camera)
                renderable_pc = SimplePointcloud(self.camera)
            renderable_obj_pc.generate_shadows = False
            renderable_pc.generate_shadows = False
            renderable_obj_pc.init_context()
            renderable_pc.init_context()
            renderable_mesh.init_context()
            self.scene.add_object(renderable_pc)
            self.scene.add_object(renderable_obj_pc)
            self.scene.add_object(renderable_mesh)
            self.main_renderable_pc = renderable_pc
            self.main_renderable_obj_pc = renderable_obj_pc
            self.main_renderable_mesh = renderable_mesh
        if self.main_renderable_mesh is None:
            renderable_mesh = SimpleMesh(self.camera)
            renderable_mesh.generate_shadows = False
            renderable_mesh.init_context()
            self.scene.add_object(renderable_mesh)
            self.main_renderable_mesh = renderable_mesh
        else:
            self.main_renderable_mesh.visible = True
        self.main_renderable_pc.set_buffers(pc)
        self.main_renderable_pc.shading = 0.
        self.main_renderable_obj_pc.set_buffers(empty_pc)
        self.main_renderable_mesh.set_buffers(mesh)
        self.gui_controller.status_text.remove(status_line)
        self.main_pc = pc
        self.main_mesh = mesh
        self.object_pc = None
        self.obj_locations = None

    def advance_global_time(self, time_delta):
        self.global_time += time_delta
        self.request_redraw()
        for renderable in self.timed_renderables:
            renderable.set_time(self.global_time)

    def reset_global_time(self, current_time=0.):
        self.global_time = current_time
        self.request_redraw()
        for renderable in self.timed_renderables:
            renderable.set_time(self.global_time)

    def get_global_time(self):
        return self.global_time

    def redraw_gui(self):
        self.restore_from_seq_fbo()
        self.gui_focused, self.gui_hovered = self.gui_controller.build_gui()
        self.gui_controller.draw_gui()
        pg.display.flip()

    def request_redraw(self):
        self.needs_redraw = True

    def main_loop(self):
        self.sliding_fps = 0.0
        sliding_fps_acc_size = 0
        self.loop_active = True
        self.reset_global_time()

        while self.loop_active:
            frame_start_time = time.time()
            elapsed_time = self.get_elapsed_time()
            self.process_events()
            if not self.loop_active:
                break
            self.gui_focused, self.gui_hovered = self.gui_controller.build_gui()

            if self.animation_active:
                self.advance_global_time(elapsed_time)
            if self.renderable_smpl is not None:
                current_dir = np.array([0, 0, -1.])
                target_dir = self.directional_light.direction
                min_rot_vector = np.cross(current_dir, target_dir)
                quat = np.roll(Rotation.from_rotvec(min_rot_vector).as_quat(), 1)
                self.smpl_model_shadowmap.camera.init_extrinsics(quat=quat,
                                                                 pose=self.renderable_smpl.global_translation + self.smpl_model_shadowmap_offset)

            self.locate_camera()
            if self.obj_locations is not None:
                objloc_frame_ind = self.obj_frame_inds[get_closest_ind_before(self.obj_locations_timestamps, self.global_time)]
                objloc_frame_name = f"{objloc_frame_ind:06d}"
                obj_location = self.obj_locations[objloc_frame_name]

                if obj_location is not None:
                    obj_model_position = np.array(obj_location["position"])
                    obj_model_quat = obj_location["quaternion"]
                    self.main_renderable_obj_pc.init_model_extrinsics(obj_model_quat, obj_model_position)
                    self.last_obj_location = (obj_model_quat, obj_model_position)

            self.draw()

            self.execute_tasks()

            if self.fps_limit is not None:
                self.fps_clock.tick(self.fps_limit)

            current_fps = 1 / (time.time() - frame_start_time)
            self.sliding_fps = (self.sliding_fps * sliding_fps_acc_size + current_fps) / (sliding_fps_acc_size + 1)
            if sliding_fps_acc_size < self.sliding_fps_window_size:
                sliding_fps_acc_size += 1

            if self.global_frame_counter == 0 and self.preload_path is not None:
                self.request_redraw()
                self.gui_controller.status_text.append(StatusTextLine(f"Restoring from {os.path.basename(self.preload_path)}", 10))
                self.load_from_description(json.load(open(self.preload_path)))

                if self.config.capturing.recvid:
                    self.screen_capture_active = True
                    rv_starting_time = self.global_time
                    self.screen_capture_mode = 'video'
                    self.require_complete_rendering = self.screen_capture_active
                    self.time_mode = 'constant' if self.screen_capture_active else 'real'
                    self.animation_active = not self.animation_active

                if self.config.capturing.rectraj:
                    te_dialog = TrajectoryEditDialog(self, self.gui_controller, self.raw_trajectory)
                    te_dialog.load_trajectory(self.config.capturing.rectraj)
                    te_dialog.run_trajectory(record=True)

                if self.config.preload.nobkg:
                    self.main_renderable_pc.visible = False

                if self.config.preload.nosmpl:
                    for renderable_smpl_model in self.smpl_models:
                        renderable_smpl_model.visible = False
                        renderable_smpl_model.generate_shadows = False

                if self.config.preload.noobj:
                    if self.main_renderable_obj_pc is not None:
                        self.main_renderable_obj_pc.visible = False

            if self.config.capturing.recvid and ((self.config.capturing.recvid > 0) and (
                    self.global_time > rv_starting_time + self.config.capturing.recvid) or (self.config.capturing.recvid < 0) and (
                    self.global_time > self.video_end_time)):
                if self.screen_capture_active:
                    self.screen_capture_active = False
                    self.animation_active = False
                    self.config.capturing.recvid = None

            self.global_frame_counter += 1
