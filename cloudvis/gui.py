import pygame as pg
import imgui
from typing import List, Optional, Dict, Callable, Tuple, Union
from .imgui_pygame import PygameRenderer
from OpenGL import GL as gl
import os
import time
from loguru import logger
import json
from pathlib import Path
from glob import glob
import numpy as np
from cloudrender.utils import list_zip
from cloudrender.render.smpl import SMPLXColoredModel, SMPLXModelBase
from functools import partial
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import gaussian
from .utils import VideoScroller
from videoio import read_video_params


def smpl_update_color(smpl: SMPLXColoredModel, color):
    smpl.set_uniform_color(color)
    smpl.update_buffers()


class GUIWindowBase:
    def __init__(self, windows_list):
        self.window_id = 0
        self.active = True
        self.first_start = True
        while self._window_id_exists(windows_list):
            self.window_id += 1

    def _window_id_exists(self, windows_list):
        for window in windows_list:
            if isinstance(window, GUIWindowBase):
                if self.window_id == window.window_id:
                    return True
        return False

    def build(self):
        if self.active:
            if self.first_start:
                self._first_start()
                self.first_start = False
            self._build()

    def _build(self):
        pass

    def _first_start(self):
        pass


class OpenDialogWindow:
    def __init__(self, starting_folder, opening_function, show_links=False, prefilter_by=None, prefilter_folders=True):
        self.curr_path = os.path.realpath(starting_folder)
        self.opening_function = opening_function
        self.show_links = show_links
        self.prefilter_by = prefilter_by
        self.prefilter_folders = prefilter_folders
        self.current_selection = 0
        self.active = True
        self.update_files()

    def update_files(self):
        if not os.path.exists(self.curr_path):
            # if path is incorrect, change it to the current directory
            self.curr_path = os.path.realpath(".")
        files = sorted(os.listdir(self.curr_path))
        self.file_entries = []
        for fname in files:
            if fname[0] != ".":
                path = os.path.join(self.curr_path, fname)
                if os.path.isdir(path):
                    fname = fname + "/"
                if (self.show_links or not os.path.islink(path)) and \
                        (self.prefilter_by is None or (not self.prefilter_folders and os.path.isdir(path)) or self.prefilter_by in fname):
                    self.file_entries.append(fname)
        self.file_entries = [".."] + self.file_entries

    def build(self):
        if self.active:
            expanded, opened = imgui.begin("Open file", True)
            if opened:
                imgui.push_item_width(-1)
                imgui.text(self.curr_path)

                clicked, self.current_selection = imgui.listbox(self.curr_path, self.current_selection, self.file_entries,
                                                                height_in_items=20)
                mouse_dc = imgui.is_mouse_double_clicked()
                cancel_pressed = imgui.button("Cancel")
                imgui.same_line(spacing=10)
                ok_pressed = imgui.button("OK")
                if cancel_pressed:
                    self.active = False
                elif mouse_dc or ok_pressed:
                    path = os.path.join(self.curr_path, self.file_entries[self.current_selection])
                    path = os.path.realpath(path)
                    self.curr_path = path
                    if os.path.isdir(path):
                        self.update_files()
                    else:
                        logger.info(f"Opening {self.curr_path}")
                        self.opening_function(self.curr_path)

                        self.active = False
            else:
                self.active = False
            imgui.end()


class TimeOffsetDialogWindow:
    def __init__(self, timed_obj):
        self.timed_obj = timed_obj
        self.offset = timed_obj.time_offset
        self.active = True

    def build(self):
        if self.active:
            expanded, opened = imgui.begin("Choose offset", True)
            if opened:
                changed, self.offset = imgui.core.input_float("Offset, sec", self.offset)
                if changed:
                    self.timed_obj.set_time_offset(self.offset)
            else:
                self.active = False
            imgui.end()


class ContactLabellingWindow(GUIWindowBase):
    def __init__(self, controller, windows_list: List[GUIWindowBase], global_time_getter: Callable[[], float],
            contacts_getter: Callable[[], Dict[str, Dict[str, Dict[str, np.ndarray]]]],
            contacts_setter: Callable[[dict], None]):
        super().__init__(windows_list)
        self.contacts_getter = contacts_getter
        self.global_time_getter = global_time_getter
        self.contacts_setter = contacts_setter
        self.first_start = True
        self.main_controller = controller

    def rebuild_ranged_array(self, ranged_array):
        ranged_array = np.asarray(ranged_array)
        if len(ranged_array) < 2:
            return ranged_array
        starts = ranged_array[:, 0]
        starts_inds_sorted = np.argsort(starts)
        sorted_ranged_array = ranged_array[starts_inds_sorted, :]
        merged_ranged_list = []
        curr_range = sorted_ranged_array[0]
        for next_range in sorted_ranged_array[1:]:
            if next_range[0] < curr_range[1]:
                curr_range[1] = max(curr_range[1], next_range[1])
            else:
                merged_ranged_list.append(curr_range)
                curr_range = next_range
        merged_ranged_list.append(curr_range)
        return np.stack(merged_ranged_list, axis=0)

    def _first_start(self):
        imgui.set_next_window_size(500, 300)
        self.scrolling_queries = set()

    def _build(self):
        expanded, opened = imgui.begin(
            f"({self.window_id}) Edit contacts", True)
        if opened:
            _, self.main_controller.draw_contacts_flag = imgui.checkbox("Draw contacts", self.main_controller.draw_contacts_flag)
            contacts_db = self.contacts_getter()
            global_time = self.global_time_getter()
            scrolling_queries = self.scrolling_queries
            for seqname, contacts_seq_info in contacts_db.items():
                # if "visible" not in contacts_seq_info:
                #     contacts_seq_info["visible"] = False
                if contacts_seq_info["type"] == "intervals":  # not "sequential"
                    expanded, _ = imgui.collapsing_header(seqname)
                    if expanded:
                        contact_surfaces = contacts_seq_info["contacts"]
                        for contact_surface_name, surface_contacts in contact_surfaces.items():
                            expanded, _ = imgui.collapsing_header(contact_surface_name + f"##{seqname}")
                            if expanded:
                                contact_region_tag = f"{seqname}, {contact_surface_name}"
                                imgui.begin_group()
                                imgui.begin_child(f"region {contact_region_tag}", -10, 200)
                                for contact_ind, contact_range in enumerate(surface_contacts):
                                    imgui.push_item_width(200)
                                    tag = f"{contact_ind} {seqname}, {contact_surface_name}"
                                    imgui.text(f"{contact_ind}")
                                    imgui.same_line(spacing=5)
                                    changed, contact_range[0] = imgui.drag_float(f"##start {tag}", contact_range[0], change_speed=0.033,
                                                                                 min_value=contacts_seq_info["sequence_mintime"],
                                                                                 max_value=min(contacts_seq_info["sequence_maxtime"],
                                                                                               contact_range[1]),
                                                                                 flags=imgui.SLIDER_FLAGS_ALWAYS_CLAMP)
                                    imgui.same_line(spacing=0)
                                    if imgui.button(f"now##start {tag}"):
                                        contact_range[0] = global_time
                                    imgui.same_line(spacing=20)
                                    changed, contact_range[1] = imgui.drag_float(f"##end {tag}", contact_range[1], change_speed=0.033,
                                                                                 min_value=max(contacts_seq_info["sequence_mintime"],
                                                                                               contact_range[0]),
                                                                                 max_value=contacts_seq_info["sequence_maxtime"],
                                                                                 flags=imgui.SLIDER_FLAGS_ALWAYS_CLAMP)
                                    imgui.same_line(spacing=0)
                                    if imgui.button(f"now##end {tag}"):
                                        contact_range[1] = global_time
                                    imgui.pop_item_width()
                                    surface_contacts[contact_ind, :] = contact_range
                                if contact_region_tag in scrolling_queries:
                                    imgui.set_scroll_from_pos_y(imgui.get_scroll_max_y() + 100, 0.)
                                    scrolling_queries.remove(contact_region_tag)
                                imgui.end_child()
                                if imgui.button(f"Add contact ##{seqname} {contact_surface_name}"):
                                    contact_surfaces[contact_surface_name] = np.concatenate(
                                        [surface_contacts, np.full((1, 2), global_time, dtype=surface_contacts.dtype)])
                                    scrolling_queries.add(contact_region_tag)
                                imgui.end_group()
                                if imgui.is_item_hovered():
                                    self.main_controller.draw_queue.append(
                                        partial(self.main_controller.draw_contact_sphere, seqname, contact_surface_name))

            if imgui.button("Resort and save"):
                for seqname, contacts_seq_info in contacts_db.items():
                    if contacts_seq_info["type"] == "intervals":
                        contact_surfaces = contacts_seq_info["contacts"]
                        for contact_surface_name, surface_contacts in contact_surfaces.items():
                            contact_surfaces[contact_surface_name] = self.rebuild_ranged_array(surface_contacts)
                self.contacts_setter(contacts_db)
                logger.info("Contacts saved successfully")
        else:
            self.active = False
        imgui.end()


class VideoWindow(GUIWindowBase):
    def __init__(self, global_time_getter: Callable[[], float], windows_list: List[GUIWindowBase], videos_dir: str,
            video_path: Optional[Union[str, Path]] = None,
            video_scale: int = 4):
        super().__init__(windows_list)
        self.video_path: Optional[Path] = None
        self.windows = windows_list
        self.videos_dir = videos_dir
        self.time_offset = 0
        self.video_scale = video_scale
        self.active = True
        self.global_time_getter = global_time_getter
        self.load_video(video_path)

    def load_video(self, video_path: Optional[Union[Path, str]]):
        if video_path is None:
            self.video_path = None
            return
        self.video_path = Path(video_path)
        self.time_offset = 0
        if self.video_path.with_suffix(".json").exists():
            video_metadata = json.load(open(self.video_path.with_suffix(".json")))
            self.time_offset = video_metadata["time_offset"]
        if self.video_scale != 1:
            video_info = read_video_params(self.video_path)
            resolution = np.array([video_info['width'], video_info['height']], dtype=int)
            scaled_resolution = resolution // self.video_scale
            self.video_handler = VideoScroller(self.video_path, output_resolution=scaled_resolution, cache_size=1000)
        else:
            self.video_handler = VideoScroller(self.video_path)

        self.init_texture()

    def init_texture(self):
        self.texturebuffer = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texturebuffer)

    def load_frame_to_tex(self, frame_ind) -> Tuple[int, Tuple[int, int]]:
        texture_data = self.video_handler.get_frame(frame_ind)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texturebuffer)
        gltexture = np.copy(texture_data.reshape(-1), order="C")
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
                        gltexture)
        return self.texturebuffer, (texture_data.shape[1], texture_data.shape[0])

    def _first_start(self):
        imgui.set_next_window_size(500, 300)

    def _build(self):
        expanded, opened = imgui.begin(
            f"({self.window_id})" + ("Select a video" if self.video_path is None else f"Video - {os.path.basename(self.video_path)}"), True)
        if opened:
            if self.video_path is not None:
                curr_frame_ind = int(self.video_handler.fps * (self.global_time_getter() + self.time_offset))
                tex_id, tex_size = self.load_frame_to_tex(curr_frame_ind)
                imgui.image(tex_id, tex_size[0], tex_size[1])

            changed, self.time_offset = imgui.core.input_float("Offset, sec", self.time_offset)
            if imgui.button("Load video"):
                self.windows.append(OpenDialogWindow(self.videos_dir, self.load_video))
            imgui.same_line()
            if imgui.button("Save offsets"):
                if self.video_path is not None:
                    json.dump({"time_offset": float(self.time_offset)}, open(self.video_path.with_suffix(".json"), "w"), indent=2)
                    logger.info(f"Saved offsets for video {self.video_path}")
            imgui.same_line()
            if imgui.button("-33 ms"):
                self.time_offset -= 0.033
            imgui.same_line()
            if imgui.button("+33 ms"):
                self.time_offset += 0.033
        else:
            self.active = False
        imgui.end()


class PrecisionDialogWindow:
    def __init__(self, scene_path, precision_change_function):
        self.scene_zip_path, self.folder_in_zip = scene_path.split(":")
        self.precision_change_function = precision_change_function
        self.file_entries = sorted(list_zip(self.scene_zip_path, self.folder_in_zip))
        self.entries = [os.path.splitext(x)[0] for x in self.file_entries]
        self.current_selection = 0
        self.active = True

    def build(self):
        if self.active:
            expanded, opened = imgui.begin("Open file", True)
            if opened:
                clicked, self.current_selection = imgui.listbox("Select precision", self.current_selection, self.entries)
                if imgui.is_mouse_double_clicked():
                    internal_path = self.file_entries[self.current_selection]
                    logger.info(f"Loading {internal_path}")
                    self.precision_change_function(internal_path)
                    self.active = False

            else:
                self.active = False
            imgui.end()


class ColorPickerDialog:
    def __init__(self, original_color, color_change_function):
        self.color = original_color
        self.color_change_function = color_change_function
        self.active = True

    def build(self):
        if self.active:
            expanded, opened = imgui.begin("Color picker", True)
            if opened:
                if imgui.button("Reset"):
                    self.color = np.array((200, 200, 200, 255))
                    self.color_change_function(self.color)
                color_changed, color_picker_color = imgui.color_edit4(f"Color", *np.array(self.color).astype(float) / 255., show_alpha=True)
                color_picker_color = (np.array(color_picker_color) * 255).astype(np.uint8)
                if color_changed:
                    self.color_change_function(color_picker_color)
                    self.color = color_picker_color
            else:
                self.active = False
            imgui.end()

class ControlsHelpWindow(GUIWindowBase):
    def __init__(self, windows_list):
        super().__init__(windows_list)

    def _first_start(self):
        imgui.set_next_window_size(500, 300)

    def _build(self):
        expanded, opened = imgui.begin(
            "Controls help", True)
        if opened:
            imgui.text_colored("Main viewport controls (click to focus):", 0.3,0.6,0)
            imgui.text("A/D, W/S, Q/E - camera movement (left/right, forward/backward, down/up)")
            imgui.text("Shift - speed up")
            imgui.text("Ctrl - slow down")
            imgui.text("Left click (hold) + Mouse move - look around")
            imgui.text("Left click (hold) + T/Y - roll the camera")
            imgui.text("Scroll wheel - change camera FOV")
            imgui.text("Space - start time (for animations)")
            imgui.text("Tab - reset time")
            imgui.text("[ or ] - decrease/increase global time")
            imgui.text("- or + - decrease/increase time increment")
            imgui.text(", and . - decrease/increase point size (only for the pointcloud rendering)")
            imgui.text("Arrow keys - control light position")
            imgui.text("Z - start/stop screen capture")
            imgui.text("Shift + Z - take a screenshot")
            imgui.text("Drag&Drop a .ply file - load a mesh/pointcloud")
            imgui.text("P - Toggle draw loop pause")
            imgui.text("1 - toggle main scene pointcloud visibility")
            imgui.text("2 - toggle main scene mesh visibility")
            imgui.text("3 - toggle kinect recordings visibility")
            imgui.text("4 - toggle SMPL bodies visibility")
            imgui.text("5 - toggle interactive objects visibility")

        else:
            self.active = False
        imgui.end()


class MainSceneWindow:
    def __init__(self, cb_texture, resolution):
        self.cb_texture = cb_texture
        self.resolution = resolution
        self.active = True

    def build(self):
        if self.active:
            expanded, opened = imgui.begin("Main window", False, imgui.core.WINDOW_NO_TITLE_BAR)
            if opened:
                imgui.image(self.cb_texture, *self.resolution, uv0=(0., 1.), uv1=(1., 0.))
            else:
                self.active = False
            imgui.end()


class StatusTextLine:
    def __init__(self, text, time_limit=None, color=None):
        self.text = text
        self.starting_time = time.time()
        self.time_limit = time_limit
        self.color = color

    def active(self):
        if self.time_limit is None:
            return True
        return time.time() < self.time_limit + self.starting_time

    def refresh(self, new_time_limit=None):
        if new_time_limit is not None:
            self.time_limit = new_time_limit
        self.starting_time = time.time()


class GuiController:

    def __init__(self, main_controller, resolution):
        self.main_controller = main_controller
        imgui.create_context()
        self.gui_renderer = PygameRenderer()
        self.io = imgui.get_io()
        self.resolution = resolution
        self.io.display_size = self.resolution
        self.load_fonts()
        self.dialogs = []
        self.status_text: List[StatusTextLine] = []

        self.mouse_tracking = False
        self.mouse_capture = False
        self.mouse_capture_pos = self.resolution / 2
        self.mouse_sensitivity = 1 / 300.
        self.movement_speed = 3.0
        self.light_movement_speed = 0.05
        self.gui_state = "main"
        self.gui_state_stack = [("main", None)]
        self.gui_state_handlers = {
            "main": self.gui_events_main
        }
        self.pause_info = StatusTextLine("Draw loop paused", color=(1, 1, 0))

    def add_main_scene(self, cb_texture, resolution):
        self.dialogs.append(MainSceneWindow(cb_texture, resolution))

    def load_fonts(self, sizes=(10, 15, 20, 30, 40)):
        self.fonts = {}
        for fontpath in glob(os.path.join(os.path.dirname(__file__), "fonts/*.ttf")):
            fontname = os.path.splitext(os.path.basename(fontpath))[0]
            for size in sizes:
                self.fonts[(fontname, size)] = self.io.fonts.add_font_from_file_ttf(fontpath, size)
        logger.info(f"Loaded {len(self.fonts) // len(sizes)} custom fonts")
        self.gui_renderer.refresh_font_texture()

    def build_main_text_bar(self, bar_height):
        imgui.set_next_window_bg_alpha(0.6)
        imgui.set_next_window_position(0, bar_height)
        gtime = self.main_controller.global_time
        sb_text = [StatusTextLine(f"FPS: {self.main_controller.sliding_fps:.2f}", color=(1., 0, 0)),
                   StatusTextLine(f"Global time: {gtime // 3600:02.0f}:{(gtime % 3600) // 60:02.0f}:{gtime % 60:05.2f}",
                                  color=(1., 0.5, 0.))]  # (0, 0.93, 1.)
        if self.main_controller.screen_capture_active and self.main_controller.screen_capture_mode == 'video':
            time_seconds = int(self.main_controller.screen_capture_frames_count // self.main_controller.screen_capture_video_fps)
            sb_text.append(
                StatusTextLine(f"REC: {self.main_controller.screen_capture_frames_count} frames ({time_seconds // 60:02d}:{time_seconds % 60:02d})",
                               color=(1, 0, 0)))
        if not self.main_controller.screen_capture_active and len(self.main_controller.screen_capture_queue) > 0:
            sb_text.append(StatusTextLine(f"Dumping buffer: {len(self.main_controller.screen_capture_queue)} frames left", color=(0, 0.8, 0.2)))

        imgui.push_font(self.fonts[("misc-fixed", 15)])
        overall_size = (0, 10)
        for line in ([x.text for x in sb_text + self.status_text if x.active()]):
            size = imgui.calc_text_size(line)
            w = 0
            if overall_size[1] != 0:
                w = 5
            overall_size = (max(overall_size[0], size[0] + 20), overall_size[1] + size[1] + w)

        imgui.set_next_window_size(*overall_size)
        imgui.begin("Text", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                    imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_INPUTS)

        height_offset = 0
        for status_line in sb_text + self.status_text:
            if status_line.active():
                if status_line.color is None:
                    imgui.text(status_line.text)
                else:
                    imgui.text_colored(status_line.text, *status_line.color)
            else:
                self.status_text.remove(status_line)

        imgui.pop_font()
        imgui.end()

    def build_gui(self):
        imgui.new_frame()
        bar_height = 0
        clicked_open_scene = False
        clicked_load_smpl = False
        if imgui.begin_main_menu_bar():
            bar_height = imgui.get_window_height()
            if imgui.begin_menu("File", True):
                clicked_open_scene, _ = imgui.menu_item("Open zipped scene", "", False, True)
                clicked, _ = imgui.menu_item("Open pointcloud", "", False, True)
                if clicked:
                    logger.info("Opened 'Open pointcloud' dialog")
                    self.dialogs.append(OpenDialogWindow(self.main_controller.config.scenes.pointclouds_dir,
                                                         lambda x: self.main_controller.add_task(
                                                             self.main_controller.load_pointcloud, x), show_links=True))
                clicked, _ = imgui.menu_item("Open mesh", "", False, True)
                if clicked:
                    logger.info("Opened 'Open mesh' dialog")
                    self.dialogs.append(OpenDialogWindow(self.main_controller.config.scenes.pointclouds_dir,
                                                         lambda x: self.main_controller.add_task(
                                                             self.main_controller.load_mesh, x), show_links=True))
                clicked_load_smpl, _ = imgui.menu_item("Add SMPL animation", "", False, True)
                clicked, _ = imgui.menu_item("Open object localization (single, old format)", "", False,
                                             self.main_controller.scene_path is not None)
                if clicked:
                    logger.info("Opened 'Open object localization' dialog")
                    self.dialogs.append(OpenDialogWindow(self.main_controller.config.objects.object_locations_dir,
                                                         lambda x: self.main_controller.add_task(
                                                             self.main_controller.load_objloc, x)))

                clicked, _ = imgui.menu_item("Open object localization", "", False,
                                             self.main_controller.scene_path is not None)
                if clicked:
                    logger.info("Opened 'Open object localization' dialog")
                    self.dialogs.append(OpenDialogWindow(self.main_controller.config.objects.object_locations_dir,
                                                         lambda x: self.main_controller.add_task(
                                                             self.main_controller.load_multi_objloc, x)))

                clicked, _ = imgui.menu_item("Open kinect seq (old format)", "", False,
                                             self.main_controller.scene_path is not None)
                if clicked:
                    logger.info("Opened 'Open kinect seq (old format)' dialog")
                    self.dialogs.append(OpenDialogWindow(self.main_controller.config.videos.kinects_dir,
                                                         lambda x: self.main_controller.add_task(
                                                             self.main_controller.load_kinects, x),
                                                         prefilter_by=".0.color.mp4"))
                clicked, _ = imgui.menu_item("Open Kinrec sequence", "", False,
                                             True)
                if clicked:
                    logger.info("Opened 'Open Kinrec sequence' dialog")
                    self.dialogs.append(OpenDialogWindow(self.main_controller.config.videos.kinrec_dir,
                                                         lambda x: self.main_controller.add_task(
                                                             self.main_controller.load_kinrec_seq, x),
                                                         prefilter_by="metadata.json", prefilter_folders=False))
                clicked, _ = imgui.menu_item("Edit trajectory", "", False, True)
                if clicked:
                    logger.info("Opened 'Edit trajectory' dialog")
                    self.dialogs.append(TrajectoryEditDialog(self.main_controller, self, self.main_controller.raw_trajectory))

                clicked, _ = imgui.menu_item("Open video", "", False, True)
                if clicked:
                    logger.info("Opened 'VideoWindow' dialog")
                    self.dialogs.append(
                        VideoWindow(self.main_controller.get_global_time, windows_list=self.dialogs,
                                    videos_dir=self.main_controller.config.videos.videos_dir))

                clicked_quit, _ = imgui.menu_item(
                    "Quit", "", False, True
                )

                if clicked_quit:
                    self.main_controller.loop_active = False

                imgui.end_menu()

            if imgui.begin_menu("Rendering", True):
                clicked, _ = imgui.menu_item("Constant time mode", "", self.main_controller.time_mode == "constant", True)
                if clicked:
                    if self.main_controller.time_mode == "constant":
                        self.main_controller.time_mode = "real"
                    else:
                        self.main_controller.time_mode = "constant"
                if imgui.begin_menu(f"Time scale", True):
                    available_scales = [0.125, 0.25, 0.5, 1., 2., 4.]
                    for scale in available_scales:
                        clicked, _ = imgui.menu_item(f"{scale:.3f}", "", np.allclose(self.main_controller.time_scale, scale), True)
                        if clicked:
                            self.main_controller.time_scale = scale
                    imgui.end_menu()
                clicked, _ = imgui.menu_item("Save video description", "", self.main_controller.video_capture_save_description,
                                             True)
                if clicked:
                    self.main_controller.video_capture_save_description = not self.main_controller.video_capture_save_description
                if imgui.begin_menu(f"Camera mode", True):
                    for cmode in self.main_controller.CAMERA_MODES:
                        clicked, _ = imgui.menu_item(cmode, "", self.main_controller.camera_mode == cmode,
                                                     cmode in self.main_controller.camera_modes_available)
                        if clicked:
                            self.main_controller.switch_camera_mode(cmode)
                    imgui.end_menu()
                if imgui.begin_menu(f"Main PC shading", hasattr(self.main_controller, "main_renderable_pc") and
                                                        self.main_controller.main_renderable_pc is not None):
                    changed, shading = imgui.slider_float("Shading", self.main_controller.main_renderable_pc.shading, 0.0, 1.0)
                    if changed:
                        self.main_controller.main_renderable_pc.shading = shading
                        self.main_controller.main_renderable_pc.set_overlay_color((255, 255, 255, int(255 * shading)))
                    imgui.end_menu()
                if imgui.begin_menu(f"Main PC saturation", hasattr(self.main_controller, "main_renderable_pc") and
                                                        self.main_controller.main_renderable_pc is not None):
                    changed, saturation = imgui.slider_float("Saturation", self.main_controller.main_renderable_pc.hsv_multiplier[1], 0.0, 2.0)
                    if changed:
                        hsv_multiplier = list(self.main_controller.main_renderable_pc.hsv_multiplier)
                        hsv_multiplier[1] = saturation
                        self.main_controller.main_renderable_pc.set_hsv_multiplier(hsv_multiplier)
                    imgui.end_menu()
                imgui.end_menu()

            if imgui.begin_menu("Scene", self.main_controller.scene_path is not None and
                                         os.path.splitext(self.main_controller.scene_path)[1] == ".zip"):
                clicked, _ = imgui.menu_item("Change pointcloud precision", "", False, True)
                if clicked:
                    logger.info("Opened 'Change pointcloud precision' dialog")
                    self.dialogs.append(PrecisionDialogWindow(self.main_controller.scene_path + ":pointcloud",
                                                              lambda x: self.main_controller.add_task(self.main_controller.change_precision,
                                                                                                      "pointcloud", x)))
                clicked, _ = imgui.menu_item("Change mesh precision", "", False, True)
                if clicked:
                    logger.info("Opened 'Change mesh precision' dialog")
                    self.dialogs.append(PrecisionDialogWindow(self.main_controller.scene_path + ":mesh",
                                                              lambda x: self.main_controller.add_task(self.main_controller.change_precision, "mesh",
                                                                                                      x)))

                imgui.end_menu()
            if imgui.begin_menu("SMPL models", len(self.main_controller.dynamic_objects) > 0):
                clicked, _ = imgui.menu_item(f"Edit contacts", "", False, True)
                if clicked:
                    self.dialogs.append(ContactLabellingWindow(self.main_controller, self.dialogs, self.main_controller.get_global_time,
                                                               self.main_controller.get_contacts,
                                                               self.main_controller.set_contacts))
                for ind, obj in enumerate(filter(lambda x: isinstance(x, SMPLXModelBase),
                                                 self.main_controller.dynamic_objects)):
                    if imgui.begin_menu(f"{ind}) {obj.tag}", True):
                        if obj.tag.startswith("SMPL"):
                            obj.last_overlay_color = np.asarray(obj.overlay_color).copy()
                            obj.set_overlay_color((255, 165, 0, 80))
                            clicked, _ = imgui.menu_item("Visible", "", obj.visible, True)
                            if clicked:
                                obj.visible = not obj.visible
                            clicked, _ = imgui.menu_item("Pick color", "", False, True)
                            if clicked:
                                logger.info("Opened color picker")
                                self.dialogs.append(ColorPickerDialog(obj.color, partial(smpl_update_color, obj)))
                            clicked, _ = imgui.menu_item("Replace", "", False, True)
                            if clicked:
                                logger.info("Opened 'load SMPL' dialog")
                                self.dialogs.append(OpenDialogWindow(self.main_controller.config.humans.smpl_motions_dir,
                                                                     lambda x: self.main_controller.add_task(self.main_controller.load_smpl_model, x,
                                                                                                             prev_renderable_smpl=obj)))
                            clicked, _ = imgui.menu_item("Choose template", "", False, True)
                            if clicked:
                                logger.info("Opened 'choose template' dialog")
                                self.dialogs.append(OpenDialogWindow(self.main_controller.config.humans.smpl_templates_dir,
                                                                     partial(self.main_controller.add_task,
                                                                             self.main_controller.replace_smpl_template,
                                                                             smpl_model=obj), prefilter_by="_unposed.pkl", prefilter_folders=False))
                            clicked, _ = imgui.menu_item("Refresh", "", False, True)
                            if clicked:
                                self.main_controller.add_task(self.main_controller.load_smpl_model, obj.sequence_path, prev_renderable_smpl=obj)
                            clicked, _ = imgui.menu_item("Set time offset", "", False, True)
                            if clicked:
                                self.dialogs.append(TimeOffsetDialogWindow(obj))
                            clicked, _ = imgui.menu_item("Delete", "", False, True)
                            if clicked:
                                self.main_controller.timed_renderables.remove(obj)
                                self.main_controller.scene.objects.remove(obj)
                                self.main_controller.smpl_models.remove(obj)
                        imgui.end_menu()
                    else:
                        if obj.tag.startswith("SMPL") and hasattr(obj, "last_overlay_color") and obj.last_overlay_color is not None:
                            obj.set_overlay_color((255, 165, 0, 0))
                            obj.last_overlay_color = None
                imgui.end_menu()
            else:
                for ind, obj in enumerate(filter(lambda x: isinstance(x, SMPLXModelBase),
                                                 self.main_controller.dynamic_objects)):
                    if obj.tag.startswith("SMPL") and hasattr(obj, "last_overlay_color") and obj.last_overlay_color is not None:
                        obj.set_overlay_color((255, 165, 0, 0))
                        obj.last_overlay_color = None

            if imgui.begin_menu("Object localization", self.main_controller.objloc_path is not None):
                clicked, _ = imgui.menu_item(f"{os.path.splitext(os.path.basename(self.main_controller.objloc_path))[0]}", "", False, False)
                clicked, _ = imgui.menu_item(f"Refresh", "", False, True)
                if clicked:
                    logger.info("Reloading object localization")
                    if self.main_controller.objloc_type == "single":
                        self.main_controller.add_task(self.main_controller.load_objloc, self.main_controller.objloc_path)
                    else:
                        self.main_controller.add_task(self.main_controller.load_multi_objloc, self.main_controller.objloc_path)
                imgui.end_menu()

            if imgui.begin_menu("Help", True):
                clicked, _ = imgui.menu_item("Controls", "Ctrl+H", False, True)
                if clicked:
                    self.dialogs.append(ControlsHelpWindow(self.dialogs))
                imgui.end_menu()
            imgui.end_main_menu_bar()

        if clicked_open_scene:
            logger.info("Opened 'open scene' dialog")
            self.dialogs.append(OpenDialogWindow(self.main_controller.config.scenes.scenes_dir,
                                                 lambda x: self.main_controller.add_task(self.main_controller.load_scene, x)))
        if clicked_load_smpl:
            logger.info("Opened 'load_smpl' dialog")
            self.dialogs.append(OpenDialogWindow(self.main_controller.config.humans.smpl_motions_dir,
                                                 lambda x: self.main_controller.add_task(self.main_controller.load_smpl_model, x)))

        for window in self.dialogs:
            window.build()
            if not window.active:
                self.dialogs.remove(window)
        self.build_main_text_bar(bar_height)
        return imgui.is_window_focused(imgui.FOCUS_ANY_WINDOW), imgui.is_window_hovered(imgui.HOVERED_ANY_WINDOW)

    def draw_gui(self):
        imgui.render()
        self.gui_renderer.render(imgui.get_draw_data())

    def handle_mousemotion(self):
        pressed_mouse = pg.mouse.get_pressed()
        if pressed_mouse[0]:
            if not self.mouse_tracking:
                self.mouse_pos = np.array(pg.mouse.get_pos(), dtype=np.float32)
                self.mouse_tracking = True
        else:
            if not self.mouse_capture:
                self.mouse_tracking = False

        if self.mouse_tracking:
            curr_mouse_pos = np.array(pg.mouse.get_pos(), dtype=np.float32)
            mouse_diff = curr_mouse_pos - self.mouse_pos
            self.mouse_pos = curr_mouse_pos
            if self.mouse_capture:
                pg.mouse.set_pos(*self.mouse_capture_pos)
                self.mouse_pos = self.mouse_capture_pos
            self.static_update = True
            self.main_controller.request_redraw()
            pressed_keys = pg.key.get_pressed()
            roll = 0
            if pressed_keys[pg.K_y]:
                roll = self.mouse_sensitivity
            if pressed_keys[pg.K_t]:
                roll = -self.mouse_sensitivity
            self.main_controller.control_camera_rot(mouse_diff, roll=roll)

    def handle_camera_keys(self):
        pressed_keys = pg.key.get_pressed()
        pressed_mods = pg.key.get_mods()
        move = np.zeros(3)
        if pressed_keys[pg.K_w]:
            move += np.array([0, 0, -self.movement_speed])
            self.static_update = True
            self.main_controller.request_redraw()
        if pressed_keys[pg.K_s]:
            move += np.array([0, 0, self.movement_speed])
            self.static_update = True
            self.main_controller.request_redraw()
        if pressed_keys[pg.K_a]:
            move += np.array([-self.movement_speed, 0, 0])
            self.static_update = True
            self.main_controller.request_redraw()
        if pressed_keys[pg.K_d]:
            move += np.array([self.movement_speed, 0, 0])
            self.static_update = True
            self.main_controller.request_redraw()
        if pressed_keys[pg.K_q]:
            move += np.array([0, -self.movement_speed, 0])
            self.static_update = True
            self.main_controller.request_redraw()
        if pressed_keys[pg.K_e]:
            move += np.array([0, self.movement_speed, 0])
            self.static_update = True
            self.main_controller.request_redraw()
        if pressed_mods & pg.KMOD_SHIFT:
            move *= 5.
        elif pressed_mods & pg.KMOD_CTRL:
            move /= 5.
        # logger.info(move)
        self.main_controller.control_camera_pos(move)

        update = False
        move = np.zeros(3)

        if pressed_keys[pg.K_LEFT]:
            move += np.array([0, -self.light_movement_speed, 0])
            update = True
        if pressed_keys[pg.K_RIGHT]:
            move += np.array([0, self.light_movement_speed, 0])
            update = True
        if pressed_keys[pg.K_DOWN]:
            move += np.array([-self.light_movement_speed, 0, 0])
            update = True
        if pressed_keys[pg.K_UP]:
            move += np.array([self.light_movement_speed, 0, 0])
            update = True
        if update:
            self.main_controller.request_redraw()
            self.main_controller.switch_camera_mode("free")
            rotvec = np.cross(self.main_controller.directional_light.direction, [0, 0, 1])
            rotvec = rotvec / np.linalg.norm(rotvec)
            rot = Rotation.from_euler('z', move[1]) * Rotation.from_rotvec(rotvec * move[0] / 2.)
            self.main_controller.directional_light.direction = \
                rot.apply(self.main_controller.directional_light.direction)
            self.main_controller.smpl_model_shadowmap_offset = -self.main_controller.directional_light.direction * 3

    def place_updatable_msg(self, status_line, prefix=""):
        for line in self.status_text:
            if prefix in line.text:
                line.text = status_line.text
                line.color = status_line.color
                line.refresh(status_line.time_limit)
                return
        self.status_text.append(status_line)

    def gui_events_main(self, event, is_any_item_focused, is_any_item_hovered):
        pressed_mods = pg.key.get_mods()
        self.main_controller.request_redraw()
        if event.type == pg.DROPFILE:
            logger.info(f"File dropped: {event.file}")
            path, ext = os.path.splitext(event.file)
            if ext == ".ply":
                self.main_controller.load_trimesh_object(event.file)
            else:
                logger.warning(f"Unknown extension: {ext}")
        elif event.type == pg.MOUSEWHEEL:
            if not imgui.is_window_hovered(imgui.HOVERED_ANY_WINDOW) and self.main_controller.camera.model == "perspective":
                self.main_controller.camera_fov = self.main_controller.camera_fov + event.y
                self.main_controller.camera.init_intrinsics(self.main_controller.viewport_resolution, fov=self.main_controller.camera_fov,
                                                            far=self.main_controller.far)
                text = f"Camera FOV: {self.main_controller.camera_fov:.2f} degrees"
                status_line = StatusTextLine(text, time_limit=4., color=(0, 0.8, 0.4))
                self.place_updatable_msg(status_line, "Camera FOV")
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                self.main_controller.animation_active = not self.main_controller.animation_active
            elif event.key == pg.K_LEFTBRACKET:
                self.main_controller.advance_global_time(-self.main_controller.time_scrolling_speed)
            elif event.key == pg.K_RIGHTBRACKET:
                self.main_controller.advance_global_time(self.main_controller.time_scrolling_speed)
            elif event.key == pg.K_MINUS:
                self.main_controller.time_scrolling_speed /= 2
                text = f"Scrolling speed: {self.main_controller.time_scrolling_speed:.2f} s"
                status_line = StatusTextLine(text, time_limit=4., color=(0, 0.8, 0.4))
                self.place_updatable_msg(status_line, "Scrolling speed")
            elif event.key == pg.K_EQUALS:
                self.main_controller.time_scrolling_speed *= 2
                text = f"Scrolling speed: {self.main_controller.time_scrolling_speed:.2f} s"
                status_line = StatusTextLine(text, time_limit=4., color=(0, 0.8, 0.4))
                self.place_updatable_msg(status_line, "Scrolling speed")

            if not is_any_item_focused:
                if event.key == pg.K_1:
                    if self.main_controller.main_renderable_pc is not None:
                        self.main_controller.main_renderable_pc.visible = not self.main_controller.main_renderable_pc.visible
                elif event.key == pg.K_2:
                    if self.main_controller.main_renderable_mesh is not None:
                        self.main_controller.main_renderable_mesh.visible = not self.main_controller.main_renderable_mesh.visible
                elif event.key == pg.K_3:
                    for kinect_renderer in self.main_controller.kinect_renderers:
                        if kinect_renderer is not None:
                            kinect_renderer.visible = not kinect_renderer.visible
                elif event.key == pg.K_4:
                    for renderable_smpl_model in self.main_controller.smpl_models:
                        renderable_smpl_model.visible = not renderable_smpl_model.visible
                elif event.key == pg.K_5:
                    if self.main_controller.main_renderable_obj_pc is not None:
                        self.main_controller.main_renderable_obj_pc.visible = not self.main_controller.main_renderable_obj_pc.visible
                elif event.key == pg.K_z:
                    self.main_controller.screen_capture_active = not self.main_controller.screen_capture_active
                    if pressed_mods & pg.KMOD_SHIFT:
                        self.main_controller.screen_capture_mode = 'screenshot'
                    else:
                        self.main_controller.screen_capture_mode = 'video'
                        self.main_controller.require_complete_rendering = self.main_controller.screen_capture_active
                        self.main_controller.time_mode = 'constant' if self.main_controller.screen_capture_active else 'real'
                        if pressed_mods & pg.KMOD_CTRL:
                            self.main_controller.animation_active = not self.main_controller.animation_active
                elif event.key == pg.K_TAB:
                    self.main_controller.reset_global_time()
                elif event.key == pg.K_p:
                    if self.main_controller.draw_pause:
                        self.status_text.remove(self.pause_info)
                    else:
                        self.status_text.append(self.pause_info)
                    self.main_controller.draw_pause = not self.main_controller.draw_pause
                elif event.key == pg.K_COMMA:
                    if self.main_controller.main_renderable_pc is not None:
                        self.main_controller.main_renderable_pc.set_splat_size(
                            self.main_controller.main_renderable_pc.get_splat_size() / 1.5)
                elif event.key == pg.K_PERIOD:
                    if self.main_controller.main_renderable_pc is not None:
                        self.main_controller.main_renderable_pc.set_splat_size(
                            self.main_controller.main_renderable_pc.get_splat_size() * 1.5)
                elif event.key == pg.K_h and (pressed_mods & pg.KMOD_CTRL):
                    self.dialogs.append(ControlsHelpWindow(self.dialogs))

    def process_ingame_controls(self, is_any_item_focused, is_any_item_hovered):
        if not is_any_item_focused:
            self.handle_camera_keys()
            if not is_any_item_hovered:
                self.handle_mousemotion()

    def switch_state(self, state, params):
        self.gui_state_stack.append((state, params))

    def return_to_prev_state(self):
        if len(self.gui_state_stack) > 1:
            self.gui_state_stack = self.gui_state_stack[:-1]

    def process_event(self, event, is_any_item_focused, is_any_item_hovered):
        self.gui_renderer.process_event(event)
        curr_state, curr_params = self.gui_state_stack[-1]
        self.gui_state_handlers[curr_state](event, is_any_item_focused, is_any_item_hovered)


class TrajectoryEditDialog:
    AVAILABLE_INTERP_TYPES = ["linear", "quadratic", "cubic"]

    def __init__(self, main_controller, gui_controller: GuiController, trajectory=None):
        self.controller = main_controller
        self.gui_controller = gui_controller
        self.active = True
        self.smoothness = 5.
        self.interp_type = "quadratic"
        self.trajectory = trajectory
        self.current_trajectory_path, self.new_trajectory_path = self.get_new_trajpaths(2)
        self.selected_camera = 0

        if self.trajectory is None:
            self.trajectory = []
            self.add_curr_camera()

        self.controller.switch_camera_mode("free")

    def get_new_trajpaths(self, count=2):
        maxind = 0
        for path in glob(os.path.join(self.controller.config.capturing.trajectory_dir, "camerapath.*.json")):
            name = os.path.basename(path)
            try:
                ind = int(name.split('.')[-2])
            except ValueError as e:
                pass
            else:
                maxind = max(maxind, ind)
        res = []
        for i in range(count):
            res.append(os.path.join(self.controller.config.capturing.trajectory_dir, f"camerapath.{maxind + i + 1:03d}.json"))
        if count == 1:
            return res[0]
        return res

    def update_text(self, text):
        self.gui_controller.status_text.append(StatusTextLine(text, time_limit=2, color=(1, 0, 0)))
        self.controller.raw_trajectory = self.trajectory

    def sort_trajectory(self):
        times = [x['time'] for x in self.trajectory]
        ind_sorted = np.argsort(times)
        self.trajectory = [self.trajectory[k] for k in ind_sorted]
        self.selected_camera = np.flatnonzero(ind_sorted == self.selected_camera)[0]

    def save_current_camera(self, check_time=True):
        pos = self.controller.camera_pos.copy()
        quat = np.roll(self.controller.camera_rot.as_quat(), 1).copy()
        time = np.array(self.controller.global_time).copy()
        if check_time:
            times = np.array([x['time'] for i, x in enumerate(self.trajectory) if i != self.selected_camera])
            diff = times - time
            if np.any(np.abs(diff) < 1e-4):
                self.update_text("Camera wasn't saved (too close to other camera)")
                return
        camera = {'quaternion': quat,
                  'position': pos,
                  'time': time}
        self.trajectory[self.selected_camera] = camera
        self.sort_trajectory()
        self.update_text("Saved current camera")

    def delete_current_camera(self):
        if len(self.trajectory) == 1:
            return
        del self.trajectory[self.selected_camera]
        if self.selected_camera >= len(self.trajectory):
            self.selected_camera = len(self.trajectory) - 1
        camera = self.trajectory[self.selected_camera]
        self.apply_camera(camera)
        self.update_text("Deleted current camera")

    def save_trajectory(self, fname):
        s_traj = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in x.items()} for x in self.trajectory]
        json.dump(s_traj, open(fname, 'w'))
        self.current_trajectory_path = fname
        self.new_trajectory_path = self.get_new_trajpaths(1)

    def load_trajectory(self, fname):
        s_traj = json.load(open(fname))
        self.trajectory = [{k: np.array(v) for k, v in x.items()} for x in s_traj]
        self.selected_camera = 0
        camera = self.trajectory[self.selected_camera]
        self.apply_camera(camera)
        self.current_trajectory_path = fname
        self.new_trajectory_path = self.get_new_trajpaths(1)

    def add_curr_camera(self):
        times = np.array([x['time'] for x in self.trajectory])
        diff = times - self.controller.global_time
        if np.any(np.abs(diff) < 1e-4):
            self.update_text("Camera wasn't added (too close to existing camera)")
            return
        self.trajectory = self.trajectory[:self.selected_camera] + [None] + self.trajectory[self.selected_camera:]
        self.save_current_camera(check_time=False)
        self.update_text("\nAdded new camera")

    def rot_gaussian_smoothing(self, rots, sigma=5.):
        def get_rot_ind(ind):
            while ind >= len(rots) or ind < 0:
                if ind >= len(rots):
                    ind = 2 * len(rots) - 1 - ind
                if ind < 0:
                    ind = -ind
            return ind

        winradius = round(2 * 3 * sigma)
        if winradius < 1:
            return rots
        weights = gaussian(winradius * 2 + 1, sigma)
        res = []
        for ind in range(len(rots)):
            window_inds = [get_rot_ind(i) for i in range(ind - winradius, ind + winradius + 1)]
            res.append(rots[window_inds].mean(weights))
        return res

    def refine_trajectory(self, time_step=1 / 60.):
        if len(self.trajectory) < 2:
            return self.trajectory
        start_time = self.trajectory[0]['time']
        end_time = self.trajectory[-1]['time']
        # t_ind = 0
        # prev_cam = self.trajectory[t_ind]
        # next_cam = self.trajectory[t_ind + 1]
        cam_times = [x['time'] for x in self.trajectory]
        cam_rots = Rotation.from_quat([np.roll(x['quaternion'], -1) for x in self.trajectory])
        # print(cam_rots)
        cam_poses = [x['position'] for x in self.trajectory]
        rot_slerp = Slerp(cam_times, cam_rots)
        interp_times = np.concatenate([np.arange(start_time, end_time, time_step), [end_time]])
        interp_rots = rot_slerp(interp_times)
        # interp_quats = [np.roll(x.as_quat(),1) for x in interp_rots]
        # interp_rotvecs = interp_rots.as_rotvec()

        pos_intrp = interp1d(cam_times, cam_poses, axis=0, kind=self.interp_type)
        interp_poses = pos_intrp(interp_times)
        # spl_list = [splrep(cam_times, cp, k=2, s=len(cp)-np.sqrt(len(cp)*2)) for cp in zip(*cam_poses)]
        # interp_poses = list(zip(*[splev(interp_times, spl) for spl in spl_list]))

        interp_poses = np.array(list(zip(*[gaussian_filter1d(x, self.smoothness) for x in zip(*interp_poses)])))
        # interp_quats = [-q if q[0]<0 else q for q in interp_quats]
        # interp_quats_abs = np.array(list(zip(*[gaussian_filter1d(x, self.smoothness) for x in zip(*np.abs(interp_quats))])))
        # interp_quats = np.sign(interp_quats) * interp_quats_abs
        # interp_quats/=np.linalg.norm(interp_quats, axis=1, keepdims=True)
        # interp_rotvecs = np.array(list(zip(*[gaussian_filter1d(x, self.smoothness) for x in zip(*interp_rotvecs)])))
        # interp_quats = Rotation.from_rotvec(interp_rotvecs).as_quat()

        interp_rots = self.rot_gaussian_smoothing(interp_rots, self.smoothness)
        interp_quats = [np.roll(x.as_quat(), 1) for x in interp_rots]

        interp_traj = [{'position': interp_poses[i], 'quaternion': interp_quats[i], 'time': interp_times[i]}
                       for i in range(len(interp_times))]
        return interp_traj

    def apply_camera(self, camera):
        self.controller.camera_rot = Rotation.from_quat(np.roll(camera['quaternion'].copy(), -1))
        self.controller.camera_pos = camera['position'].copy()
        self.controller.camera_yaw_pitch = (self.controller.camera_rot * Rotation.from_euler('x',
                                                                                             -90, degrees=True)).as_euler('xyz')[[2, 0]]
        self.controller.reset_global_time(camera['time'])
        self.controller.apply_current_camera()

    def run_trajectory(self, record=False):
        self.controller.raw_trajectory = self.trajectory
        ref_traj = self.refine_trajectory()
        self.apply_camera(ref_traj[0])
        self.controller.trajectory = ref_traj
        self.controller.camera_modes_available.add("trajectory")
        self.controller.switch_camera_mode('trajectory')
        self.controller.animation_active = True
        if record:
            self.controller.screen_capture_active = True
            self.controller.require_complete_rendering = True
            self.controller.time_mode = 'constant'
            self.controller.screen_capture_mode = "video"
        # self.controller.gui_controller.reset_state()

    def build(self):
        if self.active:
            expanded, opened = imgui.begin("Trajectory edit", True)
            if opened:
                if imgui.button("Add"):
                    self.add_curr_camera()
                imgui.same_line()
                if imgui.button("Replace"):
                    self.save_current_camera()
                imgui.same_line()
                if imgui.button("Remove"):
                    self.delete_current_camera()
                imgui.same_line()
                if imgui.button("Run"):
                    self.run_trajectory(record=False)
                imgui.same_line()
                if imgui.button("Run & record"):
                    self.run_trajectory(record=True)
                imgui.same_line()
                imgui.push_item_width(50)
                clicked, current = imgui.combo(
                    "Interpolation", self.AVAILABLE_INTERP_TYPES.index(self.interp_type), self.AVAILABLE_INTERP_TYPES
                )
                imgui.pop_item_width()
                if clicked:
                    self.interp_type = self.AVAILABLE_INTERP_TYPES[current]
                if imgui.button(f"Load trajectory"):
                    logger.info("Opened 'Load trajectory' dialog")
                    self.gui_controller.dialogs.append(OpenDialogWindow(self.controller.config.capturing.trajectory_dir, self.load_trajectory))
                imgui.same_line()
                if imgui.button(f"Save to {os.path.basename(self.current_trajectory_path)}"):
                    self.save_trajectory(self.current_trajectory_path)
                imgui.same_line()
                if imgui.button(f"Save to {os.path.basename(self.new_trajectory_path)}"):
                    self.save_trajectory(self.new_trajectory_path)

                imgui.push_item_width(-1)
                imgui.text(f"Trajectory: {len(self.trajectory)} items")
                traj_desc = [f"{i}) {c['time']:.2f} at {c['position']}" for i, c in enumerate(self.trajectory)]
                clicked, selected_camera = imgui.listbox("", self.selected_camera,
                                                         traj_desc, height_in_items=len(traj_desc))
                if clicked:
                    self.selected_camera = selected_camera
                    self.apply_camera(self.trajectory[self.selected_camera])
            else:
                self.active = False
            imgui.end()
