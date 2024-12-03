import numpy as np
import json
import os
from scipy.spatial.transform import Rotation

def compute_diff(times_master, times_sub, off):
    t1 = np.array(times_master)
    t2 = np.array(times_sub)
    if off<0:
        t1 = t1[-off:]
    else:
        t2 = t2[off:]
    minlen = min(t1.shape[0], t2.shape[0], 3)
    t1 = t1[:minlen]
    t2 = t2[:minlen]
    return np.abs(t1-t2).sum()/minlen

def find_off(times_master, times_sub, maxoff = 40):
    minlen = min(len(times_master), len(times_sub), maxoff)
    c = [compute_diff(times_master, times_sub, off) for off in range(-minlen,minlen)]
    return np.argmin(c)-minlen

class KinectMulticameraSystem:
    def __init__(self, camera_count = 3, config_path = None):
        self.camera_count = camera_count
        self.configs = [self.generate_default_config() for _ in range(camera_count)]
        self.global_config = {"time_offset":0.}
        if config_path is not None:
            config = json.load(open(config_path))

            for i, x in enumerate(self.configs):
                x.update({k:np.array(v) if isinstance(v, (tuple,list)) else v for k,v in config[str(i)].items()})
            if "global" in config:
                self.global_config.update(config['global'])

    def write_config(self, config_path):
        s_dict = {"global":{"time_offset":float(self.global_config["time_offset"])}}
        for i in range(self.camera_count):
            s_dict[i] = {"frame_offset":int(self.configs[i]["frame_offset"]),
                         "quaternion":self.configs[i]["quaternion"].tolist(),
                         "position":self.configs[i]["position"].tolist()}
        json.dump(s_dict, open(config_path, 'w'))

    def read_extrinsics(self, path):
        extrinsics = json.load(path)


    def write_extrinsics(self, outpath):
        s_dict = {}
        for i in range(self.camera_count):
            s_dict[i] = {"quaternion":self.configs[i]["quaternion"].tolist(),
                         "position":self.configs[i]["position"].tolist()}
        json.dump(s_dict, open(outpath, 'w'))

    def write_global_time(self, outpath):
        s_dict = {"global": {"time_offset": float(self.global_config["time_offset"])}}
        json.dump(s_dict, open(outpath, 'w'))

    @staticmethod
    def generate_default_config():
        config = {"frame_offset":0,
                  "quaternion":np.roll(Rotation.from_euler('x', -90, degrees=True).as_quat(), 1),
                  "position":np.zeros(3)}
        return config

    def fix_time_sync(self):
        offsets = [x['frame_offset'] for x in self.configs]
        min_off = min(offsets)
        if min_off < 0:
            for off, x in zip(offsets, self.configs):
                x['frame_offset'] -= min_off

    def get_time_offset(self):
        return self.global_config['time_offset']

    def set_time_offset(self, offset):
        self.global_config['time_offset'] = offset

    def get_location(self, kinect_ind):
        return self.configs[kinect_ind]['quaternion'], self.configs[kinect_ind]['position']

    def reset_location(self, kinect_ind):
        self.configs[kinect_ind].update({
            "quaternion":np.roll(Rotation.from_euler('x', -90, degrees=True).as_quat(), 1),
            "position":np.zeros(3)})

    def update_location(self, kinect_ind, rotate, move):
        quat, pose = self.get_location(kinect_ind)
        quat = np.roll((Rotation.from_quat(np.roll(quat, -1)) * rotate).as_quat(), 1)
        pose = pose + move
        self.set_location(kinect_ind, quat, pose)

    def set_location(self, kinect_ind, quat, pose):
        self.configs[kinect_ind]['quaternion'] = quat
        self.configs[kinect_ind]['position'] = pose

    def frame_autosync(self, times):
        master_time = times[0]
        subs_time = times[1:]
        self.configs[0]['frame_offset'] = 0
        for config, sub_time in zip(self.configs[1:], subs_time):
            config['frame_offset'] = find_off(master_time, sub_time)
        self.fix_time_sync()

    def _apply_loc_file(self, config, loc_file):
        locs = json.load(open(loc_file))
        quats, poses = [], []
        for ind, loc in sorted(locs.items()):
            pose = loc['position']
            quat = loc['quaternion']
            poses.append(pose)
            quats.append(quat)
        poses = np.array(poses)
        quats = np.array(quats)

        poses_dists = ((poses.reshape((1,-1,3))-poses.reshape((-1,1,3)))**2).sum(axis=-1)
        avg_pose_dist = poses_dists.mean(axis=1)
        median_avg_pose_dist = np.median(avg_pose_dist)
        mask = avg_pose_dist<median_avg_pose_dist
        poses = poses[mask]
        quats = quats[mask]

        pose = poses.mean(axis=0)
        quat = quats.mean(axis=0)
        config['quaternion'] = quat
        config['position'] = pose

    def apply_loc_files(self, loc_files):
        for config, loc_file in zip(self.configs, loc_files):
            self._apply_loc_file(config, loc_file)


class KinrecSequenceController:
    standard_aliases = {
        "000960615212": 1,
        "000834615212": 2,
        "001001515212": 3,
        "000903215212": 4
    }
    def __init__(self, metadata_path, extrinsics_path = None, present_in_colorspace=True):
        metadata = json.load(open(metadata_path))
        self.participating_kinects = list(metadata["participating_kinects"].keys())
        self.camera_count = len(self.participating_kinects)
        self.camera_extrinsics = {k: self.generate_eye_extrinsics() for k in self.participating_kinects}
        self.kinect_aliases = self.get_kinect_aliases_from_metadata(metadata)
        self.global_time_offset = 0.
        if extrinsics_path is not None:
            extrinsics_dict = json.load(open(extrinsics_path))
            if 'version' in extrinsics_dict:
                correction_rot = Rotation.from_euler('x', -90, degrees=True)
                extrinsics_dict = extrinsics_dict['extrinsics']
                for kinect_id, kinect_extrinsics in self.camera_extrinsics.items():
                    kinect_extrinsics_dict = extrinsics_dict[kinect_id]
                    if "depth" not in kinect_extrinsics_dict:
                        color_rot = Rotation.from_matrix(kinect_extrinsics_dict["color"]["R"])
                        color_tr = np.array(kinect_extrinsics_dict["color"]["t"])
                        color2depth_rot = Rotation.from_matrix(metadata["participating_kinects"][kinect_id]["color2depth"]["R"])
                        color2depth_tr = np.array(metadata["participating_kinects"][kinect_id]["color2depth"]["t"])
                        depth2color_rot = Rotation.from_matrix(metadata["participating_kinects"][kinect_id]["depth2color"]["R"])
                        depth2color_tr = np.array(metadata["participating_kinects"][kinect_id]["depth2color"]["t"])
                        depth_rot = color2depth_rot*color_rot*depth2color_rot
                        depth_tr = color2depth_rot.apply(color_tr + color_rot.apply(depth2color_tr)) + color2depth_tr
                    else:
                        depth_rot = Rotation.from_matrix(kinect_extrinsics_dict["depth"]["R"])
                        depth_tr = np.array(kinect_extrinsics_dict["depth"]["t"])
                    if present_in_colorspace:
                        depth2color_rot = Rotation.from_matrix(metadata["participating_kinects"][kinect_id]["depth2color"]["R"])
                        depth2color_tr = np.array(metadata["participating_kinects"][kinect_id]["depth2color"]["t"])
                        rot = depth2color_rot*depth_rot
                        tr = depth2color_rot.apply(depth_tr) + depth2color_tr
                    else:
                        rot = depth_rot
                        tr = depth_tr
                    quat = np.roll((correction_rot*rot).as_quat(), 1)
                    translation = correction_rot.apply(tr)
                    kinect_extrinsics.update({"quaternion":quat, "position":translation})
            else:
                for kinect_id, kinect_extrinsics in self.camera_extrinsics.items():
                    kinect_extrinsics.update({k:np.array(v) if isinstance(v, (tuple,list)) else v for k,v in extrinsics_dict[kinect_id].items()})

    @staticmethod
    def get_kinect_aliases_from_metadata(metadata:dict):
        kinect_ids = list(metadata["participating_kinects"].keys())
        alias_dict = {}
        for kinect_id in kinect_ids:
            alias = None
            if "alias" in metadata["participating_kinects"][kinect_id]:
                alias = metadata["participating_kinects"][kinect_id]["alias"]
            else:
                if kinect_id in KinrecSequenceController.standard_aliases:
                    alias = KinrecSequenceController.standard_aliases[kinect_id]
            alias_dict[kinect_id] = alias
        return alias_dict

    @staticmethod
    def generate_eye_extrinsics():
        return {"quaternion": np.roll(Rotation.from_euler('x', -90, degrees=True).as_quat(), 1),
        "position": np.zeros(3)}

    def write_extrinsics(self, outpath):
        s_dict = {}
        for kinect_id in self.participating_kinects:
            s_dict[kinect_id] = {"quaternion":self.camera_extrinsics[kinect_id]["quaternion"].tolist(),
                         "position":self.camera_extrinsics[kinect_id]["position"].tolist()}
        json.dump(s_dict, open(outpath, 'w'))

    def write_global_time(self, outpath):
        s_dict = {"global": {"time_offset": float(self.global_time_offset)}}
        json.dump(s_dict, open(outpath, 'w'))

    def get_time_offset(self):
        return self.global_time_offset

    def set_time_offset(self, offset:float):
        self.global_time_offset = offset

    def get_location(self, kinect_id: str):
        """
        Gets location for kinect's Depth camera in the world coordinates
        Args:
            kinect_id (str): kinect id
        Returns:
            tuple[np.ndarray, np.ndarray]: quaternion and translation
        """
        return self.camera_extrinsics[kinect_id]['quaternion'], self.camera_extrinsics[kinect_id]['position']

    def reset_location(self, kinect_id):
        self.camera_extrinsics[kinect_id].update({
            "quaternion":np.roll(Rotation.from_euler('x', -90, degrees=True).as_quat(), 1),
            "position":np.zeros(3)})

    def get_kinect_alias(self,kinect_id):
        return self.kinect_aliases[kinect_id]

    def update_location(self, kinect_id, rotate, move):
        quat, pose = self.get_location(kinect_id)
        quat = np.roll((Rotation.from_quat(np.roll(quat, -1)) * rotate).as_quat(), 1)
        pose = pose + move
        self.set_location(kinect_id, quat, pose)

    def set_location(self, kinect_id, quat, pose):
        self.camera_extrinsics[kinect_id]['quaternion'] = quat
        self.camera_extrinsics[kinect_id]['position'] = pose

    def frame_autosync(self, times):
        master_time = times[0]
        subs_time = times[1:]
        self.configs[0]['frame_offset'] = 0
        for config, sub_time in zip(self.configs[1:], subs_time):
            config['frame_offset'] = find_off(master_time, sub_time)
        self.fix_time_sync()

    def _apply_loc_file(self, config, loc_file):
        locs = json.load(open(loc_file))
        quats, poses = [], []
        for ind, loc in sorted(locs.items()):
            pose = loc['position']
            quat = loc['quaternion']
            poses.append(pose)
            quats.append(quat)
        poses = np.array(poses)
        quats = np.array(quats)

        poses_dists = ((poses.reshape((1,-1,3))-poses.reshape((-1,1,3)))**2).sum(axis=-1)
        avg_pose_dist = poses_dists.mean(axis=1)
        median_avg_pose_dist = np.median(avg_pose_dist)
        mask = avg_pose_dist<median_avg_pose_dist
        poses = poses[mask]
        quats = quats[mask]

        pose = poses.mean(axis=0)
        quat = quats.mean(axis=0)
        config['quaternion'] = quat
        config['position'] = pose

    def apply_loc_files(self, loc_files):
        for config, loc_file in zip(self.camera_extrinsics, loc_files):
            self._apply_loc_file(config, loc_file)


