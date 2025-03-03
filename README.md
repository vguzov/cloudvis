# CloudVis - Interactive Visualizer for Point Clouds, Meshes and more
An interactive pointcloud and mesh visualizer with recording capabilities.
Based on [cloudrender](https://github.com/vguzov/cloudrender) rendering framework.

## Main features:
 - Fast visualization of large pointclouds and meshes;
 - SMPL body visualization in motion;
 - Dynamic object animations;
 - Video recording with smooth camera trajectories;
 - Synchronized playback with the video


## Installation
```bash
pip install git+https://github.com/vguzov/cloudvis
```
**Note**: Tested on Ubuntu 20.04, 22.04 and 24.04 and MacOS 15; compatibility with Windows is not guaranteed.

## Usage
Cloudvis stores all the parameters in the config file. You can create your own config by following the structure of default one in [configs/default.toml](configs/default.toml).
After that, run the visualizer with the config as follows:
```bash
python runvis.py -c <path to config> [--any params you want to overwrite, e.g. --smpl_poses_dir <path>]
```

Within the window, you can look up the controls by pressing `Ctrl+H` or `Help->Controls` in the menu.




## Citation

If you use this code, please cite our paper:

```
@inproceedings{guzov24ireplica,
    title = {Interaction Replica: Tracking human–object interaction and scene changes from human motion},
    author = {Guzov, Vladimir and Chibane, Julian and Marin, Riccardo and He, Yannan and Saracoglu, Yunus and Sattler, Torsten and Pons-Moll, Gerard},
    booktitle = {International Conference on 3D Vision (3DV)},
    month = {March},
    year = {2024},
}
```