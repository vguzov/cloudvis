from cloudvis.config import get_config

from cloudvis.visualizer import InteractiveViewer

config = get_config()

if config.capturing.screen_capture_dir is not None:
    config.capturing.screen_capture_dir.mkdir(parents=True, exist_ok=True)
if config.capturing.trajectory_dir is not None:
    config.capturing.trajectory_dir.mkdir(parents=True, exist_ok=True)

viewer = InteractiveViewer(config)
viewer.main_loop()
