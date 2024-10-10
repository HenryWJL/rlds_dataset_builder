import os
import av
import numpy as np
from pathlib import Path
import scipy.interpolate as si
from scipy.spatial.transform import Rotation

# img = np.fromfile('data/20240927_000403/Depth/depth_0.000.bin', dtype='uint16')

# csv = np.loadtxt("data/test/pick cups/20240927_000403/PoseData.csv", delimiter=',', skiprows=1)
# angle = np.ones((565, 2), dtype=np.float32)
# angle[:, 0] = csv[:, 0]
# np.savetxt("data/test/pick cups/20240927_000403/AngleData.csv", angle, delimiter=",", fmt="%f")


class FormatConverter:

    def __init__(
        self,
        episode_dir: str,
        interpolate_type: str
    ) -> None:
        self.episode_dir = Path(os.path.expanduser(episode_dir)).absolute()
        self.interpolate_type = interpolate_type

    @staticmethod
    def fit_interpolator(
        timestamps: np.ndarray,
        values: np.ndarray,
        type: str
    ) -> si.interp1d:
        return si.interp1d(
            x=timestamps,
            y=values,
            kind=type,
            axis=0,
            bounds_error=False,
            fill_value=(values[0], values[-1])
        )

    def run(self) -> None:
        save_dir = self.episode_dir.joinpath("train")
        if not save_dir.is_dir():
            save_dir.mkdir()
        global_step = 0
        for mp4_path in list(self.episode_dir.glob("**/*.mp4")):
            episode_dir = mp4_path.parent
            # Language instruction
            #TODO we set the name of the directory to be the language instruction
            language_instruction = episode_dir.parent.name
            # RGB observation
            rgb = list()
            with av.open(str(mp4_path)) as container:
                for frame in container.decode(video=0):
                    frame = frame.to_ndarray(format="bgr24")
                    rgb.append(frame)
            episode_len = len(rgb)
            # Depth observation
            depth = list()
            for depth_path in list(episode_dir.joinpath("Depth").glob("*.bin")):
                depth.append(np.fromfile(str(depth_path), dtype="uint16").reshape(192, 256) / 1e4)
            assert len(depth) == episode_len
            # 6D pose
            pose = np.loadtxt(
                str(episode_dir.joinpath("PoseData.csv")),
                delimiter=",",
                skiprows=1,
                dtype=np.float32
            )
            assert pose.shape[0] == episode_len
            # Converting to euler angle
            mat = pose[:, 1:].reshape(-1, 4, 4).transpose(0, 2, 1)
            trans = mat[:, :3, 3]
            rot = Rotation.from_matrix(mat[:, :3, :3]).as_euler("xyz", degrees=True)
            pose = np.concatenate([trans, rot], axis=-1)
            # Gripper closure
            closure = np.loadtxt(
                str(episode_dir.joinpath("AngleData.csv")),
                delimiter=",",
                skiprows=1,
                dtype=np.float32
            )
            assert closure.shape[0] == episode_len
            # Action
            action = np.concatenate([pose, closure], axis=-1)
            # Force and torque
            force_l = np.loadtxt(
                str(episode_dir.joinpath("L_ForceData.csv")),
                delimiter=",",
                skiprows=1,
                dtype=np.float32
            )
            force_r = np.loadtxt(
                str(episode_dir.joinpath("R_ForceData.csv")),
                delimiter=",",
                skiprows=1,
                dtype=np.float32
            )
            force_l_interp = self.fit_interpolator(
                timestamps=force_l[:, 0],
                values=force_l[:, 1:],
                type=self.interpolate_type
            )
            force_r_interp = self.fit_interpolator(
                timestamps=force_r[:, 0],
                values=force_r[:, 1:],
                type=self.interpolate_type
            )
            force_l = force_l_interp(pose[:, 0])
            force_r = force_r_interp(pose[:, 0])
            # Saving data in a new format
            episode = list()
            for step in range(episode_len):
                episode.append({
                    'rgb': rgb[step],
                    'depth': depth[step],
                    'pose': pose[step],
                    'force_l': force_l[step],
                    'force_r': force_r[step],
                    'action': action[step],
                    'language_instruction': language_instruction
                })
            global_step += 1
            np.save(str(save_dir.joinpath(str(global_step).zfill(3))), episode)
            

if __name__ == "__main__":
    fc = FormatConverter(
        episode_dir="data/test",
        interpolate_type="nearest"
    )
    fc.run()
    
    

# t = np.array([0., 0.056, 0.124, 0.178, 0.218])
# y = np.array([[3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
# t_new = np.array([0., 0.033, 0.067, 0.1, 0.133, 0.167, 0.2, 0.233])
# interp = si.interp1d(
#     t, y, kind='nearest',
#     axis=0, bounds_error=False, 
#     fill_value=(y[0], y[-1])
# )
# y_new = interp(t_new)
# print(y_new)

# frames = []
# with av.open("data/20240927_000403/RGB.mp4") as container:
#     for frame in container.decode(video=0):
#     # Decode video frame, and convert to NumPy array in BGR pixel format (use BGR because it used by OpenCV).
#         frame = frame.to_ndarray(format='bgr24')
#         frames.append(frame)
# print(len(frames))

# import numpy as np
# import tqdm
# import os

# N_TRAIN_EPISODES = 100
# N_VAL_EPISODES = 100

# EPISODE_LENGTH = 10


# def create_fake_episode(path):
#     episode = []
#     for step in range(EPISODE_LENGTH):
#         episode.append({
#             'rgb': np.asarray(np.random.rand(480, 640, 3) * 255, dtype=np.uint8),
#             'depth': np.asarray(np.random.rand(192, 256), dtype=np.float32),
#             'pose': np.asarray(np.random.rand(6), dtype=np.float32),
#             'force_l': np.asarray(np.random.rand(6), dtype=np.float32),
#             'force_r': np.asarray(np.random.rand(6), dtype=np.float32),
#             'action': np.asarray(np.random.rand(7), dtype=np.float32),
#             'language_instruction': 'dummy instruction',
#         })
#     np.save(path, episode)


# # create fake episodes for train and validation
# print("Generating train examples...")
# os.makedirs('data/train', exist_ok=True)
# for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
#     create_fake_episode(f'data/train/episode_{i}.npy')

# print("Generating val examples...")
# os.makedirs('data/val', exist_ok=True)
# for i in tqdm.tqdm(range(N_VAL_EPISODES)):
#     create_fake_episode(f'data/val/episode_{i}.npy')

# print('Successfully created example data!')
