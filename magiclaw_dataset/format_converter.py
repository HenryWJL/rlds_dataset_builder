import os
import av
import csv
import json
import numpy as np
from pathlib import Path
from typing import Optional, Union
import scipy.interpolate as si
from scipy.spatial.transform import Rotation as R


class FormatConverter:

    def __init__(
        self,
        episode_dir: str,
        save_dir: Optional[str] = "data",
        split: Optional[str] = 'train',
        interp_type: Optional[str] = 'nearest'
    ) -> None:
        """
        Args:
            episode_dir (str): directory used for loading episodes.
            save_dir (str): directory used for saving episodes.
            split (str): dataset split.
                - 'train'
                - 'val'
            interp_type (str): interpolation type.
                - 'nearest'
                - 'linear'
                - 'zero'
                - 'slinear'
                - 'quadratic'
                - 'cubic'
        """
        self.episode_dir = Path(os.path.expanduser(episode_dir)).absolute()
        self.save_dir = Path(os.path.expanduser(save_dir)).absolute().joinpath(split)
        if not self.save_dir.is_dir():
            self.save_dir.mkdir(parents=True)
        self.interp_type = interp_type

    @staticmethod
    def load_csv(path: Union[str, Path]) -> np.ndarray:
        """
        Load CSV files into numpy arrays.

        Returns:
            data (numpy.ndarray): a (D+1)-dimensional array (N, D+1). The first
                dimension corresponds to timestamps, and the last D dimensions
                correspond to data features.
        """
        if isinstance(path, str):
            path = Path(os.path.expanduser(path)).absolute()
        data = list()
        with open(str(path), newline='') as file:
            for line in csv.reader(file):
                if all(line):
                    # Make sure empty data is not loaded
                    data.append(line)
            file.close()
        data.pop(0)  # Remove headers
        data = np.stack(data).astype(np.float32)
        return data

    def fit_interpolator(self, data: np.ndarray) -> si.interp1d:
        """
        Fit an interpolator on @data.

        Args:
            data (numpy.ndarray): a (D+1)-dimensional array (N, D+1). The first
                dimension corresponds to timestamps, and the last D dimensions
                correspond to data features.
        """
        return si.interp1d(
            x=data[:, 0],
            y=data[:, 1:],
            kind=self.interp_type,
            axis=0,
            bounds_error=False,
            fill_value=(data[0, 1:], data[-1, 1:])
        )

    def run(self) -> None:
        global_step = 0
        for mp4_path in list(self.episode_dir.glob("**/*.mp4")):
            episode_dir = mp4_path.parent
            # Language instruction
            with open(str(episode_dir.joinpath("metadata.json"))) as file:
                metadata = json.load(file)
                file.close()
            language_instruction = metadata['description']
            # RGB observation
            rgb = list()
            rgb_len = 0
            with av.open(str(mp4_path)) as container:
                for frame in container.decode(video=0):
                    frame = frame.to_ndarray(format='rgb24')
                    rgb.append(frame)
                    rgb_len += 1
            # Depth observation
            depth_ts = list()
            depth = list()
            for depth_path in list(episode_dir.glob("Depth/*.bin")):
                d_ts = np.array([depth_path.name[6: -4]]).astype(float)
                d = np.fromfile(str(depth_path), dtype='uint16').reshape(192, 256) / 1e4
                depth_ts.append(d_ts)
                depth.append(d.astype(np.float32))    
            depth_len = len(depth_ts)
            depth_ts, depth = zip(*sorted(zip(depth_ts, depth)))
            depth_ts = np.concatenate(depth_ts)
            # 6D pose
            pose = self.load_csv(episode_dir.joinpath("PoseData.csv"))
            mat = pose[:, 1:].reshape(-1, 4, 4).transpose(0, 2, 1)
            trans = mat[:, :3, 3]
            rot = R.from_matrix(mat[:, :3, :3]).as_euler('xyz').astype(np.float32)
            pose = np.concatenate([trans, rot], axis=-1)
            pose_len = pose.shape[0]
            # Gripper closure
            closure = self.load_csv(episode_dir.joinpath("AngleData.csv"))
            closure = closure[:, 1][:, np.newaxis]
            closure_len = closure.shape[0]
            # Force and torque
            force_l = self.load_csv(episode_dir.joinpath("L_ForceData.csv"))
            force_r = self.load_csv(episode_dir.joinpath("R_ForceData.csv"))
            # We assume data collected by iPhone to have the same length. If not,
            # the episode would be discarded and would not appear in the dataset.
            if not rgb_len == depth_len == pose_len == closure_len:
                print(f"WARNING: episode {episode_dir.name} is discarded.")
                continue

            forces = [force_l, force_r]
            for i, force in enumerate(forces):
                interp = self.fit_interpolator(force)
                forces[i] = interp(depth_ts)

            episode = list()
            for step in range(rgb_len):
                episode.append({
                    'rgb': rgb[step],
                    'depth': depth[step],
                    'pose': pose[step],
                    'force_l': forces[0][step],
                    'force_r': forces[1][step],
                    'action': np.concatenate((pose[step], closure[step])),
                    'language_instruction': language_instruction
                })

            # # Align data from different sensors. This is done via interpolation.
            # data = dict(
            #     pose=pose,
            #     closure=closure,
            #     force_l=force_l,
            #     force_r=force_r
            # )
            # for key, val in data.items():
            #     interp = self.fit_interpolator(val, self.interp_type)
            #     data[key] = interp(depth_ts)
                
            # episode = list()
            # for step in range(episode_len):
            #     episode.append({
            #         'rgb': rgb[step],
            #         'depth': depth[step],
            #         'pose': data['pose'][step],
            #         'force_l': data['force_l'][step],
            #         'force_r': data['force_r'][step],
            #         'action': np.concatenate((data['pose'][step], data['closure'][step]), axis=-1),
            #         'language_instruction': language_instruction
            #     })

            global_step += 1
            np.save(str(self.save_dir.joinpath(f"episode_{str(global_step).zfill(3)}.npy")), episode)




# import os
# import av
# import csv
# import json
# import numpy as np
# from pathlib import Path
# from typing import Optional, Union
# import scipy.interpolate as si
# from scipy.spatial.transform import R


# class FormatConverter:

#     def __init__(
#         self,
#         episode_dir: str,
#         save_dir: Union[str, None] = None,
#         split: Optional[str] = "train",
#         interp_type: Optional[str] = "nearest"
#     ) -> None:
#         self.episode_dir = Path(os.path.expanduser(episode_dir)).absolute()
#         self.save_dir = Path(os.path.expanduser(save_dir)).absolute() \
#             if save_dir is not None else Path.cwd().joinpath("data").joinpath(split)
#         if not self.save_dir.is_dir():
#             self.save_dir.mkdir(parents=True)
#         self.interp_type = interp_type

#     @staticmethod
#     def load_csv(path: Union[str, Path]) -> np.ndarray:
#         if isinstance(path, str):
#             path = Path(os.path.expanduser(path)).absolute()
#         data = list()
#         with open(str(path), newline='') as file:
#             for line in csv.reader(file):
#                 if all(line):
#                     # Make sure empty data is not loaded
#                     data.append(line)
#             file.close()
#         data.pop(0)  # Remove headers
#         data = np.stack(data).astype(float)
#         return data

#     @staticmethod
#     def fit_interpolator(
#         timestamps: np.ndarray,
#         values: np.ndarray,
#         type: str
#     ) -> si.interp1d:
#         return si.interp1d(
#             x=timestamps,
#             y=values,
#             kind=type,
#             axis=0,
#             bounds_error=False,
#             fill_value=(values[0], values[-1])
#         )

#     def run(self) -> None:
#         global_step = 0
#         for mp4_path in list(self.episode_dir.glob("**/*.mp4")):
#             episode_dir = mp4_path.parent
#             # Language instruction
#             with open(str(episode_dir.joinpath("metadata.json"))) as file:
#                 metadata = json.load(file)
#                 file.close()
#             language_instruction = metadata['description']
#             # RGB observation
#             rgb = list()
#             with av.open(str(mp4_path)) as container:
#                 fps = container.streams.video[0].average_rate
#                 delta = float(1 / fps)
#                 for i, frame in enumerate(container.decode(video=0)):
#                     frame_ts = np.array([delta * i])
#                     frame = frame.to_ndarray(format="bgr24").reshape(-1)
#                     rgb.append(np.concatenate((frame_ts, frame)))
#             rgb = np.stack(rgb)
#             # Depth observation
#             depth = list()
#             for depth_path in list(episode_dir.joinpath("Depth").glob("*.bin")):
#                 depth_ts = np.array([depth_path.name[6: -4]]).astype(float)
#                 d = np.fromfile(str(depth_path), dtype="uint16") / 1e4  # shape: (192*256,)
#                 depth.append(np.concatenate((depth_ts, d)))
#             depth = np.stack(depth)
#             depth = depth[depth[:, 0].argsort()]
#             # 6D pose
#             pose = self.load_csv(episode_dir.joinpath("PoseData.csv"))
#             pose_ts = pose[:, 0][:, np.newaxis]
#             mat = pose[:, 1:].reshape(-1, 4, 4).transpose(0, 2, 1)
#             trans = mat[:, :3, 3]
#             rot = R.from_matrix(mat[:, :3, :3]).as_euler("xyz", degrees=True)
#             pose = np.concatenate([pose_ts, trans, rot], axis=-1)
#             # Gripper closure
#             closure = self.load_csv(episode_dir.joinpath("AngleData.csv"))
#             # Force and torque
#             force_l = self.load_csv(episode_dir.joinpath("L_ForceData.csv"))
#             force_r = self.load_csv(episode_dir.joinpath("R_ForceData.csv"))
#             # Align data from different sensors. This is done via interpolation.
#             # Intuitively, we preserve the data with the maximum frequency and
#             # interpolate other data with lower frequencies.
#             data_list = [rgb, depth, pose, closure, force_l, force_r]
#             len_list = [data.shape[0] for data in data_list]
#             episode_len = max(len_list) 
#             episode_ts = data_list[len_list.index(episode_len)][:, 0]
#             for i, data in enumerate(data_list):
#                 if data.shape[0] < episode_len:
#                     interp = self.fit_interpolator(
#                         timestamps=data[:, 0],
#                         values=data[:, 1:],
#                         type=self.interp_type
#                     )
#                     data_list[i] = interp(episode_ts)
#                     assert data_list[i].shape[0] == episode_len
#                 else:
#                     data_list[i] = data[:, 1:]
#             # Saving data in a new format
#             episode = list()
#             for step in range(episode_len):
#                 episode.append({
#                     'rgb': data_list[0][step].reshape(480, 640, 3).astype(int),
#                     'depth': data_list[1][step].reshape(192, 256),
#                     'pose': data_list[2][step],
#                     'force_l': data_list[4][step],
#                     'force_r': data_list[5][step],
#                     'action': np.concatenate((data_list[2][step], data_list[3][step]), axis=-1),
#                     'language_instruction': language_instruction
#                 })
#             global_step += 1
#             np.save(str(self.save_dir.joinpath(f"episode_{str(global_step).zfill(3)}.npy")), episode)