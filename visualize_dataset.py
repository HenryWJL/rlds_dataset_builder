import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tqdm
import importlib
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


dataset_name = "magiclaw_dataset"
module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, split='train')
ds = ds.shuffle(6)

# Visualize RGB observation
for i, episode in enumerate(ds.take(1)):
    image = list(episode['steps'])[100]['observation']['rgb'].numpy()
    instruction = list(episode['steps'])[100]['language_instruction'].numpy().decode("utf-8")
    plt.figure()
    plt.imshow(image)
    plt.title(instruction)

# Visualize action and state statistics
actions, poses, forces_l, forces_r = [], [], [], []
for episode in tqdm.tqdm(ds.take(1)):
    for step in episode['steps']:
        actions.append(step['action'].numpy())
        poses.append(step['observation']['pose'].numpy())
        forces_l.append(step['observation']['force_l'].numpy())
        forces_r.append(step['observation']['force_r'].numpy())

actions = np.array(actions)
poses = np.array(poses)
forces_l = np.array(forces_l)
forces_r = np.array(forces_r)
action_mean = actions.mean(0)
pose_mean = poses.mean(0)
force_l_mean = forces_l.mean(0)
force_r_mean = forces_r.mean(0)

def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]
    n_elems = vector.shape[1]
    plt.figure(tag, figsize=(5*n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem+1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(vector_mean[elem])

vis_stats(actions, action_mean, 'action_stats')
vis_stats(poses, pose_mean, 'pose_stats')
vis_stats(forces_l, force_l_mean, 'force_l_stats')
vis_stats(forces_r, force_r_mean, 'force_r_stats')
plt.show()


