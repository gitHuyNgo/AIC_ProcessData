import scipy
from sklearn.metrics import silhouette_score
from omegaconf import DictConfig
import hydra
from utils import build_files_list, load_files_list, wait_for_file
import os
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "config"


def hierarchical_clustering_avg(features):
    n = len(features)
    labels = np.arange(n)
    best_score = -float("Inf")
    best_labels = labels.copy()
    best_num_clusters = n

    full_dist = scipy.spatial.distance.cdist(features, features)

    def cluster_dist(id1, id2):
        mask1 = labels == id1
        mask2 = labels == id2
        return full_dist[np.ix_(mask1, mask2)].mean()

    consec_dists = np.zeros(n - 1)
    for i in range(n - 1):
        consec_dists[i] = cluster_dist(i, i + 1)

    maximum_num_clusters = int(n**0.5) + 1
    num_clusters = n

    while num_clusters >= 3:
        min_id = np.argmin(consec_dists)
        min_id_first, min_id_sec = min_id, min_id + 1

        labels = np.where(labels >= min_id_sec, labels - 1, labels)
        num_clusters -= 1

        consec_dists[min_id_sec:-1] = consec_dists[min_id_sec + 1:]
        consec_dists = consec_dists[:-1]

        if min_id_first < len(consec_dists):
            consec_dists[min_id_first] = cluster_dist(
                min_id_first, min_id_first + 1)
        if min_id_first >= 1:
            consec_dists[min_id_first - 1] = cluster_dist(
                min_id_first - 1, min_id_first
            )

        if num_clusters <= maximum_num_clusters:
            try:
                score = silhouette_score(features, labels)
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_labels = labels.copy()
                best_num_clusters = num_clusters

    kf_ids = []
    for i in range(best_num_clusters):
        cluster_mask = best_labels == i
        cluster = features[cluster_mask]
        center = cluster.mean(0)
        dist = ((cluster - center[None, :]) ** 2).sum(-1)
        kf_ids.append(np.argmin(dist) + np.nonzero(cluster_mask)[0][0])

    return best_labels, best_score, kf_ids


def process_scene(scene_boundary, frame_embs):
    l, r = scene_boundary
    scene_length = r - l + 1
    scene_embs = frame_embs[l: r + 1]

    if scene_length < 5:
        center = scene_embs.mean(0)
        dist = ((scene_embs - center[None, :]) ** 2).sum(-1)
        fid = np.argmin(dist)
        return [fid + l]

    best_labels, best_score, kf_ids = hierarchical_clustering_avg(scene_embs)
    index = np.sort(kf_ids)
    return [x + l for x in index]


def process_one(scenes_boundary_path, frames_embedding_path, output_path):
    scenes_boundary = np.genfromtxt(
        scenes_boundary_path, delimiter=" ", dtype="int")
    frame_embs = np.load(frames_embedding_path)

    total_frames = len(frame_embs)

    keyframe_indices = []
    for scene_boundary in tqdm(scenes_boundary, leave=False):
        keyframe_indices += process_scene(scene_boundary, frame_embs)

    selected = len(keyframe_indices)
    ratio = selected / total_frames * 100 if total_frames > 0 else 0
    video_name = Path(frames_embedding_path).stem
    print(
        f"  [{video_name}] total frames: {total_frames} | selected: {selected} | ratio: {ratio:.1f}%"
    )

    np.savetxt(output_path, np.sort(keyframe_indices), fmt="%s")


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def main(config: DictConfig):
    emb_root = PROJECT_ROOT / "dataraw" / "embeddings"
    files_list_root = PROJECT_ROOT / "dataraw" / "folder_file_list"
    scene_root = PROJECT_ROOT / "ProcessedData" / "scence_boundary"

    for subfolder in sorted(os.listdir(emb_root)):
        print(subfolder)
        subfolder_path = os.path.join(emb_root, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        embs_path = load_files_list(
            base_dir=subfolder_path,
            files_list_path=f"{files_list_root}/files_list_{subfolder}.txt",
            with_extension=".npy",
        )
        scenes_bound_path = load_files_list(
            base_dir=f"{scene_root}/{subfolder}",
            files_list_path=f"{files_list_root}/files_list_{subfolder}.txt",
            with_extension=".txt",
        )
        outputs_path = load_files_list(
            base_dir=f"{emb_root}/keyframes_indices_B32_{subfolder}",
            files_list_path=f"{files_list_root}/files_list_{subfolder}.txt",
            with_extension=".txt",
            mkdir=True,
        )

        for emb_p, scene_bp, out_p in tqdm(
            zip(embs_path, scenes_bound_path, outputs_path), total=len(outputs_path)
        ):
            wait_for_file(emb_p)
            wait_for_file(scene_bp)

            if os.path.exists(out_p):
                continue

            process_one(scene_bp, emb_p, out_p)

            if config.clean_frames_embedding:
                with open(emb_p, "w"):
                    pass


if __name__ == "__main__":
    main()
