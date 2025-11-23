import argparse
import copy
import os
from typing import Dict, List

import habitat
import numpy as np
import cv2

import torch

# from torchvision.transforms import transforms

# from eval_CAM import GradCAM, show_cam_on_image
from model import E2EModel

from main_util import make_print_to_file

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="gpus",
)
parser.add_argument("--difficulty", default="easy", type=str)
parser.add_argument("--actions", default=3, type=int)
parser.add_argument(
    "--model_path", default="your_model.pt", type=str
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="Show RGB observations during evaluation",
)
parser.add_argument("--col_th", default=3, type=int, help="collision threshold")
parser.add_argument("--rec_steps", default=5, type=int, help="recovery steps")
parser.add_argument("--rec_mode", default="best", type=str, choices=["opp", "best"])
args = parser.parse_args()


def observations_to_image(
    observation: Dict, mode="panoramic", local_imgs=None, clip=None, center_agent=True
) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    size = 2.0
    egocentric_view = []

    rgb = observation["panoramic_rgb"]
    if not isinstance(rgb, np.ndarray):
        rgb = rgb.cpu().numpy()
    rgb = cv2.putText(
        np.ascontiguousarray(rgb),
        "current_obs",
        (5, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
    )
    egocentric_view.append(rgb)

    goal_rgb = observation["target_goal"] * 255
    if not isinstance(goal_rgb, np.ndarray):
        goal_rgb = goal_rgb.cpu().numpy()

    if len(goal_rgb.shape) == 4:
        """if info is not None:
        goal_rgb = goal_rgb * (1 - info['total_success']).reshape(-1, *[1] * len(goal_rgb.shape[1:]))"""
        goal_rgb = np.concatenate(
            np.split(goal_rgb[:, :, :, :3], goal_rgb.shape[0], axis=0), 1
        ).squeeze(axis=0)
    else:
        goal_rgb = goal_rgb[:, :, :3]
    goal_rgb = cv2.putText(
        np.ascontiguousarray(goal_rgb),
        "target_obs",
        (5, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
    )
    egocentric_view.append(goal_rgb.astype(np.uint8))

    if len(egocentric_view) > 0:
        if mode == "panoramic":
            egocentric_view = np.concatenate(egocentric_view, axis=0)
        else:
            egocentric_view = np.concatenate(egocentric_view, axis=1)

        frame = cv2.resize(egocentric_view, dsize=None, fx=size * 0.75 * 5, fy=size * 5)
    else:
        frame = None

    return frame


if __name__ == "__main__":
    make_print_to_file(fileName="eval.log")
    from env_utils.make_env_utils import add_panoramic_camera
    from configs.default import get_config
    from env_utils.task_search_env import SearchEnv

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    """load model"""
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    action_space = args.actions

    model = E2EModel(action_space=action_space, device=device, batch_size=1)
    if torch.cuda.is_available():
        model = model.to(device)
    model_path = args.model_path
    sd = torch.load(model_path, map_location=device)  # your model!!!
    model.load_state_dict(sd["state_dict"])
    model.eval()
    overall_success: List[float] = []
    overall_spl: List[float] = []
    overall_dst: List[float] = []
    """set scene"""
    training_scenes = [
        "Cantwell",
        "Denmark",
        "Eastville",
        "Edgemere",
        "Elmira",
        "Eudora",
        "Greigsville",
        "Mosquito",
        "Pablo",
        "Ribera",
        "Sands",
        "Scioto",
        "Sisters",
        "Swormville",
    ]
    for scene in training_scenes:
        """set config"""
        config = get_config()
        config.defrost()
        config.DIFFICULTY = args.difficulty
        habitat_api_path = os.path.dirname(
            habitat.__file__
        )  # os.path.join(os.path.dirname(habitat.__file__), '../')
        config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(
            habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR
        )
        config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(
            habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH
        )
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 500
        config.NUM_PROCESSES = 1
        config.NUM_VAL_PROCESSES = 0
        config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene]  # ["Nicut"]  # environment
        config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG)
        config.record = True
        config.freeze()
        action_list = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS

        env = SearchEnv(config)
        env.habitat_env._sim.seed(1124)
        obs = env.reset()
        env.build_path_follower()
        scene = env.habitat_env.current_episode.scene_id.split("/")[-2]
        total_success: List[float] = []
        total_spl: List[float] = []
        total_dst: List[float] = []
        episode_limit = config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES
        episode_idx = 0

        cc, rc_act, rc_cnt = 0, False, 0

        while True:

            if rc_act:
                if rc_cnt < args.rec_steps:
                    if args.rec_mode == "best":
                        ra = env.get_best_action() - 1  # map to 0,1,2
                        if ra is None:
                            ra = 0
                    else:
                        rseq = [1, 0, 2, 0, 1, 0]
                        ra = rseq[rc_cnt % len(rseq)]
                    obs, _, env_done, info = env.step(ra)
                    rc_cnt += 1
                else:
                    rc_act, rc_cnt, cc = False, 0, 0
            else:
                cur_bgr = torch.tensor(obs["panoramic_rgb"], dtype=torch.float32).unsqueeze(0).to(device)
                target_bgr = torch.tensor(obs["target_goal"], dtype=torch.float32).unsqueeze(0).to(device)
                cur_depth = torch.tensor(obs["panoramic_depth"], dtype=torch.float32).unsqueeze(0).to(device)
                position = torch.tensor(obs["position"], dtype=torch.float32).unsqueeze(0).to(device)
                rotation = torch.tensor(obs["rotation"], dtype=torch.float32).unsqueeze(0).to(device)

                action, _, _ = model(
                    current_bgr=cur_bgr,
                    current_depth=cur_depth,
                    target_bgr=target_bgr[:, :, :, :3],
                    position=position,
                    rotation=rotation[0],
                )
                action = action.item()

                obs, _, env_done, info = env.step(action)

                col = info['collisions']['is_collision']
                if col:
                    cc += 1
                    if cc >= args.col_th:
                        rc_act, rc_cnt = True, 0
                else:
                    cc = 0

            if args.visualize:
                img = observations_to_image(obs)
                img = cv2.resize(img, dsize=(560, 576))
                cv2.imshow("render", img[:, :, [2, 1, 0]])
                cv2.waitKey(5)

            # Track metrics
            distance_to_goal = info.get("distance_to_goal", None)
            if distance_to_goal is not None:
                total_dst.append(max(distance_to_goal - 1, 0))

            if env_done:
                total_success.append(float(info.get("success", 0)))
                spl_val = info.get("spl", 0.0)
                if np.isnan(spl_val):
                    spl_val = 0.0
                total_spl.append(float(spl_val))
                episode_idx += 1
                if episode_idx >= episode_limit:
                    break
                obs = env.reset()
                cc, rc_act, rc_cnt = 0, False, 0
                continue

        print("===============================")
        print("scene:", env.habitat_env.current_episode.scene_id)
        success_rate = float(np.mean(total_success)) if len(total_success) > 0 else 0.0
        spl = float(np.mean(total_spl)) if len(total_spl) > 0 else 0.0
        dst = float(np.mean(total_dst)) if len(total_dst) > 0 else 0.0
        print("success rate:", success_rate)
        print("spl:", spl)
        print("dst:", dst)
        print("===============================")
        overall_success.append(success_rate)
        overall_spl.append(spl)
        overall_dst.append(dst)

    if len(overall_success) > 0:
        print("====== Overall Metrics ======")
        print(f"Average success rate: {np.mean(overall_success):.4f}")
        print(f"Average SPL: {np.mean(overall_spl):.4f}")
        print(f"Average DST: {np.mean(overall_dst):.4f}")
