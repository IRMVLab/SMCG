from typing import Dict

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import imageio
import cv2
import copy


class BC_trainer(nn.Module):
    def __init__(self, agent, device="cuda:0", no_stop=1):
        self.no_stop = no_stop
        super().__init__()
        self.agent = agent
        self.torch_device = device
        self.optim = optim.Adam(
            list(filter(lambda p: p.requires_grad, self.agent.parameters())), lr=0.0001
        )

    def save(self, file_name=None, epoch=0, step=0):
        if file_name is not None:
            save_dict = {}

            save_dict["trained"] = [epoch, step]
            save_dict["state_dict"] = self.agent.state_dict()
            torch.save(save_dict, file_name)

    def forward(self, batch, train=True):
        self.agent.graph_init()
        (
            demo_rgb,
            demo_depth,
            demo_act,
            positions,
            rotations,
            targets,
            target_img,
            scene,
            start_pose,
            aux_info,
        ) = batch
        demo_rgb, demo_depth, demo_act = (
            demo_rgb.to(self.torch_device),
            demo_depth.to(self.torch_device),
            demo_act.to(self.torch_device),
        )
        target_img, positions, rotations = (
            target_img.to(self.torch_device),
            positions.to(self.torch_device),
            rotations.to(self.torch_device),
        )
        # aux_info = {'have_been': aux_info['have_been'].to(self.torch_device),
        #           'distance': aux_info['distance'].to(self.torch_device)}
        self.B = demo_act.shape[0]

        lengths = (demo_act > -10).sum(dim=1)

        T = lengths.max().item()
        # T = 5
        # hidden_states = torch.zeros(self.agent.net.num_recurrent_layers, self.B, self.agent.net._hidden_size).to(self.torch_device)
        actions = torch.zeros([self.B]).to(self.torch_device)
        results = {
            "imgs": [],
            "node_num": [],
            "node_list": [],
            "actions": [],
            "gt_actions": [],
            "target": [],
            "scene": scene[0],
            "A": [],
            "position": [],
            "have_been": [],
            "distance": [],
            "pred_have_been": [],
            "pred_distance": [],
        }
        actions_logits_all = []
        gt_actions_all = []
        gt_act_rate = [0, 0, 0, 0]
        pre_act_rate = [0, 0, 0, 0]
        cnt_1 = torch.sum(torch.eq(demo_act, 1))
        cnt_2 = torch.sum(torch.eq(demo_act, 2))
        cnt_3 = torch.sum(torch.eq(demo_act, 3))
        cnt_total = cnt_1 + cnt_2 + cnt_3
        if self.no_stop:
            action_weight = torch.tensor(
                [cnt_total / cnt_1 * 3, cnt_total / cnt_2 * 3, cnt_total / cnt_3 * 3]
            )
        else:
            action_weight = torch.tensor(
                [1, cnt_total / cnt_1 * 3, cnt_total / cnt_2 * 3, cnt_total / cnt_3 * 3]
            )
        action_weight = action_weight.to(self.torch_device) * 10

        for t in range(T):
            masks = lengths > t
            if t == 0:
                masks[:] = False
            target_goal = target_img[
                torch.range(0, self.B - 1).long(), targets[:, t].long()
            ]
            pose_t = positions[:, t]
            # obs_t = self.env_wrapper.step([demo_rgb[:,t], demo_depth[:,t], pose_t, target_goal, torch.ones(self.B).cuda()*t, (~masks).detach().cpu().numpy()])
            obs_t = demo_rgb[:, t]
            depth_t = demo_depth[:, t]
            rotat_t = rotations[:, t]
            # if t < lengths[0]:
            #     results['imgs'].append(demo_rgb[0, t].cpu().numpy())
            #     results['target'].append(target_goal[0].cpu().numpy())
            #     results['position'].append(positions[0, t].cpu().numpy())
            #     results['have_been'].append(aux_info['have_been'][0, t].cpu().numpy())
            #     results['distance'].append(aux_info['distance'][0, t].cpu().numpy())
            observation = {}
            observation["panoramic_rgb"] = obs_t[0]
            observation["target_goal"] = target_goal[0]
            img = observations_to_image(observation)
            img = cv2.resize(img, dsize=(400, 280))
            cv2.imshow("render", img[:, :, [2, 1, 0]])
            cv2.waitKey(5)

            gt_act = copy.deepcopy(demo_act[:, t])
            if -1 in gt_act:  # TODO: handle missing ground-truth actions
                b = torch.where(gt_act == -1)
                gt_act[b] = 0
            if -100 in actions:
                b = torch.where(actions == -100)
                actions[b] = 0
            (pred_act, actions_logits,) = self.agent(
                current_bgr=obs_t,
                target_bgr=target_goal[:, :, :, :3],
                position=pose_t,
                current_depth=depth_t,
                rotation=rotat_t,
            )
            if not (gt_act == -100).all():
                # print(gt_act)
                # print("predict:", pred_act)
                for ij in range(len(gt_act)):
                    if gt_act[ij].long() != -100:
                        gt_act_rate[gt_act[ij].long()] += 1
                for ij in range(len(pred_act)):
                    pre_act_rate[pred_act[ij].long()] += 1
                gt_act -= self.no_stop
                valid_indices = gt_act.long() >= 0
                actions_logits_all.append(actions_logits[valid_indices])
                gt_actions_all.append(gt_act[valid_indices])
                # loss = F.cross_entropy(actions_logits[valid_indices].view(-1, actions_logits[valid_indices].shape[1]),
                #                        gt_act[valid_indices].long().view(-1), weight=action_weight)
                # # if loss is NAN
                # if torch.isnan(loss):
                #     print("actions_logits:", actions_logits, "gt_act:", gt_act, "valid_indices:", valid_indices,"weight:",action_weight)
                # else:
                #     losses.append(loss)

            else:
                results["actions"].append(-1)
                results["gt_actions"].append(-1)

            actions = demo_act[:, t].contiguous()

        actions_logits_all = torch.cat(actions_logits_all, 0)
        gt_actions_all = torch.cat(gt_actions_all, 0)
        losses = F.cross_entropy(
            actions_logits_all.reshape(-1, actions_logits_all.shape[-1]),
            gt_actions_all.reshape(-1).long(),
        )

        results["node_num"] = self.agent.MainGraph[0].num_nodes

        if train:
            self.optim.zero_grad()
            losses.backward()
            self.optim.step()

        loss_dict = {}
        loss_dict["loss"] = losses.item()
        loss_dict["gt_act_1"] = gt_act_rate[1] / sum(gt_act_rate)
        loss_dict["gt_act_2"] = gt_act_rate[2] / sum(gt_act_rate)
        loss_dict["gt_act_3"] = gt_act_rate[3] / sum(gt_act_rate)
        loss_dict["pre_act_1"] = pre_act_rate[1] / sum(pre_act_rate)
        loss_dict["pre_act_2"] = pre_act_rate[2] / sum(pre_act_rate)
        loss_dict["pre_act_0"] = pre_act_rate[0] / sum(pre_act_rate)
        print("gt_act_rate:", gt_act_rate)
        print("pred:", pre_act_rate)

        return results, loss_dict


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
        rgb = rgb.cpu().numpy().astype(np.uint8)
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
        goal_rgb = goal_rgb.cpu().numpy().astype(np.uint8)

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

        frame = cv2.resize(egocentric_view, dsize=None, fx=size * 0.75, fy=size)
    else:
        frame = None

    return frame
