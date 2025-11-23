import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import no_grad
import torchvision
import numpy as np
from utils.obj_type import CATEGORIES
from GAT.gat_model import GAT
import networkx as nx
from detectron.obj_detect import ObjDetect
from utils.graph_pooling_utils import normalize, GCN_3_layers_
from graph import Graph
import matplotlib.pyplot as plt
import copy


class E2EModel(nn.Module):
    def __init__(self, action_space=3, device="cpu", batch_size=4):
        super(E2EModel, self).__init__()
        self.timestamp = 0
        self.last_time_obj = []
        for i in range(batch_size):
            self.last_time_obj.append([])
        self.device = device
        self.batch_size = batch_size

        """Resnet50"""
        self.resnet = torchvision.models.resnet50(pretrained=True).to(
            device
        )  # b*3*144*192, bgr, 0-1--->b*1000
        self.resnet_rgbd = torchvision.models.resnet50(pretrained=True).to(
            device
        )  # b*3*144*192, bgr, 0-1--->b*1000
        self.resnet_rgbd.conv1 = nn.Conv2d(
            4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )  # b*4*144*192--b*64*72*96
        self.resnet_fc = ResnetFC().to(device)  # b*1000--b*256

        """detectron2"""
        self.detect_model = ObjDetect()

        """policy module"""
        self.policyNet = PolicyNet(action_space).to(
            device
        )  # feature_vector ---> action_prob

        self.default_value = 1e-6

        """Semantic Graph"""
        self.SemanticGraph = Graph(self.device)  # Create an empty graph

        a_raw = torch.load("Semantic/adjmat_w.dat")  # 42*42
        """setting weight"""
        mean_val = a_raw.mean()  # TODO:adjust
        std_val = a_raw.std()
        a_raw = (a_raw - mean_val) / std_val
        max_val = a_raw.max()
        min_val = a_raw.min()
        a_raw = (a_raw - min_val) / (max_val - min_val)
        try:
            a_raw = a_raw.numpy()
        except Exception as e:
            print(e)
        x, y = np.where(a_raw > 0)  # Locate all edge indices
        x = torch.tensor(x)  # Convert to tensor
        y = torch.tensor(y)  # Convert to tensor
        for i in range(len(CATEGORIES["gibson"])):
            self.SemanticGraph.add_node(
                visual_feature=None,
                timestamp=self.timestamp,
                robot_pos=None,
                detect_type=CATEGORIES["gibson"][i],
                detect_results=i,
                wordvec=CATEGORIES["vec"][i],
                isReal=False,
            )
        self.max_virtual_node = len(CATEGORIES["gibson"]) - 1
        # Add edges
        for i in range(x.size(0)):
            source = x[i].item()
            target = y[i].item()
            weight = a_raw[source][target]
            if source != target:
                self.SemanticGraph.add_edge(source, target, weight=weight)

        print("Semantic Graph has been created successfully!")
        """Main Graph"""
        self.MainGraph = []
        self.VirtualNode = []  # Track virtual nodes to avoid duplicates
        for i in range(self.batch_size):
            self.MainGraph.append(Graph())  # Graph containing semantic nodes
            self.VirtualNode.append(
                [-1] * self.max_virtual_node
            )  # Store indices of virtual nodes

        """GCN"""
        # self.gcn_fc = GcnFC().to(device)  # 142-->256
        self.GAT = GAT(
            nclass=1,
            dropout=0.6,
            nheads=4,
            alpha=0.2,
            nhid=256,
            nfeat=512,
            device=self.device,
        ).to(device)

    def graph_init(self):
        print("Init Graph...")
        for i in range(self.batch_size):
            self.last_time_obj[i] = []
        self.MainGraph = []
        self.VirtualNode = []
        for i in range(self.batch_size):
            self.MainGraph.append(Graph())  # Reset graph
            self.VirtualNode.append(
                [-1] * self.max_virtual_node
            )  # Reset virtual node indices

    def debug_node_num(self, episode, point_value=None):
        self.MainGraph[0].visualize_info(
            save_path="img3/episode_" + str(episode) + ".png", point_value=point_value
        )
        # plt.plot(self.plt_time, self.plt_nodes, marker='o', linestyle='-', color='b', label='Node Count')
        # plt.xlabel('Timestamp')
        # plt.ylabel('Node Count')
        # plt.title('Node Count Over Time')
        # plt.legend()
        # plt.grid(True)
        # plt.xticks(rotation=45)  # Rotate labels to avoid overlap
        #
        # # Save as image
        # plt.savefig('img/node_count_plot_'+episode+'.png')
        #
        # # Close the plot to avoid overlap across saves
        # plt.close()

    def forward(
        self,
        current_bgr=None,
        target_bgr=None,
        position=None,
        current_depth=None,
        rotation=None,
        # detect_features=None, current_visual_feature_=None,
        # number_of_nodes=None, all_edges=None, topological_graph=None,
        name="gathering",
    ):
        time_cout = []
        if name == "gathering":
            self.batch_size = current_bgr.size(0)
            with no_grad():
                """resnet"""
                resnet_current_input = torch.Tensor(
                    np.array(current_bgr.cpu()).transpose((0, 3, 1, 2))
                ).to(
                    self.device
                )  # b*3*144*192, bgr, 0-1
                resnet_target_input = torch.Tensor(
                    np.array(target_bgr.cpu()).transpose((0, 3, 1, 2))
                ).to(
                    self.device
                )  # b*3*144*192, bgr, 0-1
                resnet_depth_input = torch.Tensor(
                    np.array(current_depth.cpu()).transpose((0, 3, 1, 2))
                ).to(self.device)

                current_visual_feature_ = self.resnet_rgbd(
                    torch.cat((resnet_current_input, resnet_depth_input), dim=1)
                )  # batch_size*3*144*192 --> b*1000
                target_visual_feature_ = self.resnet(
                    resnet_target_input
                )  # batch_size*3*144*192 --> b*1000
                """detectron"""
                obj_detect_results = []
                for i in range(self.batch_size):
                    input_ = np.round(current_bgr[i].cpu().numpy() * 255).astype(
                        np.uint8
                    )  # 1*144*192*3, 0-255
                    # obj_detect_results.append(self.detect_model.obj_detect(input_))
                    res, img_ = self.detect_model.obj_detect(input_)
                    obj_detect_results.append(res)
                    # detect_results: [boxes, confidence, classes]

            current_visual_feature = self.resnet_fc(
                current_visual_feature_
            )  # b*1000 --> b*256
            target_visual_feature = self.resnet_fc(
                target_visual_feature_
            )  # b*1000 --> b*256
            start = time.time()
            """Observation builds the topological graph on the fly"""
            self.timestamp += 1
            for b in range(self.batch_size):
                new_nodes_index = []
                if len(obj_detect_results[b]) > 0 and len(obj_detect_results[b][2]) > 0:
                    for i in range(
                        len(obj_detect_results[b][2])
                    ):  # Iterate over all detected objects
                        target_category = obj_detect_results[b][2][i].dtype.num

                        self.MainGraph[b].add_node(
                            current_visual_feature[b].to(self.device),
                            self.timestamp,
                            robot_pos=torch.cat(
                                (
                                    position[b].to(self.device),
                                    torch.tensor([rotation[b]]).to(self.device),
                                ),
                                dim=0,
                            ).to(self.device),
                            detect_results=[
                                obj_detect_results[b][0][i],
                                obj_detect_results[b][1][i],
                                obj_detect_results[b][2][i],
                            ],
                            detect_type=CATEGORIES["gibson"][target_category],
                            wordvec=CATEGORIES["vec"][target_category],
                        )  # Add node
                        index = self.MainGraph[b].num_nodes - 1
                        new_nodes_index.append(index)  # Track newly added node indices

                        """Add virtual nodes"""
                        if target_category <= self.max_virtual_node:
                            # If the semantic graph node category matches the detection, find neighbors
                            for j in range(len(self.SemanticGraph.graph.edges)):
                                if (
                                    self.SemanticGraph.graph.edges[j][0]
                                    == target_category
                                ):
                                    neighbor_node = self.SemanticGraph.graph.edges[j][1]
                                    # Connect edges
                                    if (
                                        self.VirtualNode[b][neighbor_node] < 0
                                    ):  # If the virtual node does not exist
                                        self.MainGraph[b].add_node_copy(
                                            self.SemanticGraph.graph.nodes[
                                                neighbor_node
                                            ]
                                        )
                                        self.VirtualNode[b][neighbor_node] = (
                                            self.MainGraph[b].num_nodes - 1
                                        )
                                    self.MainGraph[b].add_edge(
                                        index,
                                        self.VirtualNode[b][neighbor_node],
                                        self.SemanticGraph.graph.edges[j][2],
                                    )

                    for old_node in self.last_time_obj[b]:
                        for new_node in new_nodes_index:
                            self.MainGraph[b].add_edge(old_node, new_node, 1)
                            self.MainGraph[b].add_edge(new_node, old_node, 1)
                    self.last_time_obj[b] = new_nodes_index

                # '''Debug'''
                # self.plt_time.append(self.timestamp)
                # self.plt_nodes.append(self.MainGraph[b].real_node_total)

            """graph pooling"""
            try:
                main_graph_feature = 0
                for b in range(self.batch_size):
                    nx_MainGraph = self.MainGraph[b].to_networkx()
                    A = nx.adjacency_matrix(nx_MainGraph).todense()  # matrix n*n
                    # A = sp.csr_matrix(nx_MainGraph).todense()     #matrix n*n
                    Anormed = normalize(torch.FloatTensor(A), True).to(
                        self.device
                    )  # num_node*num_node

                    node_features_ = np.array(
                        [
                            nx_MainGraph.nodes[i]["feature"].detach().cpu().numpy()
                            for i in nx_MainGraph.nodes
                        ]
                    )  # ndarray n*512
                    node_features_ = torch.tensor(
                        node_features_
                    )  # Convert node feature matrix

                    # self.GAT = GAT(nclass=256, dropout=0.6, nheads=4, alpha=0.2, nhid=256, nfeat=512).to(self.device)
                    main_graph_feature_ = self.GAT(
                        adj=Anormed.float().to(self.device),
                        x=node_features_.float().to(self.device),
                    )  # nclass(1)*n
                    # main_graph_feature_ = self.gcn_fc(main_graph_feature_, n=main_graph_feature_.size(1))  # tensor 1*256
                    if b == 0:
                        main_graph_feature = main_graph_feature_
                    else:
                        main_graph_feature = torch.cat(
                            (main_graph_feature, main_graph_feature_), dim=0
                        )  # b*256
                    # print('gcn_fc:', topological_semantic_feature)

                """policy module"""
                # main_graph_feature = main_graph_feature.repeat(self.batch_size, 1)  # batch_size*256 TODO:BUG
                # zero = torch.zeros(self.batch_size, 256).to(self.device)
                feature_vector = torch.cat(
                    [current_visual_feature, main_graph_feature, target_visual_feature],
                    dim=1,
                )  # shape: (batch_size, 256 + 256 + 256)

                # Reshape feature_vector to match the desired shape (batch_size, 3 * (1000 + 256 + 1000))
                feature_vector = feature_vector.view(
                    self.batch_size, -1
                )  # shape: (batch_size, 768)
                # value = self.valueNet(feature_vector)  # b*768 --> b*1
                pi = self.policyNet(feature_vector)  # b*768 --> b*3
                pre_action = torch.argmax(pi, dim=1)  # b*1
                # print("Time:", time.time() - start)
                # print("Num of nodes:", self.MainGraph[0].num_nodes)

                return pre_action, pi, img_
            except Exception as e:  # Handle initial empty-graph case
                print(e)
                zero = torch.zeros(self.batch_size, 256).to(self.device)
                feature_vector = torch.cat(
                    [current_visual_feature, zero, target_visual_feature], dim=1
                )  # batch_size * 768
                feature_vector = feature_vector.view(
                    self.batch_size, -1
                )  # shape: (batch_size, 768)
                # value = self.valueNet(feature_vector)  # b*768 --> b*1
                pi = self.policyNet(feature_vector)  # b*768 --> b*3
                pre_action = torch.argmax(pi, dim=1)  # b*1
                return pre_action, pi, img_


class PolicyNet(nn.Module):  # actor
    def __init__(self, action_size):
        super(PolicyNet, self).__init__()
        # self.policy_output = nn.Linear(in_features=512, out_features=action_size)

        self.mlp_action_prob = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, action_size),
        )

    def forward(self, feature_vector):
        action_prob = F.softmax(self.mlp_action_prob(feature_vector), dim=1)
        return action_prob


class ValueNet(nn.Module):  # critic
    def __init__(self):
        super(ValueNet, self).__init__()
        # self.value_output = nn.Linear(in_features=512, out_features=1)
        self.mlp_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, feature_vector):
        value = self.mlp_value(feature_vector)
        return value


class ResnetFC(nn.Module):
    def __init__(self):
        super(ResnetFC, self).__init__()
        # self.policy_output = nn.Linear(in_features=512, out_features=action_size)

        self.resnet_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 256),
        )

    def forward(self, resnet_output):
        visual_feature = self.resnet_fc(resnet_output)
        return visual_feature


class GcnFC(nn.Module):
    def __init__(self):
        super(GcnFC, self).__init__()
        self.gcn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 256),
        )

    def forward(self, node_features_, n=128):
        self.n = n
        topological_semantic_feature = self.gcn_fc(node_features_)
        return topological_semantic_feature


class LoopDetectionNet(nn.Module):
    def __init__(self):
        super(LoopDetectionNet, self).__init__()
        self.mlp_loopDet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(2, 1),
        )

    def forward(self, pose):
        loop = self.mlp_loopDet(pose)  # 0 = no loop closure; 1 = loop closure
        return loop


class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


class PositionNet(nn.Module):  # actor
    def __init__(self):
        super(PositionNet, self).__init__()

        self.mlp_position = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, 2),
        )

    def forward(self, feature_vector2):
        position = self.mlp_position(feature_vector2)
        return position


class ThetaNet(nn.Module):
    def __init__(self):
        super(ThetaNet, self).__init__()
        self.mlp_theta = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, feature_vector2):
        theta = self.mlp_theta(feature_vector2)
        return theta
