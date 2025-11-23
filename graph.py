import random
import time

import torch
import torch_geometric as pyg
import networkx as nx
import matplotlib.pyplot as plt


class Node:
    def __init__(
        self,
        visual_feature,
        timestamp,
        robot_pos,
        detect_results=None,
        detect_type=None,
        wordvec=None,
        isReal=True,
    ):
        self.visual_feature = visual_feature  # tensor 1*256
        self.timestamp = timestamp  # int
        self.robot_pos = robot_pos  # ndarray 1*2
        self.detect_results = detect_results
        self.isReal = isReal  # bool
        self.box = torch.zeros(5, dtype=torch.float32)  # ndarray 1*5
        self.detect_type_label = detect_type  # label
        self.confidence = 0  # float
        self.wordvec = wordvec  # tensor 1*300
        if isReal:
            self.detect_type_index = detect_results[2].item()
            self.box = detect_results[0]
            self.confidence = detect_results[1].item()
        else:
            self.detect_type_index = detect_results
            self.confidence = 1


#
# class Edge:
#     def __init__(self, weight):
#         self.weight = weight


class Graph:
    def __init__(self, device="cpu"):
        self.graph = pyg.data.Data()
        self.graph.nodes = []
        self.graph.edges = []
        self.real_node_total = 0
        self.num_nodes = 0
        self.device = device

    def add_node(
        self,
        visual_feature,
        timestamp,
        robot_pos,
        detect_results=None,
        detect_type=None,
        wordvec=None,
        isReal=True,
    ):
        node = Node(
            visual_feature,
            timestamp,
            robot_pos,
            detect_results,
            detect_type,
            wordvec,
            isReal,
        )
        self.graph.nodes.append(node)
        if isReal:
            self.real_node_total += 1
        self.num_nodes += 1

    def add_node_copy(self, node):
        # #   Look up whether the node already exists in the graph
        # for i, n in enumerate(self.graph.nodes):
        #     if node.isReal == False and n.detect_type == node.detect_type:
        #         return i
        self.graph.nodes.append(node)
        if node.isReal:
            self.real_node_total += 1
        self.num_nodes += 1
        return -1

    def add_edge(self, source, target, weight):
        # edge = Edge(weight)
        # self.graph.edges.append((source, target, edge))
        self.graph.edges.append((source, target, weight))

    def get_neighbors(self, node_index):
        neighbors = []
        for i, edge in enumerate(self.graph.edges):
            source, target, _ = edge
            if source == node_index:
                neighbors.append(target)
            elif target == node_index:
                neighbors.append(source)
        return neighbors

    def to_networkx(self):
        nx_graph = nx.Graph()

        # Add nodes
        for i, node in enumerate(self.graph.nodes):
            nx_graph.add_node(
                i, feature=self.node_feature(node)
            )  # Use the index as the node identifier

        # Add edges
        for source, target, weight in self.graph.edges:
            nx_graph.add_edge(source, target, weight=weight)

        return nx_graph

    def node_feature(self, node):
        if not node.isReal:
            tensor1 = torch.zeros(30, dtype=torch.float32).to(self.device)
            tensor2 = (
                torch.tensor([node.detect_type_index], dtype=torch.int64)
                .repeat(96)
                .to(self.device)
            )
            word2vec = (
                torch.tensor(node.wordvec.flatten()).repeat(5).to(self.device)
            )  # Flatten word2vec 74
            tensor3 = torch.zeros(16, dtype=torch.float32).to(self.device)
            return torch.cat((tensor1, tensor2, word2vec, tensor3))
        visual_feature = node.visual_feature.flatten().to(
            self.device
        )  # Flatten visual feature 256
        timestamp = (
            torch.tensor([node.timestamp], dtype=torch.int64).repeat(32).to(self.device)
        )  # Repeat timestamp tensor 32
        robot_pos = (
            torch.tensor(node.robot_pos.flatten(), dtype=torch.float32)
            .repeat(24)
            .to(self.device)
        )  # Flatten robot position 96
        box = (
            torch.tensor(node.box.flatten(), dtype=torch.float32)
            .repeat(5)
            .to(self.device)
        )  # Flatten bounding box (zero if None) 20
        detect_type = (
            torch.tensor([node.detect_type_index], dtype=torch.int64)
            .repeat(18)
            .to(self.device)
        )  # Repeat detection type tensor 18
        word2vec = torch.tensor(node.wordvec.flatten()).to(
            self.device
        )  # Flatten word2vec 74
        confidence = (
            torch.tensor([node.confidence], dtype=torch.float32)
            .repeat(16)
            .to(self.device)
        )  # Repeat confidence tensor 16

        concatenated = torch.cat(
            (
                visual_feature,
                robot_pos,
                detect_type,
                word2vec,
                box,
                confidence,
                timestamp,
            )
        )
        return concatenated  # tensor 512

    def print_info(self):
        nx_graph = nx.Graph()

        # Add nodes
        node_colors = []
        for node in self.graph.nodes:
            nx_graph.add_node(node.isReal)
            if node.isReal:
                node_colors.append("red")
            else:
                node_colors.append("blue")

        # Add edges
        for source, target, weight in self.graph.edges:
            nx_graph.add_edge(source, target, weight=weight)

        # Create figure
        fig, ax = plt.subplots()

        # Draw graph
        pos = nx.circular_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, ax=ax, node_color=node_colors)

        # Show and save plot
        plt.show()
        plt.savefig("image" + str(time.time()) + ".png")

    def visualize_info(self, save_path=None, point_value=None):
        point_value = point_value[0]
        if self.num_nodes == 0:
            return
        nx_graph = self.to_networkx()

        # Draw graph
        plt.figure(figsize=(7, 7))
        cmap = plt.cm.Greens  # Replace 'Blues' with your preferred colormap
        cmap_red = plt.cm.Reds
        norm = plt.Normalize(min(point_value), max(point_value))

        # Assign colors to nodes based on the colormap
        node_colors = [cmap(norm(value) + 0.1) for value in point_value]
        pos = nx.spring_layout(nx_graph, seed=42)
        # nx.draw_networkx(nx_graph, pos=pos, with_labels=False)  # Draw graph

        # Set node colors
        # node_colors = ['r'] * self.num_nodes
        for i in range(self.num_nodes):
            if self.graph.nodes[i].isReal:
                node_colors[i] = cmap(random.random() * 0.6)
            else:
                node_colors[i] = cmap_red(norm(point_value[i]))
        nx.draw_networkx_nodes(nx_graph, pos=pos, node_color=node_colors, cmap="Set2")

        # Configure edge appearance
        for edge in nx_graph.edges():
            # Get the edge value

            if self.graph.nodes[edge[0]].isReal and self.graph.nodes[edge[1]].isReal:
                edge_color = "gray"  # Solid line style
            else:
                edge_color = "#ADD8E6"  # Dashed line style for other edges

            # Draw the edge with defined style
            nx.draw_networkx_edges(
                nx_graph, pos, edgelist=[edge], edge_color=edge_color, width=0.5
            )

        # edge_weights = [weight for _, _, weight in self.graph.edges]
        # edge_labels = {(u, v): f"{weight:.2f}" for (u, v, weight) in self.graph.edges}
        # nx.draw_networkx_edges(nx_graph, pos=pos,edge_color='gray')
        # nx.draw_networkx_edge_labels(nx_graph, pos=pos, edge_labels=edge_labels)
        # nx.draw_networkx_edge_labels(nx_graph, pos=pos)
        # Configure node labels
        node_labels = {
            node: f"{self.graph.nodes[node].detect_type_index}"
            for node in nx_graph.nodes
        }
        nx.draw_networkx_labels(nx_graph, pos=pos, labels=node_labels, font_size=10)
        # Show final image
        plt.axis("off")
        plt.savefig(save_path)
        plt.close()
