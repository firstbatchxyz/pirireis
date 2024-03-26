from abc import abstractmethod
import numpy as np
import pandas as pd
from anytree import NodeMixin
from graspologic.partition import leiden
from sklearn.base import BaseEstimator
import pickle
from sklearn.cluster import AffinityPropagation
from src.dstruct import NetworkTree, NetworkTreeNode
from typing import List
import networkx as nx
import re

def symmetrze_nx(g):
    """Leiden requires a symmetric/undirected graph. This converts a directed graph to
    undirected just for this community detection step"""
    sym_g = nx.Graph()
    for source, target, weight in g.edges.data("weight"):
        if sym_g.has_edge(source, target):
            sym_g[source][target]["weight"] = (
                    sym_g[source][target]["weight"] + weight * 0.5
            )
        else:
            sym_g.add_edge(source, target, weight=weight * 0.5)
    return sym_g


class BaseNetworkTree(NodeMixin, BaseEstimator):
    def __init__(
            self,
            min_split=4,
            max_levels=4,
            verbose=False,
    ):
        self.min_split = min_split
        self.max_levels = max_levels
        self.verbose = verbose

    @property
    def node_data(self):
        if self.is_root:
            return self._node_data
        else:
            return self.root.node_data.loc[self._index]

    def _check_node_data(self, adjacency, node_data=None):
        if node_data is None and self.is_root:
            node_data = pd.DataFrame(index=range(adjacency.shape[0]))
            node_data["adjacency_index"] = range(adjacency.shape[0])
            self._node_data = node_data
            self._index = node_data.index

    def fit(self, adjacency, node_data=None):
        self._check_node_data(adjacency, node_data)

        if self.check_continue_splitting(adjacency):
            if self.verbose > 0:
                print(
                    f"[Depth={self.depth}, Number of nodes={adjacency.shape[0]}] Splitting subgraph..."
                )
            partition_labels = self._fit_partition(adjacency)
            self._split(adjacency, partition_labels)

        return self

    def check_continue_splitting(self, adjacency):
        return adjacency.shape[0] >= self.min_split and self.depth < self.max_levels

    def _split(self, adjacency, partition_labels):
        index = self._index
        node_data = self.root.node_data
        label_key = f"labels_{self.depth}"
        if label_key not in node_data.columns:
            node_data[label_key] = pd.Series(
                data=len(node_data) * [None], dtype="Int64"
            )

        unique_labels = np.unique(partition_labels)
        if self.verbose > 0:
            print(
                f"[Depth={self.depth}, Number of nodes={adjacency.shape[0]}] Split into {len(unique_labels)} groups"
            )
        if len(unique_labels) > 1:
            for i, label in enumerate(unique_labels):
                mask = partition_labels == label
                sub_adjacency = adjacency[np.ix_(mask, mask)]
                self.root.node_data.loc[index[mask], f"labels_{self.depth}"] = i
                # sub_node_data = self.node_data.loc[index[mask]]
                child = self.__class__(**self.get_params())
                child.parent = self
                child._index = index[mask]
                child.fit(sub_adjacency)

    @abstractmethod
    def _fit_partition(self, adjacency):
        pass

    def _hierarchical_mean(self, key):
        if self.is_leaf:
            index = self.node_data.index
            var = self.root.node_data.loc[index, key]
            return np.mean(var)
        else:
            children = self.children
            child_vars = [child._hierarchical_mean(key) for child in children]
            return np.mean(child_vars)

class LeidenTree(BaseNetworkTree):
    def __init__(
            self,
            trials=1,
            resolution=1.0,
            min_split=32,
            max_levels=4,
            verbose=False,
    ):
        super().__init__(
            min_split=min_split,
            max_levels=max_levels,
            verbose=verbose,
        )
        self.trials = trials
        self.resolution = resolution

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def _fit_partition(self, adjacency):
        """Fits a partition to the current subgraph using Leiden"""
        partition_map = leiden(adjacency, trials=self.trials, resolution=self.resolution)
        partition_labels = np.vectorize(partition_map.get)(
            np.arange(adjacency.shape[0])
        )
        return partition_labels

    def _generate_llm_summary_keywords(self, keywords, haiku):
        """Generates a summary using LLM"""
        system_prompt = """You are an AI assistant that specializes in generating concise and coherent paragraphs based on a given set of keywords. Your task is to create a well-structured, informative paragraph that explores the main topic represented by the provided keywords. The paragraph should be written in a clear, formal style suitable for a textbook or academic context with a maximum of 4 sentences.

                        When crafting the paragraph, focus on the following aspects:

                        1. Introduce the main topic and establish its significance or relevance.
                        2. Explain the key concepts, ideas, or themes related to the topic, drawing connections between the provided keywords.
                        3. Provide relevant examples, explanations, or contextual information to support the main points and enhance understanding.
                        4. Use precise, academic language and maintain a logical flow of ideas throughout the paragraph.
                        5. Conclude the paragraph by summarizing the main points or highlighting the implications of the discussed topic.

                        Your goal is to create a 3-4 sentence paragraph that effectively conveys the essential information about the topic, demonstrates the relationships between the keywords, and provides a comprehensive overview suitable for an educational or informative context.
                        """
        return haiku.generate(system_prompt, f"{keywords}")

    def _generate_llm_summary_text(self, summaries, depth, haiku):
        """Generates a summary using LLM"""

        prompt = f"""
                I will provide you with a set of summaries on a related topic:
                
                <summaries>
                {summaries}
                </summaries>
                
                Please read through these summaries carefully. Identify the key themes and concepts that are covered
                across the different summaries.
                
                Then, brainstorm a concise, high-level summary that ties together the main ideas and encapsulates
                the key concepts from the summaries. Your goal is to create an overarching summary that captures the
                essence of what the individual summaries are about.
                
                Provide your final summary inside <answer> tags.
                """

        rep = haiku.generate(None, prompt)
        try:
            answer = re.findall(r'<answer>(.*?)<\/answer>', rep, re.DOTALL)
            return answer[0]
        except:
            return rep.split("<answer>")[1]

    def generate_summary(self, leaves, haiku, embedding):
        for leaf in leaves:
            if leaves[0].height > 0:
                keywords = [l.summary for l in leaf.children]
                summary = self._generate_llm_summary_text(keywords, leaf.height, haiku)
            else:
                summary = self._generate_llm_summary_keywords(leaf.keywords, haiku)
            leaf.summary = summary
            leaf.embedding = embedding.encode(summary)
        return leaves

    def generate_graph(self, contract_id, leaves, bert, haiku, embedding):
        """
        Generates a hierarchical graph based on the clustering results.
        1- Generate embeddings
        2- Calculate pairwise distances, remove diagonal
        3- Cluster using Affinity Propagation and distance matrix
        4- generate summaries + queries for each leaf
        5- create nodes for leaves and clusters
        6- add parent-child relations
        7- repeat
        :param leaves:
        :return:
        """
        hierarchy = NetworkTree()

        def __iterate(leaves: List[NetworkTreeNode], depth):

            print(f"Depth: {depth}")
            if len(leaves) == 1:
                self.generate_summary(leaves, haiku, embedding)
                return hierarchy

            if depth == 0:
                keywords = [" ".join(l.keywords) for l in leaves]
            else:
                keywords = [" ".join([ll.summary for ll in l.children]) for l in leaves]
            embeddings = bert.generate_embeddings(keywords)
            dists = [bert.maxsim(e.unsqueeze(0), embeddings) for e in embeddings]
            adj = np.array(dists)
            np.fill_diagonal(adj, 0)

            # Step 3: Cluster using Affinity Propagation and distance matrix
            clustering = AffinityPropagation(random_state=5, affinity="precomputed").fit(adj)
            clusters = {}
            indices = {}
            for i, l in enumerate(clustering.labels_):
                if l not in clusters:
                    clusters[l] = []
                    indices[l] = []
                clusters[l].append(leaves[i])
                indices[l].append(i)

            if len(clusters) >= len(leaves):
                self.generate_summary(leaves, haiku, embedding)
                cluster_node = hierarchy.create_node(name=f"Cluste root d {depth}", summary="", query="", keywords=[])
                for leaf in leaves:
                    hierarchy.add_parent(leaf, cluster_node)
                return __iterate([cluster_node], depth+1)

            # Step 4: Generate summaries + queries for each leaf
            self.generate_summary(leaves, haiku, embedding)

            # Step 5: Create nodes for clusters
            cluster_nodes = []
            for label in indices.keys():
                cluster_node = hierarchy.create_node(name=f"Cluster {label} d {depth}", summary="", query="",
                                                     keywords=[])
                for ind in indices[label]:
                    hierarchy.add_parent(leaves[ind], cluster_node)
                cluster_nodes.append(cluster_node)

            return __iterate(cluster_nodes, depth + 1)

        leaf_nodes = [hierarchy.create_node(name="Leaf", keywords=leaf, summary="", query="", cid=contract_id) for leaf in leaves]
        return __iterate(leaf_nodes, 0)