import networkx as nx
from anytree import NodeMixin
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
import json

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.embeddings = {}

    def add_data(self, data):
        for item in data:
            node_1 = item['node_1']
            node_2 = item['node_2']
            weight = max(0, 8 - item['distance'])
            path = item['path']
            chunk_id = item['chunk_id']

            if not self.graph.has_node(node_1):
                self.graph.add_node(node_1)
            if not self.graph.has_node(node_2):
                self.graph.add_node(node_2)
            if not self.graph.has_edge(node_1, node_2):
                self.graph.add_edge(node_1, node_2, weight=weight, internodes=path, chunk_id=chunk_id)

    def get_nodes(self):
        return list(self.graph.nodes())

    def get_edges(self):
        return list(self.graph.edges(data=True))

    def add_edge_attribute(self, node_1, node_2, attribute, value):
        if self.graph.has_edge(node_1, node_2):
            edges = self.graph.get_edge_data(node_1, node_2)
            for key in edges:
                edges[key][attribute] = value
        else:
            raise ValueError(f"Edge between '{node_1}' and '{node_2}' does not exist in the graph.")

    def get_chunk_ids(self):
        chunk_ids = set()
        for _, _, data in self.graph.edges(data=True):
            chunk_ids.add(data['chunk_id'])
        return list(chunk_ids)

    def save_graph(self, file_path):
        # Convert the graph to a dictionary
        nx.write_gml(self.graph, file_path)
        print("Graph saved successfully.")

    def load_graph(self, file_path):
        self.graph = nx.read_gml(file_path)
        print("Graph loaded successfully.")

class NetworkTreeNode(NodeMixin, BaseEstimator):
    def __init__(self, name, keywords=None, summary=None, query=None, parent=None, cid=None):
        super().__init__()
        self.name = name
        self.keywords = keywords or []
        self.summary = summary
        self.query = query
        self.parent = parent
        self.cid = cid
        self.embedding = []

    def add_child(self, node):
        if not isinstance(node, NetworkTreeNode):
            raise ValueError("Child node must be an instance of NetworkTreeNode.")
        node.parent = self

    def add_parent(self, node):
        if not isinstance(node, NetworkTreeNode):
            raise ValueError("Parent node must be an instance of NetworkTreeNode.")
        self.parent = node

    def get_siblings(self):
        if self.parent is None:
            return []
        return [sibling for sibling in self.parent.children if sibling != self]

    def traverse(self, mode="levelorder"):
        yield from self.traverse_helper(mode)

    def traverse_helper(self, mode):
        if mode == "preorder":
            yield self
        for child in self.children:
            yield from child.traverse_helper(mode)
        if mode == "postorder":
            yield self
        elif mode == "levelorder":
            yield self

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def traverse_to_leaf(self, embedding, top_k=3):
        if self.is_leaf():
            return [self], [cosine_similarity([embedding], [self.embedding])[0][0]]

        max_similarity = -1.0
        most_suitable_child = None

        for child in self.children:
            similarity = cosine_similarity([embedding], [child.embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                most_suitable_child = child

        if most_suitable_child is None:
            return [self], [1.0]

        path, distances = most_suitable_child.traverse_to_leaf(embedding)
        return [self] + path, [max_similarity] + distances

    def traverse_to_leaf_with_beam(self, embedding, top_k=3):
        # Start with the current node in the queue
        queue = [(self, 0)]  # Tuple of node and its similarity
        next_level = []

        while queue:
            current_level = []
            for node, _ in queue:
                if node.is_leaf():
                    # If it's a leaf, we add it directly to the current level for final comparison
                    current_level.append((node, cosine_similarity([embedding], [node.embedding])[0][0]))
                else:
                    # For non-leaf nodes, calculate similarity for all children and add to next level
                    child_similarities = [(child, cosine_similarity([embedding], [child.embedding])[0][0]) for child in
                                          node.children]
                    # Sort by similarity and take the top_k
                    next_level.extend(sorted(child_similarities, key=lambda x: x[1], reverse=True)[:top_k])

            # If we are at a level with leaves, select the best one based on similarity
            if current_level:
                best_leaf, best_sim = max(current_level, key=lambda x: x[1])
                # Build the path back to the root and the similarities
                path = [best_leaf]
                similarities = [best_sim]
                while path[-1].parent:
                    parent = path[-1].parent
                    path.append(parent)
                    # Recalculate similarity for the path (this might be optimized depending on the use case)
                    similarities.append(cosine_similarity([embedding], [parent.embedding])[0][0])
                # The path is built from leaf to root, so we reverse it
                return list(reversed(path)), list(reversed(similarities))

            # Prepare for the next level
            queue = next_level
            next_level = []

        # In case the root is the only node and it's a leaf
        if self.is_leaf():
            return [self], [cosine_similarity([embedding], [self.embedding])[0][0]]
        # If no leaves are found (should not happen in a properly constructed tree)
        return [], []

    def traverse_to_leaf_beam_search(self, embedding, top_k=1):
        # Base case: if the current node is a leaf, return it and its similarity
        if self.is_leaf():
            return [([self], cosine_similarity([embedding], [self.embedding])[0][0])]

        # If not a leaf, proceed with beam search among children
        child_similarities = []
        for child in self.children:
            similarity = cosine_similarity([embedding], [child.embedding])[0][0]
            child_similarities.append((child, similarity))

        # Sort children by similarity and select the top_k
        child_similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_children = child_similarities[:top_k]

        # For each of the selected children, traverse their subtree
        paths_and_similarities = []
        for child, _ in top_k_children:
            paths = child.traverse_to_leaf_beam_search(embedding, top_k)
            for path, path_similarity in paths:
                # Prepend the current child to the path
                new_path = [child] + path
                # The new path and its similarity are added to the list
                paths_and_similarities.append((new_path, path_similarity))

        # Sort all collected paths by their similarities and return the top_k
        paths_and_similarities.sort(key=lambda x: x[1], reverse=True)
        top_paths_and_similarities = paths_and_similarities[:top_k]

        return top_paths_and_similarities

class NetworkTree(BaseEstimator):
    def __init__(self):
        self.nodes = []

    def create_node(self, name, keywords=None, summary=None, query=None, cid=None):
        node = NetworkTreeNode(name=name, keywords=keywords, summary=summary, query=query, cid=cid)
        self.nodes.append(node)
        return node

    def add_parent(self, node, parent):
        node.add_parent(parent)

    def get_root(self) -> NetworkTreeNode:
        roots = [node for node in self.nodes if node.is_root()]
        if len(roots) == 0:
            raise ValueError("No root node found in the tree.")
        elif len(roots) > 1:
            raise ValueError("Multiple root nodes found in the tree.")
        return roots[0]

    def traverse(self, mode="levelorder"):
        root = self.get_root()
        yield from root.traverse(mode)

    def get_nodes_at_depth(self, depth):
        nodes = []
        for node in self.traverse("levelorder"):
            if node.depth == depth:
                nodes.append(node)
        return nodes

    def to_json(self):
        def node_to_dict(node):
            node_dict = {
                'name': node.name,
                'summary': node.summary,
                'query': node.query,
                'keywords': node.keywords,
                'cid': node.cid,
                'children': [node_to_dict(child) for child in node.children]
            }
            return node_dict
        tree_dict = node_to_dict(self.get_root())
        json_string = json.dumps(tree_dict)
        return json_string

    @classmethod
    def from_json(cls, json_string):
        def dict_to_node(node_dict):
            node = NetworkTreeNode(
                name=node_dict['name'],
                summary=node_dict['summary'],
                query=node_dict['query'],
                keywords=node_dict['keywords'],
                cid=node_dict['cid']
            )
            for child_dict in node_dict['children']:
                child_node = dict_to_node(child_dict)
                node.add_child(child_node)
            return node
        tree_dict = json.loads(json_string)
        root_node = dict_to_node(tree_dict)
        tree = cls()
        tree.nodes = [n for n in root_node.traverse()]
        return tree



if __name__ == "__main__":

    #k = KnowledgeGraph()
    #k.load_graph("graph_5KNcQ-phrxlJlKmHR4RYxEtJglR79-d7coRbA0HqTK0.gml")

    tree = NetworkTree()
    node1 = tree.create_node("Node 1", keywords=["keyword1", "keyword2"], summary="Summary 1", query="Query 1")
    node2 = tree.create_node("Node 2", keywords=["keyword3"], summary="Summary 2", query="Query 2")
    node3 = tree.create_node("Node 3", keywords=["keyword4"], summary="Summary 3", query="Query 3")

    tree.add_parent(node2, node1)
    tree.add_parent(node3, node1)

    d = tree.to_json()

    new_tree = NetworkTree().from_json(d)