import os

import networkx as nx
from tqdm import tqdm
import numpy as np
from itertools import chain
from ast import literal_eval
from dria import Dria
from decouple import config
from src import *
import os


def generate_knowledge_graph(contract_id, num_samples=50):
    """
    Generate a knowledge graph from a contract.
    :param contract_id:
    :param num_samples:
    :return:
    """
    db = DBWorker()
    haiku = Haiku()
    gpt = OpenAIWorker()
    kg = KnowledgeGraph()

    client = Dria(api_key=config("DRIA_API_KEY"))
    client.set_contract(contract_id)
    entry_count = client.entry_count() - 1
    error = 0
    samples = []
    skip = int(entry_count / num_samples)
    ctr = 0
    while len(samples) < num_samples:
        sample = np.random.randint(skip * ctr, skip * (ctr + 1))
        samples.append(sample)
        ctr += 1

    item_ids = [client.fetch(samples[i:i + 50]) for i in range(0, len(samples), 50)]
    item_ids = list(chain.from_iterable(item_ids))

    for i, item in tqdm(enumerate(item_ids)):

        text = item["metadata"]["metadata"]["text"]
        prompt = create_prompt(text)

        resp = haiku.generate(SYS_PROMPT_H2, prompt)
        try:
            g = literal_eval(resp)
            for el in g:
                el["chunk_id"] = samples[i]
            kg.add_data(g)
        except:
            print("error")
            try:
                resp = gpt.ask(SYS_PROMPT_CORRECTIVE_H2, prompt)
                g = literal_eval(resp)
                for el in g:
                    el["chunk_id"] = samples[i]
                    el["contract"] = contract_id
                kg.add_data(g)
            except:
                error += 1
                continue

    print(f"Finished graph with {(error/num_samples) * 100}% errors on JSON calling.")
    db.update_knowledge_graph(contract_id, knowledge_graph=kg.to_string())


def generate_context_tree(contract_id)-> NetworkTree:
    """
    Generate a context hierarchy from a knowledge graph.
    :param contract_id:
    :return:
    """
    db = DBWorker()
    gml = db.get_knowledge_graph(contract_id)

    kg = KnowledgeGraph()
    kg.from_string(gml)
    g = kg.graph
    sym_g = symmetrze_nx(g)
    adjacency = nx.to_scipy_sparse_array(sym_g, nodelist=list(g.nodes))

    bert = BertEmbedding()
    haiku = Haiku()
    je = JinaEmbedding()

    lt = LeidenTree(verbose=True, max_levels=2, min_split=32, resolution=1.0)
    lt.fit(adjacency)

    leaves = [leaf.node_data["adjacency_index"].tolist() for leaf in lt.leaves]
    leaves = [l for l in leaves if len(l) > 2]
    leaves_w = [[list(g.nodes)[ind] for ind in l] for l in leaves]
    hgraph = lt.generate_graph(leaves=leaves_w, contract_id=contract_id, bert=bert, haiku=haiku, embedding=je)

    gml = hgraph.to_json()
    db.update_context_tree(contract_id, context_tree=gml)
    return hgraph



if __name__ == "__main__":

    #os.environ["CONTRACT_ID"] = ""
    #os.environ["NUM_SAMPLES"] = 50
    contract_id = os.getenv("CONTRACT_ID")
    num_samples = os.getenv("NUM_SAMPLES")

    generate_knowledge_graph("0USsyWjNe6nXWGkebYfYKC-VjQwJaA2ZS2HzALzgpgU", num_samples=50)
    tree = generate_context_tree("0USsyWjNe6nXWGkebYfYKC-VjQwJaA2ZS2HzALzgpgU")
    embd = JinaEmbedding()
    embedding = embd.encode("How does Arweave provide security?")
    root = tree.get_root()
    path, distance = root.traverse_to_leaf_with_beam(embedding=embedding, top_k=3)
    path2 = root.traverse_to_leaf_beam_search(embedding=embedding, top_k=3)
    #p, d = tree.best_leaf(embedding=embedding)

    print([node.summary for node in path])
    print(distance)