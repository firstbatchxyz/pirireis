import time

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from decouple import config

class DBWorker:
    def __init__(self):
        self._db = None
        self.mongo_uri = config("MONGO_URI")
        self.dbname = "dria"
        self.index_col = "dbs"
        self.graph_col = "graphs"
        self.client = MongoClient(self.mongo_uri, server_api=ServerApi('1'))

    def get_knowledge_graph(self, contract_id) -> str:
        data = self.client.get_database(self.dbname).get_collection(self.graph_col).find_one(
            {"contract_id": contract_id}, {"knowledge_graph": 1, "_id": 0}
        )
        return data["knowledge_graph"]

    def update_knowledge_graph(self, contract_id: str, knowledge_graph: str):
        self.client.get_database(self.dbname).get_collection(self.graph_col).update_one(
            {"contract_id": contract_id},
            {"$set": {
                "knowledge_graph": knowledge_graph,
                "timestamp": int(time.time()),
            }},
            upsert=True
        )

    def get_context_tree(self, contract_id) -> str:
        data = self.client.get_database(self.dbname).get_collection(self.graph_col).find_one(
            {"contract_id": contract_id}, {"context_tree": 1, "_id": 0}
        )
        return data["context_tree"]

    def update_context_tree(self, contract_id: str, context_tree: str):
        self.client.get_database(self.dbname).get_collection(self.graph_col).update_one(
            {"contract_id": contract_id},
            {"$set": {
                "context_tree": context_tree,
                "timestamp": int(time.time()),
            }},
            upsert=True
        )