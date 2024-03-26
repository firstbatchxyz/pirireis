from typing import Dict, List, Optional
from enum import Enum
from abc import ABC, abstractmethod
from openai import OpenAI
import requests
import tiktoken
from decouple import config

DEFAULT_MODEL_ID = 'BAAI/bge-large-en-v1.5'

class ModelEnum(str, Enum):
    jina_embeddings_v2_base_en = 'jinaai/jina-embeddings-v2-base-en'
    jina_embeddings_v2_small_en = 'jinaai/jina-embeddings-v2-small-en'
    text_embedding_ada_002 = 'openai/text-embedding-ada-002'
    text_embedding_3_small = 'openai/text-embedding-3-small'
    text_embedding_3_large = 'openai/text-embedding-3-large'
    bge_large_en_v1_5 = 'BAAI/bge-large-en-v1.5'


Dimensions = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 512,
    "text-embedding-3-large": 1024
}


class EmbeddingCore(ABC):

    @property
    def dimension(self):
        return self.dimension

    @dimension.setter
    def dimension(self, value):
        self._dimension = value

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str], model: str) -> List[List[float]]:
        pass


class JinaEmbedding(EmbeddingCore):

    def __init__(self):
        self.dimension = 768
        self.url = 'https://api.jina.ai/v1/embeddings'
        self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': config("JINAAI_API_KEY")
        }

    def encode_batch(self, texts: List[str], model: str):

        texts = [self.encoding.decode(self.encoding.encode(fr)[:8191]) for fr in texts]
        data = {
            'input': texts,
            'model': model
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        try:
            d = response.json()
            e = [dd['embedding'] for dd in d["data"]]
        except Exception as e:
            print("Invalid response from Jina API {}".format(e))
        return e

    def encode(self, text: str):
        data = {
            'input': [text],
            'model': 'jina-embeddings-v2-base-en'
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        try:
            d = response.json()
            e = [dd['embedding'] for dd in d["data"]]
        except Exception as e:
            print("Invalid response from Jina API {}".format(e))
        return e[0]


class JinaEmbeddingSmall(EmbeddingCore):

    def __init__(self):
        self.dimension = 768
        self.url = 'https://api.jina.ai/v1/embeddings'
        self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': config("JINAAI_API_KEY")
        }

    def encode_batch(self, texts: List[str], model: str):

        texts = [self.encoding.decode(self.encoding.encode(fr)[:8191]) for fr in texts]
        data = {
            'input': texts,
            'model': model
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        try:
            d = response.json()
            e = [dd['embedding'] for dd in d["data"]]
        except Exception as e:
            print("Invalid response from Jina API {}".format(e))
        return e

    def encode(self, text: str):
        data = {
            'input': [text],
            'model': 'jina-embeddings-v2-small-en'
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        try:
            d = response.json()
            e = [dd['embedding'] for dd in d["data"]]
        except Exception as e:
            print("Invalid response from Jina API {}".format(e))
        return e[0]


class OpenAIEmbedding(EmbeddingCore):
    def __init__(self):
        self.client = OpenAI(api_key=config("OPENAI_API_KEY"))
        self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

    def encode(self, text: str):
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )

        return response.data[0].embedding

    def encode_batch(self, texts: List[str], model: str):
        texts = [self.encoding.decode(self.encoding.encode(fr)[:8191]) for fr in texts if fr != ""]
        response = self.client.embeddings.create(
            input=texts,
            model=model,
            dimensions=Dimensions.get(model, 1536)
        )
        return [r.embedding for r in response.data]


class DeepInfraCore:
    def __init__(self,  deepinfra_api_token: str, model_id: str):
        self.client = requests.Session()  # Using Session for connection pooling
        self.model_id = model_id
        self.deepinfra_api_token = deepinfra_api_token
        self.model_kwargs = None

    def call(self, prompt: List[str]) -> Dict:
        url = f"https://api.deepinfra.com/v1/inference/{self.model_id}"
        headers = {
            'Authorization': f"Bearer {self.deepinfra_api_token}",
            'Content-Type': 'application/json',
        }
        response = self.client.post(url, headers=headers, json={"inputs": prompt})
        response.raise_for_status()  # Raises HTTPError, if one occurred
        return response.json()


class DeepInfraEmbedding(EmbeddingCore):

    def __init__(self, model=DEFAULT_MODEL_ID):
        self.client = DeepInfraCore(deepinfra_api_token=config("DEEPINFRA_API_KEY"), model_id=model)
        self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

    def encode(self, text: str):
        response = self.client.call([text])
        if response['embeddings']:
            return [float(x) for x in response['embeddings'][0]]
        else:
            raise ValueError("No embeddings returned")

    def encode_batch(self, texts: List[str], model: str):
        texts = [self.encoding.decode(self.encoding.encode(fr)[:4096]) for fr in texts if fr != ""]
        response = self.client.call(texts)
        if response['embeddings']:
            return [[float(val) for val in x] for x in response['embeddings']]
        else:
            raise ValueError("No embeddings returned")
