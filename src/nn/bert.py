import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


class BertEmbedding:
    def __init__(self, model_name='bert-base-uncased', random_seed=42):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    def generate_embeddings(self, texts, add_special_tokens=True, cls_only=False, max_length=None):
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            padding='max_length' if max_length else True,
            max_length=max_length,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=add_special_tokens
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state

        if cls_only:
            cls_embeddings = word_embeddings[:, 0, :]
            return cls_embeddings
        else:
            return word_embeddings

    @staticmethod
    def cosim(vec1, vec2):
        return cosine_similarity(vec1.numpy(), vec2.numpy())

    @staticmethod
    def maxsim(query_embeddings, doc_embeddings):
        query_embeddings_np = query_embeddings.numpy()
        doc_embeddings_np = doc_embeddings.numpy()

        dists = []
        for d in doc_embeddings_np:
            qv = query_embeddings_np[0,:,:]
            similarity_scores = cosine_similarity(qv, d)
            # Find the maximum cosine similarity for each query embedding
            max_similarities = similarity_scores.max(axis=1)
            # Calculate the sum of the maximum cosine similarities
            sum_max_similarities = max_similarities.sum()
            dists.append(sum_max_similarities)
        return dists