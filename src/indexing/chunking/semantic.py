from typing import Literal

from nltk.tokenize import sent_tokenize
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from settings import settings
from utils.context import embedding_model_context

# import nltk
# nltk.download('punkt')


class SemanticTextSplitter:
    """A class to cluster text data using embeddings and clustering algorithms."""

    def __init__(
        self,
        embedding_model_path: str = settings.EMBEDDING_MODEL_PATH,
        device: str = settings.DEVICE,
        algorithm: Literal["dbscan", "agglomerative"] = "dbscan",
    ):
        self.embedding_model_path = embedding_model_path
        self.device = device
        match algorithm:
            case "dbscan":
                self.clustering_model = DBSCAN(eps=0.7, min_samples=1, metric="cosine")
            case "agglomerative":
                self.clustering_model = AgglomerativeClustering(
                    distance_threshold=0.6,
                    metric="cosine",
                    linkage="average",
                )
            case _:
                raise ValueError("Invalid algorithm")

    def split_text(self, text: str) -> list[str]:
        if len(text) == 0:
            raise ValueError("text is empty")

        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return sentences

        # Generate embeddings
        with embedding_model_context(
            path=self.embedding_model_path, device=self.device
        ) as model:
            batch_size = 32
            embeddings = []
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                embeds = model.encode(batch, batch_size=batch_size)["dense_vecs"]
                embeddings.extend(embeds)

        # Perform clustering
        # todo: explore other clustering algorithms
        # todo: explore using a sliding window approach
        labels = self.clustering_model.fit_predict(embeddings)

        # Group texts by cluster labels
        chunks = {}
        for idx, label in enumerate(labels):
            if label not in chunks:
                chunks[label] = []
            chunks[label].append(sentences[idx])

        # Filter out noise (label -1) and concatenate sentences in each cluster
        return [
            " ".join(sentences) for label, sentences in chunks.items() if label != -1
        ]
