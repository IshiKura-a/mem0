import logging
from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase
from FlagEmbedding import BGEM3FlagModel


class BGEEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "multi-qa-MiniLM-L6-cos-v1"

        self.model = BGEM3FlagModel(self.config.model, **self.config.model_kwargs)

        self.config.embedding_dims = self.config.embedding_dims

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        return self.model.encode(text)['dense_vecs']
