import os
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain.indexes import SQLRecordManager
from langchain_openai import OpenAIEmbeddings
from langchain_core.indexing import RecordManager
from chatbot.config.config import ConfigHelper


class VectorStoreManager():

    def __init__(self, config: ConfigHelper):
        self.config = config

    def get_vector_store(self) -> VectorStore:
        # get chroma db config to init vector store
        chroma_conf = self.config.get_chroma_db_config()
        vector_store = Chroma(
            collection_name=chroma_conf.collection_name,
            embedding_function=self._get_embedding_model(),
            persist_directory=chroma_conf.persist_directory
        )

        return vector_store

    def get_record_manager(self) -> RecordManager:
        # initialize record manager
        sqllite_conf = self.config.get_sqllite_config()
        record_manager = SQLRecordManager(
            "chroma/index", db_url=f"sqlite://{sqllite_conf.db_file_path}")

        if not os.path.exists(sqllite_conf.db_file_path):
            record_manager.create_schema()

        return record_manager

    def _get_embedding_model(self):
        # generate embeddings
        embedding_conf = self.config.get_embedding_model_config()
        embeddings_model = OpenAIEmbeddings(
            openai_api_base=embedding_conf.api_url, openai_api_key=embedding_conf.api_key, chunk_size=32)

        return embeddings_model
