import os
import click
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import index
from chatbot.config.config import ConfigHelper
from chatbot.vectorstore.vectorstore import VectorStoreManager
from chatbot.doc.loader import DocLoader


class DocIndexer():

    def __init__(self, config: ConfigHelper):
        self.vector_store_mgr = VectorStoreManager(config)
        self.loader = DocLoader(config)

    def index(self):
        # load docs from confluence
        docs = self.loader.load()

        # get vectore store impl and record manager
        vector_store = self.vector_store_mgr.get_vector_store()
        record_manager = self.vector_store_mgr.get_record_manager()

        # split docs to chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)

        # do index for the docs
        index(all_splits, record_manager, vector_store,
              cleanup="incremental", source_id_key="source")


@click.command()
@click.option("--env-file-path", required=False, default=None, help="env file path contains the required os env vars")
def main(env_file_path):
    doc_indexer = DocIndexer(ConfigHelper(env_file_path))
    doc_indexer.index()


if __name__ == "__main__":
    main()
