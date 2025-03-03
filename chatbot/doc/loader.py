from langchain_community.document_loaders import ConfluenceLoader
from langchain_core.documents import Document
from chatbot.config.config import ConfigHelper


class DocLoader:

    def __init__(self, config: ConfigHelper):
        self.config = config

    def load(self) -> list[Document]:
        # load docs from confluence space
        loader_conf = self.config.get_confluence_loader_config()
        loader = ConfluenceLoader(
            url=loader_conf._url,
            token=loader_conf.token,
            space_key=loader_conf.space_key
        )
        return loader.load()
