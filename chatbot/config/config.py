import os
import dotenv


class ConfluenceLoaderConfig():

    def __init__(self, url, token, space_key):
        self._url = url
        self._token = token
        self._space_key = space_key

    @property
    def url(self):
        return self._url

    @property
    def token(self):
        return self._token

    @property
    def space_key(self):
        return self._space_key


class LLMConfig():

    def __init__(self, api_url, api_key, model_name):
        self._api_url = api_url
        self._api_key = api_key
        self._model_name = model_name

    @property
    def api_url(self):
        return self._api_url

    @property
    def api_key(self):
        return self._api_key

    @property
    def model_name(self):
        return self._model_name


class EmbeddingModelConfig():

    def __init__(self, api_url, api_key):
        self._api_url = api_url
        self._api_key = api_key

    @property
    def api_url(self):
        return self._api_url

    @property
    def api_key(self):
        return self._api_key


class ChromaDbConfig():

    def __init__(self, collection_name, persist_directory):
        self._collection_name = collection_name
        self._persist_directory = persist_directory

    @property
    def collection_name(self):
        return self._collection_name

    @property
    def persist_directory(self):
        return self._persist_directory


class SqlLiteConfig():
    def __init__(self, db_file_path):
        self._db_file_path = db_file_path

    @property
    def db_file_path(self):
        return self._db_file_path


class ConfigHelper():

    def __init__(self, env_file_path=None):
        if env_file_path:
            dotenv.load_dotenv(env_file_path)
        else:
            dotenv.load_dotenv()

    @classmethod
    def get_llm_config(self) -> LLMConfig:
        return LLMConfig(
            self._get_env_var("LLM_API_URL"),
            self._get_env_var("LLM_API_KEY"),
            self._get_env_var("LLM_MODEL_NAME")
        )

    @classmethod
    def get_embedding_model_config(self) -> EmbeddingModelConfig:
        return EmbeddingModelConfig(
            self._get_env_var("EMBEDDING_API_URL"),
            self._get_env_var("EMBEDDING_API_KEY")
        )

    @classmethod
    def get_confluence_loader_config(self) -> ConfluenceLoaderConfig:
        return ConfluenceLoaderConfig(
            self._get_env_var("CONLFUENCE_URL"),
            self._get_env_var("CONLFUENCE_TOKEN"),
            self._get_env_var("CONLFUENCE_SPACE_KEY")
        )

    @classmethod
    def get_chroma_db_config(self) -> ChromaDbConfig:
        return ChromaDbConfig(
            self._get_env_var("CHROMA_DB_COLLECTION_NAME"),
            self._get_env_var("CHROMA_DB_PERSISTENT_DIR"),
        )

    @classmethod
    def get_sqllite_config(self) -> SqlLiteConfig:
        return SqlLiteConfig(
            self._get_env_var("SQLLITE_DB_FILE_PATH")
        )

    @classmethod
    def is_debug_logging_enabled(self):
        return bool(self._get_env_var("DEBUG_LOGGING", default=False))

    def _get_env_var(key, default=None):
        value = os.getenv(key, default)
        if value is None:
            error_message = f"{key} environment variable is not set."
            raise ValueError(error_message)
        return value
