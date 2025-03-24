import warnings
from langchain_community.llms import VLLMOpenAI
from langchain import hub
from chatbot import llm
from chatbot.config.config import ConfigHelper
from chatbot.vectorstore.vectorstore import VectorStoreManager


class RetrievalAndGeneration():

    def __init__(self, config: ConfigHelper):
        self.config = config
        # disable warning log from langsmith
        warnings.filterwarnings("ignore")
        llm_conf = self.config.get_llm_config()
        self.llm = VLLMOpenAI(
            openai_api_key=llm_conf.api_key,
            openai_api_base=f"{llm_conf.api_url}/v1",
            model_name=llm_conf.model_name,
            max_tokens=4096,
        )
        self.vectore_store = VectorStoreManager(self.config).get_vector_store()

    def query(self, question):
        if not question:
            return "No user input found"

        # it means the retrieved docs are relevant to the user question, then call llm with rag prompt
        # prompt template can be found from https://smith.langchain.com/hub/rlm/rag-prompt-mistral
        # TODO: we can introduce a prompt selector/resolver to choose different template by model
        retrieved_docs = self.vectore_store.similarity_search(question)
        doc_contents = [doc.page_content for doc in retrieved_docs]
        if self.config.is_debug_logging_enabled():
            print(f"docs are relevant to user question: {retrieved_docs}")
        prompt = hub.pull("rlm/rag-prompt-mistral", include_model=True)
        prompt = prompt.invoke({"question": question, "context": doc_contents})
            
        return self.llm.invoke(prompt)
