import warnings
from langchain_community.llms import VLLMOpenAI
from langchain import hub

from chatbot.config.config import ConfigHelper
from chatbot.vectorstore.vectorstore import VectorStoreManager


class RetrievalAndGeneration():

    def __init__(self, config: ConfigHelper):
        # disable warning log from langsmith
        warnings.filterwarnings("ignore")
        llm_conf = config.get_llm_config()
        self.llm = llm = VLLMOpenAI(
            openai_api_key=llm_conf.api_key,
            openai_api_base=f"{llm_conf.api_url}/v1",
            model_name=llm_conf.model_name,
            max_tokens=4096,
        )
        self.vectore_store = VectorStoreManager(config).get_vector_store()

    def query(self, question):
        if not question:
            return "No user input found"
        # define prompt for question-answering
        prompt = hub.pull("rlm/rag-prompt")
        retrieved_docs = self.vectore_store.similarity_search(question)
        docs_content = "\n".join(doc.page_content for doc in retrieved_docs)
        print(docs_content)
        prompt = prompt.invoke({"question": question, "context": docs_content})
        return self.llm.invoke(prompt)
