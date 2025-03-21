import warnings
from langchain_community.llms import VLLMOpenAI
from langchain import hub
from langchain.chains import LLMChain
from langchain.output_parsers import RegexParser
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
        
        retrieved_docs = self.vectore_store.similarity_search(question)
        doc_contents = [doc.page_content for doc in retrieved_docs]
        # check if retrieved docs are relevant to the user question
        doc_relevance_prompt = hub.pull("rlm/rag-document-relevance")
        output_parser = RegexParser(
            regex=r"Answer: (\d)",
            output_keys=["score"],
            default_output_key="score"
        )
        llm_chain = LLMChain(prompt=doc_relevance_prompt, llm=self.llm, output_parser=output_parser)
        result = llm_chain.invoke({"input": {"documents": doc_contents, "question": question}})
        if self.config.is_debug_logging_enabled():
            print(f"docs are relevant to user question: {result}")
        prompt = None
        if int(result['text']['score']) == 1:
            # it means the retrieved docs are relevant to the user question, then call llm with rag prompt
            # prompt template can be found from https://smith.langchain.com/hub/rlm/rag-prompt-mistral
            # TODO: we can introduce a prompt selector/resolver to choose different template by model
            prompt = hub.pull("rlm/rag-prompt-mistral", include_model=True)
            docs_content = "\n".join(doc_contents)
            prompt = prompt.invoke({"question": question, "context": docs_content})
        else:
            prompt = question
            
        return self.llm.invoke(prompt)
