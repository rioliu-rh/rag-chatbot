import streamlit as st
from chatbot.llm.rag import RetrievalAndGeneration
from chatbot.config.config import ConfigHelper


@st.cache_resource
def get_llm():
    llm = RetrievalAndGeneration(ConfigHelper())
    return llm


llm = get_llm()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "What can I do for you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("What can I do for you?"):
    # Add prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get answer
    result = llm.query(prompt)

    # Add answer
    st.chat_message("assistant").write(result)
    st.session_state.messages.append(
        {"role": "assistant", "content": result})
