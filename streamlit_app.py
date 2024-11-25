import streamlit as st
from streamlit_cookies_controller import CookieController

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_huggingface import HuggingFaceEndpointEmbeddings

import cookie_history
from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_txt_files

import feedback

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG")
cookie_controller = CookieController()
def show_ui(qa, prompt_to_user="How may I help you?"):
    # Initialize the cookie controller
    cookie_history.load_cookie_chat_history(cookie_controller)
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Display feedback form only for assistant responses
            if message["role"] == "assistant" and message != st.session_state.messages[0]:
                feedback.display_feedback(message)

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.write_stream(ask_question(qa, prompt))
                # st.markdown(response)
            message = {"role": "assistant", "content": response}  # response.content
            st.session_state.messages.append(message)
            feedback.display_feedback(message)
            # Update the cookie with the new chat history
            cookie_controller.set('chat_history',  st.session_state.messages)

@st.cache_resource
def get_retriever(huggingfacehub_api_token=None):
    docs = load_txt_files()
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",
                                               huggingfacehub_api_token=huggingfacehub_api_token)
    return ensemble_retriever_from_docs(docs, embeddings=embeddings)


def get_chain(groq_api_key=None, huggingfacehub_api_token=None):
    ensemble_retriever = get_retriever(huggingfacehub_api_token=huggingfacehub_api_token)
    chain = create_full_chain(ensemble_retriever,
                              huggingfacehub_api_token=huggingfacehub_api_token,
                              chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain


def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def run():
    ready = True

    groq_api_key = st.session_state.get("GROQ_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    with st.sidebar:
        if not groq_api_key:
            groq_api_key = get_secret_or_input('GROQ_API_KEY', "GROQ_API_KEY",
                                               info_link="")
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
                                                           info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

    if not groq_api_key:
        st.warning("Missing GROQ_API_KEY")
        ready = False
    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False

    if ready:
        chain = get_chain(groq_api_key=groq_api_key, huggingfacehub_api_token=huggingfacehub_api_token)
        st.subheader("Ask me questions about this week's meal plan")
        show_ui(chain, "What would you like to know?")
    else:
        st.stop()


run()
