import streamlit as st
from streamlit_cookies_controller import CookieController

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_txt_files

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG")
# Initialize the cookie controller
controller = CookieController()
# Retrieve chat history from cookies
chat_history = controller.get('chat_history')
if chat_history is None:
    chat_history = []


def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user, "feedback": {}}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Display feedback form only for assistant responses
            if message["role"] == "assistant" and message != st.session_state.messages[0]:
                display_feedback(message)

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
            message = {"role": "assistant", "content": response,
                       "feedback": {"thumbs": None, "stars": None, "faces": None, "text": None}}  # response.content
            st.session_state.messages.append(message)
            display_feedback(message)
            # Update the cookie with the new chat history
            controller.set('chat_history',  st.session_state.messages)


def display_feedback(message):
    if "feedback" not in message:
        message["feedback"] = {"thumbs": None, "stars": None, "faces": None, "text": None}
    print(f"message {message}")

    feedback = message["feedback"]
    feedback_types = ["thumbs", "stars", "faces", "text"]

    # Initialize session state for feedback components if they exist in the message
    for feedback_type in feedback_types:
        key = f"{feedback_type}_feedback_{id(message)}"
        st.session_state[key] = feedback.get(feedback_type, None)
        if feedback_type == "text":
            st.text_input(
                label="[Optional] Please provide additional details",
                key=key,
                on_change=handle_feedback,
                args=(message,)
            )
        else:
            st.feedback(
                options=feedback_type,
                key=key,
                on_change=handle_feedback,
                args=(message,)
            )

    # Display feedback form only for assistant responses
    # st.session_state.setdefault("thumbs_feedback", None)
    # st.session_state.setdefault("faces_feedback", None)
    # st.session_state.setdefault("text_feedback", "")
    #
    # st.feedback(options="thumbs", key=f"thumbs_feedback_{len(st.session_state.messages)}",
    #             on_change=handle_feedback)
    # st.feedback(options="faces", key=f"faces_feedback_{len(st.session_state.messages)}", on_change=handle_feedback)
    # st.text_input(
    #     label="[Optional] Please provide additional details",
    #     key=f"text_feedback_{len(st.session_state.messages)}",
    #     on_change=handle_feedback,
    # )


def handle_feedback(message):
    # Update feedback directly within the message object
    message["feedback"] = {
        "thumbs": st.session_state.get(f"thumbs_feedback_{id(message)}"),
        "stars": st.session_state.get(f"stars_feedback_{id(message)}"),
        "faces": st.session_state.get(f"faces_feedback_{id(message)}"),
        "text": st.session_state.get(f"text_feedback_{id(message)}")
    }
    return message


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


def load_chat_history():
    st.write("Cookies: ", chat_history)
    # Display existing chat messages
    for message in chat_history:
        with st.chat_message(message['role']):
            st.write(message['content'])
            display_feedback(message)


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
        load_chat_history()
        show_ui(chain, "What would you like to know?")
    else:
        st.stop()


run()
