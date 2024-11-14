import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_txt_files

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG")


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
        message = {"role": "assistant", "content": response, "feedback": {"thumbs": None, "faces": None, "text": None}} # response.content
        st.session_state.messages.append(message)

    # Feedback components
    # st.session_state.setdefault("thumbs_feedback", None)
    # st.session_state.setdefault("faces_feedback", None)
    # st.session_state.setdefault("text_feedback", "")
    # with st.form(key="feedback_form"):
    #     st.feedback(options="thumbs", key="thumbs_feedback", on_change=handle_feedback)
    #     st.feedback(options="faces", key="faces_feedback", on_change=handle_feedback)
    #     st.text_input(
    #         label="[Optional] Please provide additional details",
    #         key="text_feedback",
    #         on_change=handle_feedback,
    #     )

def display_feedback(message):
    if "feedback" not in message:
        message["feedback"] = {"thumbs": None, "faces": None, "text": None}
    print(f"message {message}")

    feedback = message["feedback"]

    # Initialize session state for feedback components if they exist in the message
    if feedback["thumbs"] is not None:
        st.session_state[f"thumbs_feedback_{id(message)}"] = feedback["thumbs"]
    if feedback["faces"] is not None:
        st.session_state[f"faces_feedback_{id(message)}"] = feedback["faces"]
    if feedback["text"] is not None:
        st.session_state[f"text_feedback_{id(message)}"] = feedback["text"]

    # Display feedback form with unique keys based on the message's object ID
    st.feedback(
        options="thumbs",
        key=f"thumbs_feedback_{id(message)}",
        on_change=handle_feedback,
        args=(message,)
    )
    st.feedback(
        options="faces",
        key=f"faces_feedback_{id(message)}",
        on_change=handle_feedback,
        args=(message,)
    )
    st.text_input(
        label="[Optional] Please provide additional details",
        key=f"text_feedback_{id(message)}",
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
        "faces": st.session_state.get(f"faces_feedback_{id(message)}"),
        "text": st.session_state.get(f"text_feedback_{id(message)}")
    }
    # thumbs = st.session_state.get(f"thumbs_feedback_{index}")
    # faces = st.session_state.get(f"faces_feedback_{index}")
    # text = st.session_state.get(f"text_feedback_{index}")
    #
    # # Store feedback in the specific message's feedback attribute
    # st.session_state.messages[index]["feedback"] = {
    #     "thumbs": thumbs,
    #     "faces": faces,
    #     "text": text
    # }
    #
    # # Display feedback as confirmation below the message
    # st.markdown(f"**Thumbs Feedback:** {thumbs}")
    # st.markdown(f"**Faces Feedback:** {faces}")
    # st.markdown(f"**Additional Comments:** {text}")

@st.cache_resource
def get_retriever(huggingfacehub_api_token=None):
    docs = load_txt_files()
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", huggingfacehub_api_token=huggingfacehub_api_token)
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
