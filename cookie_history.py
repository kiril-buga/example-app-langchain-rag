import streamlit as st
from streamlit_cookies_controller import CookieController
import feedback

def load_cookie_chat_history(controller):
    # Initialize the cookie controller
    # controller = CookieController()
    # Retrieve chat history from cookies
    chat_history = controller.get('chat_history')
    # Initialize chat history if it doesn't exist
    if chat_history is None:
        chat_history = []
    elif "messages" not in st.session_state.keys():
        # st.session_state.messages = chat_history
        st.session_state.messages = []
        for message in chat_history:
            st.session_state.messages.append(message)
    st.write("Cookies: ", chat_history)
    # st.write("Session state: ", st.session_state.messages)
    # Append messages to session state
    # for message in chat_history:
    #     st.session_state.messages.append(message)
    return controller