import streamlit as st

def display_feedback(message):
    if "feedback" not in message:
        message["feedback"] = {"thumbs": 1, "text": None}

    feedback = message["feedback"]
    feedback_types = ["thumbs", "text"]

    # Initialize session state for feedback components if they exist in the message
    for feedback_type in feedback_types:
        key = f"{feedback_type}_feedback_{id(message)}"
        if key not in st.session_state:
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
                args=(message,),

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
        "text": st.session_state.get(f"text_feedback_{id(message)}")
    }
    # return message

def main():
    display_feedback({"role": "assistant", "content": "Hello, how can I help you today?", "feedback": {"thumbs": 1,}})

if __name__ == "__main__":
    main()