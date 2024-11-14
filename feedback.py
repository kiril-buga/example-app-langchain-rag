import streamlit as st

def handle_feedback():
    selected = st.feedback(options="thumbs", key=None, disabled=False, on_change=None, args=None, kwargs=None)
    if selected is not None:
        st.markdown(f"You selected {selected} star(s).")

def main():
    handle_feedback()

if __name__ == "__main__":
    main()