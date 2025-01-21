import streamlit as st
from streamlit_chat import message
import os
import PyPDF2
from dotenv import load_dotenv

def process_documents(uploaded_files):
    """
    1. This function processes multiple uploaded documents and combines their content.
    2. Creates chunks of text from the combined content.
    3. Next it creates the embeddings of the chunks.
    4. Finally, it creates a vector store from the embeddings and returns the vector store.
    """
    return ""

def main():
    # Load environment variables
    load_dotenv()

    # Configure Streamlit page
    st.set_page_config(layout="wide")

    # Sidebar
    with st.sidebar:
        st.title("Documents: ")
        # Multiple file upload in sidebar
        uploaded_files = st.file_uploader(
            "Upload documents:", 
            type=["pdf", "txt"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write("Uploaded files:")
            for file in uploaded_files:
                st.write(f"📄 {file.name}")

    # Main chat area
    st.title("Chat with your Documents")

    if uploaded_files:
        # Process all documents
        documents_data = process_documents(uploaded_files)

        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Create a container for chat messages
        chat_container = st.container()
        
        # Display chat messages in the container
        with chat_container:
            for msg in st.session_state.messages:
                message(msg["message"], is_user=msg["is_user"])

        # User input area at the bottom
        st.markdown("<br>" * 2, unsafe_allow_html=True)  # Add some space
        with st.form(key="chat_input", clear_on_submit=True):
            user_input = st.text_input("You:", key="user_input")
            col1, col2 = st.columns([6,1])  # Adjust the ratio as needed
            with col2:
                submit_button = st.form_submit_button("Send", use_container_width=True)

            # Style the button
            st.markdown(
                """
                <style>
                div.stButton > button:first-child {
                    border-radius: 20px;
                    padding: 10px 24px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.messages.append({"message": user_input, "is_user": True})

            # Generate chatbot response (replace with your actual logic)
            bot_response = f"You said: {user_input}"

            # Add bot message to chat history
            st.session_state.messages.append({"message": bot_response, "is_user": False})

            # Clear the user input field
            st.session_state.user_input = ""

            # Rerun the Streamlit app to update the chat display
            st.experimental_rerun()
    else:
        st.info("Please upload documents to start a conversation.")

if __name__ == "__main__":
    main()