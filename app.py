import streamlit as st
from streamlit_chat import message
import os
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text_from_documents(uploaded_files):
    """
    Process multiple uploaded documents and combine their content.
    Args:
        uploaded_files: List of uploaded file objects
    Returns:
        Dictionary containing combined text from all documents
    """
    print("Extracting text from documents...")
    combined_text = ""
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    combined_text += page.extract_text() + "\n\n"
                    
            elif file_extension == 'txt':
                combined_text += uploaded_file.getvalue().decode('utf-8') + "\n\n"
                
        print("Length of combined text: ", len(combined_text))
        return {"text": combined_text, "text_length": len(combined_text)}
    return None

def create_chunks(text):
    """
    Split a given text into smaller chunks.
    Args:
        text: The input text to be split into chunks
    Returns:
        List of chunks as strings
    """
    print("Creating chunks...")
    # Define the chunk size (adjust this value based on your requirements)
    chunk_size = 1000
    chunks_overlap=200

    character_text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap= chunks_overlap,
        length_function=len
    )

    # Split documents into chunks
    chunks = character_text_splitter.split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks

def create_embeddings(documents):
    """
    1. This function creates embeddings for a list of texts.
    2. It uses OpenAI's embedding model to generate embeddings for each chunk.
    3. It returns a list of embeddings.
    Args:
        texts: A list of strings representing the input texts.
        Returns:
            List of embeddings as numpy arrays.
    """
    print("Creating embeddings...")
    # Initialize OpenAI embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    db = FAISS.from_texts(texts=documents, embedding=embeddings)
    print("Vector Store Created.")
    return db


def process_documents(uploaded_files):
    """
    1. This function processes multiple uploaded documents and combines their content.
    2. Creates chunks of text from the combined content.
    3. Next it creates the embeddings of the chunks.
    4. Finally, it creates a vector store from the embeddings and returns the vector store.
    """
    # Extract text from the uploaded documents
    raw_text = extract_text_from_documents(uploaded_files)["text"]

    # Create chunks of text from the combined content
    chunks = create_chunks(raw_text)

    # Create embeddings from the chunks
    vector_store = create_embeddings(chunks)

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
            print("No of uploaded files: ", len(uploaded_files))
            st.write("Uploaded files:")
            for file in uploaded_files:
                st.write(f"📄 {file.name}")

    # Main chat area
    st.title("Chat with your Documents")

    if uploaded_files:
        # Create a placeholder for the loader
        loader_placeholder = st.empty()
        
        with loader_placeholder.container():
            with st.spinner('Processing your documents... This may take a moment...'):
                # Process all documents
                print("Processing documents...")
                documents_data = process_documents(uploaded_files)
        
        # Clear the loader after processing is complete
        loader_placeholder.empty()

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
                /* Custom CSS for the loader */
                .processing-loader {
                    text-align: center;
                    padding: 20px;
                    background: #f0f2f6;
                    border-radius: 10px;
                    margin: 20px 0;
                }
                .stSpinner {
                    text-align: center;
                    margin: 20px 0;
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