import streamlit as st
from streamlit_chat import message
import os
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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
    # Define the chunk size
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

def create_conversation(vector_store):
    """
    1. This function creates a conversation chain with a vector store.
    2. It uses a chat model to generate responses based on user input.
    3. It returns a conversation chain.
    """
    print("Creating conversation...")
    # Initialize OpenAI chat model
    chat_model = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.7, model_name="gpt-4o")
    conversation_memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vector_store.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=chat_model, retriever=retriever, memory=conversation_memory)
    print("Conversation created.")
    return conversation_chain


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

    # Create a conversation chain with the vector store
    conversation_chain = create_conversation(vector_store)
    return conversation_chain

def main():
    # Load environment variables
    load_dotenv()

    # Configure Streamlit page
    st.set_page_config(layout="wide")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
        
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

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

    if uploaded_files and not st.session_state.documents_processed:
        with st.spinner('Processing your documents... This may take a moment...'):
            # Process all documents
            print("Processing documents...")
            st.session_state.conversation_chain = process_documents(uploaded_files)
            st.session_state.documents_processed = True

    if uploaded_files:
        # Create a container for chat messages
        chat_container = st.container()
                
        with chat_container:
            st.markdown("""
                <style>
                .chat-container {
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: #212121;
                    border-radius: 10px;
                }
                .chat-message {
                    padding: 1.5rem;
                    margin: 0;
                    display: flex;
                    align-items: flex-start;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                .chat-message.user {
                    background-color: #2A2A2A;
                }
                .chat-message.assistant {
                    background-color: #212121;
                }
                .avatar {
                    width: 32px;
                    height: 32px;
                    margin-right: 1rem;
                    border-radius: 0.2rem;
                }
                .message-content {
                    flex-grow: 1;
                    font-size: 16px;
                    line-height: 1.5;
                    color: #FFFFFF;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                }
                .message-content p {
                    margin: 0;
                }
                .message-content code {
                    background-color: #3A3A3A;
                    color: #E0E0E0;
                    padding: 0.2em 0.4em;
                    border-radius: 3px;
                    font-size: 85%;
                    font-family: Monaco, 'Courier New', monospace;
                }
                .message-content pre {
                    background-color: #3A3A3A;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    overflow-x: auto;
                    color: #E0E0E0;
                }
                
                /* Style for links in messages */
                .message-content a {
                    color: #00ADB5;
                    text-decoration: none;
                }
                .message-content a:hover {
                    text-decoration: underline;
                }
                
                /* Scrollbar styling */
                ::-webkit-scrollbar {
                    width: 10px;
                    height: 10px;
                }
                ::-webkit-scrollbar-track {
                    background: #212121;
                    border-radius: 5px;
                }
                ::-webkit-scrollbar-thumb {
                    background: #888;
                    border-radius: 5px;
                }
                ::-webkit-scrollbar-thumb:hover {
                    background: #555;
                }
                
                /* Input box styling */
                .stTextInput > div > div > input {
                    background-color: #2A2A2A;
                    color: #FFFFFF;
                    border: 1px solid #404040;
                }
                .stTextInput > div > div > input:focus {
                    border-color: #00ADB5;
                    box-shadow: 0 0 0 1px #00ADB5;
                }
                
                /* Markdown text color */
                .st-emotion-cache-1y4p8pa {
                    color: #FFFFFF;
                }
                
                /* Adjust the page background */
                .main {
                    background-color: #1A1A1A;
                }
                </style>
            """, unsafe_allow_html=True)

            # Display messages
            for idx, msg in enumerate(st.session_state.messages):
                # Determine message type and avatar
                is_user = msg["is_user"]
                avatar = "👤" if is_user else "🤖"
                message_type = "user" if is_user else "assistant"

                # Create message container
                st.markdown(f"""
                    <div class="chat-message {message_type}">
                        <div class="avatar">{avatar}</div>
                        <div class="message-content">
                            {msg["message"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='height: 1px;'></div>", unsafe_allow_html=True)

        def handle_message_submit():
            if st.session_state.user_input.strip():
                # Add user message to chat history
                st.session_state.messages.append(
                    {"message": st.session_state.user_input, "is_user": True}
                )

                # Generate chatbot response
                bot_response = st.session_state.conversation_chain.invoke({"question":st.session_state.user_input})

                if bot_response:
                    bot_response = bot_response["answer"]
                else:
                    bot_response = "I'm sorry, I don't understand. Can you please rephrase your question?"
                    
                # Add bot message to chat history
                st.session_state.messages.append(
                    {"message": bot_response, "is_user": False}
                )

                # Clear the input
                st.session_state.user_input = ""

        st.markdown("""
            <style>
            /* Main container structure */
            .main .block-container {
                padding-bottom: 0 !important;
                padding-top: 2rem !important;
                max-width: 100% !important;
            }
            
            /* Chat container */
            .chat-container {
                max-width: 800px;
                margin: 0 auto;
                background-color: #212121;
                border-radius: 10px;
                overflow-y: auto;
                height: calc(100vh - 120px);
            }
            
            /* Input container */
            .input-container {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background-color: #1A1A1A;
                border-top: 1px solid #404040;
                padding: 1rem;
                z-index: 1000;
            }
            
            /* Content wrapper */
            .content-wrapper {
                max-width: 800px;
                margin: 0 auto;
                position: relative;
            }
            
            /* Input field styling */
            .stTextInput > div > div > input {
                background-color: #2A2A2A !important;
                color: #FFFFFF !important;
                border: 1px solid #404040 !important;
                border-radius: 10px !important;
                padding: 15px !important;
                font-size: 16px !important;
                line-height: 1.5 !important;
                width: 100% !important;
                box-shadow: 0 0 10px rgba(0,0,0,0.1) !important;
            }
            
            /* Focus state */
            .stTextInput > div > div > input:focus {
                border-color: #00ADB5 !important;
                box-shadow: 0 0 0 2px rgba(0,173,181,0.2) !important;
                outline: none !important;
            }
            
            /* Placeholder */
            .stTextInput > div > div > input::placeholder {
                color: #888888 !important;
            }
            
            /* Reset streamlit elements */
            .stTextInput {
                position: static !important;
                margin: 0 !important;
            }
            
            .stTextInput > div {
                background-color: transparent !important;
            }
            
            /* Hide default streamlit components */
            .stApp > header {
                display: none !important;
            }
            
            /* App background */
            .stApp {
                background-color: #1A1A1A;
            }
            
            /* Remove bottom padding/margin from elements */
            .element-container {
                margin-bottom: 0 !important;
            }
            
            footer {
                display: none !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Create input container
        st.markdown('<div class="input-container"><div class="content-wrapper">', unsafe_allow_html=True)

        # Create the input field
        st.text_input(
            "",
            placeholder="Send a message...",
            key="user_input",
            on_change=handle_message_submit,
            args=(),
            kwargs={},
        )

    else:
        st.info("Please upload documents to start a conversation.")

if __name__ == "__main__":
    main()