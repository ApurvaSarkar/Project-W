import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Page config
pageicon = Image.open("files/cat.png")
st.set_page_config(page_title="Project W", page_icon=pageicon, layout="wide")
st.title("Project W")

def get_respons(user_input):
    return "I'm still under development, but I'm getting better at responding to your questions. How can I help you today?"

# Using session state to fix the Streamlit reloading issue
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! What would you like to know?"),
    ]

# Getting text vectors from URLs in document form
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents_chunks = text_splitter.split_documents(documents)
    return documents_chunks

# Get PDF chunks
def get_pdf_chunks(uploaded_file):
    text = ""
    for pdf in uploaded_file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

    
# Sidebar
with st.sidebar:
    st.header("Settings")
    website_links = st.text_area("Website links (separate with commas)", value="")
    uploaded_file = st.file_uploader("Upload your pdf or csv here:", accept_multiple_files=True)
    # To center the submit button
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        submit_button = st.button("Submit")

# Submit button logic 
if submit_button:
    with st.spinner("Processing"):
        text_chunks = get_pdf_chunks(uploaded_file)
        document_chunks = []
        if website_links:
            url_list = [url.strip() for url in website_links.split(",")]
            url_list = [url for url in url_list if url]  # Keep only non-empty strings
            for url in url_list:
                url_chunks = get_vectorstore_from_url(url)
                document_chunks.extend(url_chunks)

    with st.sidebar:
                st.write(document_chunks)
                st.write(text_chunks)

# Enable chat section regardless of website links
user_query = st.chat_input("You're Question")
if user_query is not None and user_query != "":
    response = get_respons(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                if isinstance(message.content, str):
                    st.write(message.content)
                else:
                    st.write_stream(message.content)
        elif isinstance(message, HumanMessage):
                    with st.chat_message("User"):
                        st.write(message.content)

                
