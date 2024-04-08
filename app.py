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

#page config
pageicon = Image.open("files/cat.png")
st.set_page_config(page_title="project W", page_icon=pageicon, layout="wide")
st.title("Project W")

#AI respons
def get_respons(user_input):
    return "I dont know"

#using session state to fix the streamlit reloading issue
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="Hi what would you like to know?"),
    ]

#getting text vectors from urls in document form
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    #splliting documents into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    documents_chunks = text_splitter.split_documents(documents)
    return documents_chunks

#get pdf text
def get_pdf_text(uploaded_file):
    text =""
    for pdf in uploaded_file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
#converting raw pdf text to chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n" ,
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Sidebar
with st.sidebar:
    st.header("Setting")

    website_links = st.text_area("Website links (separate with commas)", value="")
    uploaded_file = st.file_uploader("Upload your pdf or csv here:", accept_multiple_files=True)
    #To center the submit button
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        submit_button = st.button("Submit")


#submit button logic
if submit_button:
    with st.spinner("processing"):
        #get text chunks from pdf
        raw_text = get_pdf_text(uploaded_file)
        text_chunks = get_text_chunks(raw_text)
        st.sidebar.write(text_chunks)

        #Checks for URl
        if website_links is None or website_links == "":
            st.info("Please enter at least one website URL")
        else:
            # Split the URLs into a list, handling empty entries
            url_list = [url.strip() for url in website_links.split(",")]
            url_list = [url for url in url_list if url]  # Keep only non-empty strings

            # Process each valid URL and merge the document chunks
            document_chunks = []
            if url_list:  # Check if there are any valid URLs
                for url in url_list:
                    url_chunks = get_vectorstore_from_url(url)
                    document_chunks.extend(url_chunks)
            else:
                st.warning("No valid website URLs found. Please enter valid URLs separated by commas.")

            with st.sidebar:
                st.write(document_chunks)

            #getting and appending user & Ai input
            user_query = st.chat_input("You're Question")
            if user_query is not None and user_query != "":
                response = get_respons(user_query)
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=response))

            #conversation
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
