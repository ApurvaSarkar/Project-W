import os

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

def main():
    # Set the title and subtitle of the app
    st.title('🦜🔗 Chat With Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')

    url = st.text_input("Insert The website URL")

    prompt = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Query", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        # Load data from the specified URL
        loader = WebBaseLoader(url)
        data = loader.load()

        # Split the loaded data
        text_splitter = CharacterTextSplitter(separator='\n', 
                                        chunk_size=1000, 
                                        chunk_overlap=40)

        docs = text_splitter.split_documents(data)

        # Create Ollama embeddings
        #openai_embeddings = OpenAIEmbeddings()
        ollama_embeddings = OllamaEmbeddings(model="mistral")

        # Create a Chroma vector database from the documents
        vectordb = Chroma.from_documents(documents=docs, 
                                        embedding=ollama_embeddings,
                                        persist_directory=DB_DIR)

        vectordb.persist()

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a mistral llm from Ollama
        llm = Ollama(model="llama2")

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Run the prompt and return the response
        response = qa(prompt)
        st.write(response)
        

if __name__ == '__main__':
    main()