import streamlit as st
from langchain.llms import Ollama
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler

search_internet = st.checkbox("check internet?", value=False, key="internet")
prompt = st.text_input("prompt", value="", key="prompt")

if prompt!="":
    response = ""
    if not search_internet:
        llm = Ollama(model="llama2-uncensored:latest") # 👈 stef default
        response = llm.predict(prompt)
    else:
        llm = Ollama(
            model="llama2-uncensored:latest", 
            callback_manager=CallbackManager([FinalStreamingStdOutCallbackHandler()])
        )
        agent = initialize_agent(
            load_tools(["ddg-search"])
            ,llm 
            ,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
            ,verbose=True
            ,handle_parsing_errors=True
        )
        response = agent.run(prompt, callbacks=[StreamlitCallbackHandler(st.container())])
        # BUG 2023Nov05 can spiral Q&A: https://github.com/langchain-ai/langchain/issues/12892
        # to get out, refresh browser page
        
    st.markdown(response)


#https://medium.aiplanet.com/implementing-rag-using-langchain-ollama-and-chainlit-on-windows-using-wsl-92d14472f15d
#https://medium.com/@stefnestor/python-streamlit-local-llm-2aaa75961d03https://medium.com/@stefnestor/python-streamlit-local-llm-2aaa75961d03
