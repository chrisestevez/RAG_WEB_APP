from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from lib.utils import create_vector_store
import streamlit as st
from streamlit_chat import message
import os
from io import StringIO
base_path = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def vector_store_wrapper(path_n_file:str, llm_model:str):
    """Method is a wrapper to in memory vector store. The \
       cache avoids reloding of the resource after file has\
       been sent to vector db.

    Args:
        path_n_file (str): Path & file name
        llm_model (str): Model used

    Returns:
        VectorStoreRetriever: Vector retriever
    """
    vectorstore = create_vector_store(doc_path=path_n_file, model=llm_model)
    retriever = vectorstore.as_retriever()
    return retriever 

def restart_chat()->None:
    """Methos clears chat after file change.
    """
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['chat_history'] =[]

def get_query():
    """Method to extract chat message.

    Returns:
        str: Message for LLM.
    """
    input_text = st.chat_input("Ask a question about your documents...")
    return input_text

st.title("Retrieval-Augmented Generation(RAG) QA Bot using Langchain & llama")
url = "https://www.linkedin.com/in/christopher-estevez/"
st.write("Created by [Chris Estevez](%s)" % url)
st.header("Ask anything about your documents... 🤖")

llm_model = "llama3.1"
llm = ChatOllama(temperature=0.0, model=llm_model)
upload_doc = st.sidebar.file_uploader("Upload your own document limit 200MB",
                                       type=["txt"],
                                       accept_multiple_files=False,on_change=restart_chat)

if upload_doc is not None:

    string_data = StringIO(upload_doc.getvalue().decode("utf-8")).read()
    file_name = os.path.basename(upload_doc.name).split('.')[0]
    save_file_path = os.path.join(base_path,'docs',f'{file_name}.txt')

    with open(save_file_path, "w",encoding='utf8') as file:
        file.writelines(string_data)
    retriever = vector_store_wrapper(path_n_file=save_file_path,
                                  llm_model=llm_model)

    st.text(f"Loaded {file_name}")

else:
    
    st.markdown('''
    Loaded default article.
                
    [source:](https://www.nbcnews.com/science/hurricane-milton-restoring-power-flooding-death-toll-rcna175147)            

    The chat bot is aware of past questions to improve performance.
    Sample questions from the article:
    
    What happend in tampa?
                
    Tampa Bay was spared from the worst-case scenario of storm surge
    up to 15 feet, but strong winds caused considerable damage and heavy
    rain drove severe urban and inland flooding. Water levels rose as high
    as four feet in some neighborhoods, submerging mailboxes and requiring
    rescues by emergency crews.
                
    Where are water levels expected to raise?

    Water levels are expected to continue rising in the coming days,
    posing a major flood risk in low-lying parts of Pasco County,
    as well as along the Manatee River, where "elevated water
    levels will continue", according to the National Weather Service.
    Additionally, elevated risks of flooding due to high water levels
    were also forecast for the St. Johns and Ocklawaha rivers.    
    
    ''')

    retriever = vector_store_wrapper(path_n_file=os.path.join(base_path,'docs','news.txt'),
                                  llm_model=llm_model)

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is.")

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use five sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
         MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = get_query()
if user_input:
    result = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])
    st.session_state.chat_history.extend(
        [
        HumanMessage(content=user_input),
        AIMessage(content=result['answer']), 
    ]
    )

if st.session_state['generated']:
    for i, item in  enumerate(st.session_state['generated']):
        message(st.session_state['past'][i], is_user=True, key=str(i)+ '_user')
        message(st.session_state['generated'][i], key=str(i))
