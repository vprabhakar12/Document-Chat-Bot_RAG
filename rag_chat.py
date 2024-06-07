import streamlit as st
from langchain_openai import OpenAIEmbeddings
import pinecone
from pinecone import PodSpec
from langchain_community.vectorstores import Pinecone
import os

# Function to load the document from the user
def load_document(file):
    name, extension = os.path.splitext(file)

    # Various loaders for each document type
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file} ...')
        loader = PyPDFLoader(file)
    elif extension == '.md':
        from langchain.document_loaders import UnstructuredMarkdownLoader
        print(f'Loading {file} ...')
        loader = UnstructuredMarkdownLoader(file)
    else:
        print('Document format not supported')

    data = loader.load()
    return data


# Chunking (Splitting the entire text into parts to embed as vectors)
def chunk_data(data, chunk_size=256, chunk_overlap=50):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Function to create a vector database using the text of the uploaded document
def insert_embeddings(chunks):
    pc = pinecone.Pinecone()

    # Instantiating the embedding model
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Delete the old index, if already present with the same name
    index_name = 'rag_document_chat'
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # Create a new index
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=PodSpec(
            environment='gcp-starter'
        )
    )

    # Converting the chunks into vector and embedding into the vector database in pinecone
    vector_store = Pinecone.from_documents(chunks, embedding_model, index_name=index_name)
    print('Ok')

    return vector_store

# Function to create the chain to invoke the llm by the user question
def create_chain(vector_store, k=3):
    # Uncomment the line below if using Google Gemini as the LLM
    # from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory

    # Uncomment the line below if using Google Gemini as the LLM
    # llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.5, convert_system_message_to_human=True)

    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type='stuff',
        verbose=False
    )

    return crc

# Function to invoke the LLM with the user question
def ask_question(chain, q):
    result = chain.invoke({'question': q})
    return result

# Function to calculate the embedding cost of the document
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens/1000000*0.02

# FRONT END Code
if __name__ == "__main__":
    # Load the dotenv file with the API keys
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.subheader('Document Chatbot 📄 💬 🤖')

    # The sidebar to upload documents
    with st.sidebar:
        # File Upload
        user_doc = st.file_uploader("Upload a document:", type=['pdf', 'md'])
        k = st.number_input('No. of search results to consider (k):', min_value=1, max_value=10, value=5)
        add_doc = st.button('Add Doc')

        if user_doc and add_doc:
            with st.spinner('Reading, chunking and embedding document ...'):
                # Store the user file in the working directory
                bytes_doc = user_doc.read()
                file_name = os.path.join('./', user_doc.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_doc)

                # Load the document and create chunks
                data = load_document(file_name)
                chunks = chunk_data(data)

                # Printing the embedding cost
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Total tokens : {tokens}, Embedding cost : ${embedding_cost:.6f}')

                # Creating an index and storing the vectors in a vector database
                vector_store = insert_embeddings(chunks)

                # Save the vector store for use later
                st.session_state.vs = vector_store
                st.success('Document uploaded, chunked and embedded successfully !')

                # Initialize the chain with the given 'k' value save it for future use
                chain = create_chain(vector_store, k)
                st.session_state.chain = chain

    # Get the user input
    q = st.text_input('Ask anything you want to know in the document :')

    if q:
        if 'vs' in st.session_state:
            # Invoke the chain and print the response
            chain = st.session_state.chain
            chat_reply = ask_question(chain, q)
            st.text_area('Response: ', value=chat_reply['answer'])

            # Chat History with the LLM
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # The current question and response
            answer = chat_reply['answer']
            value = f'USER: {q} \nBOT_: {answer}'

            # Append the chat history for every cycle
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # Setting the widget for the history
            st.text_area(label='Chat History', value=h, key='history', height=400)