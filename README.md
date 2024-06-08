# Document-Chat-Bot_RAG 📄 💬 🤖
Search functions in document readers are a boon to finding relevant texts. But in large documents, searching through thousands of results is futile. Through the use of LLMs and vector databases, we can speedup this process with better accuracy. This project explores the world of Retrieval Augmented Generation (RAG) models which have made information retrieval from documents a one-shot action.

## Project Description 🔍
The implementation follows the basic principles of a RAG Model:
1. **Chunking** - Splitting the text of a document into smaller parts (chunks).
2. **Embedding** - Each chunk is converted to a numeric vector using an embedding model and is stored in a vector database.
3. **Query** - The user query is also converted into a vector, which is used to search the vector database for similar vectors.
4. **Response** - These similar vectors are then passed through an LLM as a context with the question. The LLM generates a coherent response in natural language.

Technical details:
Programming Language - Python
Integration Framework - LangChain
Embedding Model - OpenAI Text Embedding 3 Small
LLMs - OpenAI GPT-3.5-turbo-0125, Google Gemini Pro
Front End - Streamlit

## Environment Setup 🛠️
This project uses Streamlit to run the file programmed in Python, as a web app. The needed installation packages for the various libraries are in the requirements.txt file. Use the following command to install the requirements ```pip install -r ./requirements.txt```

## How to RUN 🕹️
1. Download the python file & the dotenv template locally in your project directory.
2. Fill in your API's in the dotenv file using a text editor (such as notepad).
3. If using Google Gemini Pro as the LLM, edit the python as per instructions in the comments
4. Open CMD in the project directory and run using the following command :
``` streamlit run rag_chat.py ```

## Credits 🙌
The basic foundations of building GenAI applications using LangChain taught in the course 'LangChain Mastery: Develop LLM Apps with LangChain & Pinecone' by Andrei Dumitrescu aided me in understanding the concepts of RAG models, and using chains to build apps by combining LLMs and vector databases.
