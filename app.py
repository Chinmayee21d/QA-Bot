import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Load API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API Key not found. Please set it in the environment variables.")

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks using embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up a conversational chain using a prompt template
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context". Do not provide incorrect information.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle PDF-based user input and generate a response
def pdf_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

# Function to handle general chatbot interaction without PDFs
def general_user_input(user_question):
    model = genai.GenerativeModel('gemini-pro')
    st.session_state.chat_session = model.start_chat(history=[])
    gemini_response = st.session_state.chat_session.send_message(user_question)
    return gemini_response.text

# Main function to render the Streamlit app
def main():
    st.set_page_config(page_title="QA Bot", page_icon="💡")
    st.header("What can I help with ? 🤖")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        pdf_uploaded = st.button("Submit & Process")

    # Initialize chat session if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])

    # If PDFs are uploaded, process them
    if pdf_uploaded:
        if not pdf_docs:
            st.warning("Please upload PDF files before proceeding.")
            return

        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Processing complete. You can now ask questions based on the PDF content.")

    # Main chat interface
    user_question = st.chat_input("Ask a Question...")

    if user_question:
        # Display user's question in the chat
        st.session_state.chat_history.append({"role": "user", "message": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate response based on whether PDFs are uploaded
        if pdf_uploaded and pdf_docs:  # If PDFs were uploaded, use PDF-based chat
            bot_response = pdf_user_input(user_question)
        else:  # Else, use general chatbot interaction
            bot_response = general_user_input(user_question)

        # Display bot's response in the chat
        st.session_state.chat_history.append({"role": "assistant", "message": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

# Run the main function
if __name__ == "__main__":
    main()
