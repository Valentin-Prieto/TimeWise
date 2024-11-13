import os
import reflex as rx
import tempfile
from embedchain import App

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional

class QA(rx.Base):
    """Un par de pregunta y respuesta."""
    question: str
    answer: str

DEFAULT_CHATS = {
    "Chat #1": [],
}

conversational_chain = None

class State(rx.State):
    """Estado de la aplicación."""

    chats: dict[str, list[QA]] = DEFAULT_CHATS
    current_chat = "Chat #1"
    question: str
    processing: bool = False
    new_chat_name: str = ""
    #db_path: str = tempfile.mkdtemp()
    #pdf_filename: str = ""
    knowledge_base_files: list[str] = []            # Listado con los nombres de los archivos procesados
    upload_status: str = ""
    #conversational_chain: Optional[RunnableWithMessageHistory] = None

    ########## ARCHIVOS ##########

    def split_documents(self, files: list[rx.UploadFile]):
        """Carga y procesamiento de archivos."""
        # if not files:
        #     self.upload_status = "No hay archivos cargados"
        #     return

        # file = files[0]
        # upload_data = await file.read()
        # outfile = rx.get_upload_dir() / file.filename
        # self.pdf_filename = file.filename

        # with outfile.open("wb") as file_object:
        #     file_object.write(upload_data)

        # app = self.get_app_instance()
        # app.add(str(outfile), data_type="pdf_file")

        # self.upload_status = f"¡Se procesó y agregó {self.pdf_filename} a la base de datos!"

        split_docs = []
        for pdf in files:
            with open(pdf.name, "wb") as f:
                f.write(pdf.getbuffer())
            loader = PyPDFLoader(pdf.name)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=256, disallowed_special=(), separators=["\n\n", "\n"," "])
            split_docs.extend(splitter.split_documents(documents))
            self.knowledge_base_files.append(pdf.name)
        
        return split_docs
    
    def upload_to_vectordb(self, split_docs):
        embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local('vectorstore/db_faiss')
        
        return db

    def get_conversation_chain(self, retriever):
        global conversational_chain

        llm = Ollama(model="llama3.2")
        contextualize_q_system_prompt = ("Dado el historial del chat y la última pregunta, respondé directamente a la pregunta basándote en los documentos provistos.")
        contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        system_prompt = ("Como asistente personal, provea información certera y relevante, basándose en el conocimiento almacenado. No haga preguntas como respuesta. Si no sabe la respuesta según sus documentos, responda que no cuenta con esa información.")
        qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        store = {}

        def get_session_history(session_id):
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]
        
        conversational_chain = RunnableWithMessageHistory( rag_chain, get_session_history, input_messages_key="input", history_msessages_key="chat_history", output_messages_key="answer")

        #return conversational_rag_chain
    
    def handle_upload(self, files):
        split_documents = self.split_documents(files)
        vector_database = self.upload_to_vectordb(split_documents)
        retriever = vector_database.as_retriever()
        self.upload_status = f"¡Se procesaron y agregaron los archivos a la base de datos!"
        self.get_conversation_chain(retriever)

    # REVISAR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def delete_file(self, file_name:str):
        self.knowledge_base_files = [file for file in self.knowledge_base_files if file!=file_name]
        self.remove_embeddings(file_name)

    def get_database_embeddings(self):
        app_instance = self.get_app_instance()
        database = app_instance.vectordb
        return database
    
    def remove_embeddings(self, file_name):
        """Eliminar los datos del archivo de la base de datos de vectores"""
        database = self.get_database_embeddings()
        database.delete(conditions={"file_name": file_name})


    ########## CHATS ##########

    def create_chat(self):
        """Crear chat nuevo."""
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Eliminar chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Cambiar el nombre del chat."""
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Lisado de nombres de los chats."""
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        """Procesar prompt."""
        global conversational_chain
        question = form_data["question"]
        
        if question == "":
            return
        
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)
        
        self.processing = True
        yield
        
        chat_rag = conversational_chain
        response = chat_rag.invoke({"input": question}, config={"configurable": {"session_id": self.current_chat}})
        context_docs = response.get('context', [])

        self.chats[self.current_chat][-1].answer += context_docs
        yield
        self.chats = self.chats
        yield
        
        self.processing = False
        
        # model = self.ollama_process_question

        # async for value in model(question):
        #     yield value

    # async def ollama_process_question(self, question: str):
    #     """Respuesta del LLM."""

    #     qa = QA(question=question, answer="")
    #     self.chats[self.current_chat].append(qa)            # Agrego la pregunta/prompt al listado de preguntas

    #     self.processing = True
    #     yield

    #     app_instance = self.get_app_instance()              # Instancia de la aplicación.
    #     prompt = question

    #     answer_llm = app_instance.chat(prompt)              # Respuesta LLM

    #     self.chats[self.current_chat][-1].answer += answer_llm
    #     yield
    #     self.chats = self.chats
    #     yield

    #     self.processing = False

    # def get_app_instance(self):
    #     """Configurar la aplicación con Ollama y Chroma."""
    #     return App.from_config(
    #         config={
    #             "llm": {
    #                 "provider": "ollama",
    #                 "config": {
    #                     "model": "llama3.2:1b",
    #                     "max_tokens": 250,
    #                     "temperature": 0.5,
    #                     "stream": True,
    #                     "base_url": 'http://localhost:11434'
    #                 }
    #             },
    #             "vectordb": {
    #                 "provider": "chroma",
    #                 "config": {"dir": self.db_path}
    #             },
    #             "embedder": {
    #                 "provider": "ollama",
    #                 "config": {
    #                     "model": "llama3.2:1b",
    #                     "base_url": 'http://localhost:11434'
    #                 }
    #             }
    #         }
    #     )
        