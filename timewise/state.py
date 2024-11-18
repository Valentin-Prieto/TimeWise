# pip install -U langchain-community faiss-cpu langchain-huggingface pymupdf tiktoken langchain-ollama python-dotenv

import os
import reflex as rx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from typing import List

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
UPLOAD_FOLDER = "./uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
vector_store:FAISS

class QA(rx.Base):
    """Un par de pregunta y respuesta."""
    question: str
    answer: str

DEFAULT_CHATS = {
    "Chat #1": [],
}
class State(rx.State):
    """Estado de la aplicación."""

    chats: dict[str, list[QA]] = DEFAULT_CHATS
    current_chat = "Chat #1"
    question: str
    processing: bool = False
    new_chat_name: str = ""
    knowledge_base_files: list[str] = []            # Listado con los nombres de los archivos procesados
    upload_status: str = ""
    docs: List = []
    processing_pdf: bool = False

    ###### INGESTA DE DATOS #####
    
    async def save_uploaded_file(self, uploaded_file):
        """Cargar los archivos subidos a la carpeta 'UPLOAD_FOLDER'"""
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        with open(file_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)
        #return file_path
    
    def process_files(self, uploaded_files:list[rx.UploadFile]):
        """Procesar los archivos subidos y cargar los documentos de texto de los PDFs."""
        pdfs = []
        for root, dirs, files in os.walk('uploaded_files'):
            for file in files:
                self.knowledge_base_files.append(file)
                if file.endswith('.pdf'):
                    pdfs.append(os.path.join(root, file))
        self.docs = []
        for pdf in pdfs:
            loader = PyMuPDFLoader(pdf)
            pages = loader.load()

            self.docs.extend(pages)
        return self.docs

    def generate_chunks(self, docs):
        """Dividir documentos largos en fragmentos más pequeños (chunks)."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=256)
        chunks = text_splitter.split_documents(docs)                                            # La cantidad de chunks es cuántos tenemos por cada documento (chunk = parte de la página)
        return chunks
    
    def generate_vector_embedding(self):
        """Generar un vector store (base de datos de embeddings) vacío."""
        global vector_store
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
        single_vector = embeddings.embed_query("this is some text data")
        index = faiss.IndexFlatL2(len(single_vector))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        return vector_store
    
    def create_vector_db(self,chunks):
        """Añadir chunks a la base de datos de vectores (vector store)."""
        global vector_store
        ids = vector_store.add_documents(documents=chunks)
        return ids
    
    def delete_all_files(self):
        """Eliminar todos los archivos subidos y limpiar la lista de nombres."""
        directory = "./uploaded_files"
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"{filename} has been deleted.")
            else:
                print(f"{filename} is not a file and was skipped.")
        self.knowledge_base_files = []


    async def handle_upload(self, uploaded_files:list[rx.UploadFile]):
        """Manejar el proceso completo de carga y procesamiento de archivos."""
        if not uploaded_files:
            self.upload_status = "No hay archivos cargados"
            return
        self.processing_pdf = True
        yield
        self.delete_all_files()
        for file in uploaded_files:
            await self.save_uploaded_file(file)
        self.process_files(uploaded_files)
        chunks = self.generate_chunks(self.docs)
        self.generate_vector_embedding()
        self.create_vector_db(chunks)
        self.upload_status = "¡Se procesaron y agregaron a la base de datos!"
        self.processing_pdf = False

    def delete_file(self, file_name:str):
        """Eliminar un archivo específico y sus embeddings asociados."""
        self.knowledge_base_files = [file for file in self.knowledge_base_files if file!=file_name]
        self.remove_embeddings(file_name)

    def get_database_embeddings(self):
        """Obtener la base de datos de vectores del estado de la aplicación."""
        app_instance = self.get_app_instance()
        database = app_instance.vectordb
        return database
        
    def remove_embeddings(self, file_name):
        """Eliminar los datos del archivo (embeddings) de la base de datos de vectores"""
        database = self.get_database_embeddings()
        database.delete(conditions={"file_name": file_name})

    ##### CHATS #####

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
        """Seleccionar un chat como actual."""
        self.current_chat = chat_name

    def format_docs(self, docs):
        """Para transformar el contexto en cadena de texto"""
        return "\n\n".join([doc.page_content for doc in docs])

    @rx.var
    def chat_titles(self) -> list[str]:
        """Lisado de nombres de los chats."""
        return list(self.chats.keys())

    
    async def process_question(self,form_data: dict[str, str]):
        """Procesar una pregunta, buscar contexto relevante y generar una respuesta."""
        global vector_store

        question = form_data["question"]
        
        if question == "":
            print("no")
            return
        
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)            # Agrego la pregunta/prompt al listado de preguntas
        self.processing = True
        yield
        
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs = {'k': 3, 'fetch_k': 200,'lambda_mult': 1})
    
        model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")
        prompt = """
            Eres un asistente para tareas de preguntas y respuestas. Utiliza los siguientes fragmentos de contexto recuperado para responder la pregunta.
            Responde en viñetas. Asegúrate de que tu respuesta sea relevante para la pregunta y esté basada únicamente en el contexto proporcionado.
            Question: {question} 
            Context: {context} 
            Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt)
        rag_chain = (
            {"context": retriever|self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        output = rag_chain.invoke(question)

        self.chats[self.current_chat][-1].answer += output
        yield
        self.chats = self.chats
        yield
        self.processing = False