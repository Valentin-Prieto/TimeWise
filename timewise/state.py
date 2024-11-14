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
    #db_path: str = tempfile.mkdtemp()
    #pdf_filename: str = ""
    knowledge_base_files: list[str] = []            # Listado con los nombres de los archivos procesados
    upload_status: str = ""
    #vector_store: FAISS
    docs: List = []

    ###### INGESTA DE DATOS #####
    
    async def save_uploaded_file(self, uploaded_file):
        """Cargar los archivos subidos a la carpeta 'UPLOAD_FOLDER'"""
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        with open(file_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)
        #return file_path
    
    def process_files(self, uploaded_files:list[rx.UploadFile]):
        pdfs = []
        for root, dirs, files in os.walk('uploaded_files'):
            # print(root, dirs, files)
            for file in files:
                if file.endswith('.pdf'):
                    pdfs.append(os.path.join(root, file))
        self.docs = []
        for pdf in pdfs:
            loader = PyMuPDFLoader(pdf)
            pages = loader.load()

            self.docs.extend(pages)
        return self.docs

    def generate_chunks(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)                                            # La cantidad de chunks es cuántos tenemos por cada documento (chunk = parte de la página)
        return chunks
    
    def generate_vector_embedding(self):
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
        global vector_store
        ids = vector_store.add_documents(documents=chunks)
        return ids
    
    # def store_db(self, vector_store):
    #     """Guardar la base de datos de los vectores de manera local."""
    #     db_name = "Vectors_Database"
    #     vector_store.save_local(db_name)

    # def load_db(self, db_name, vector_store, embeddings):
    #     new_vector_store = FAISS.load_local(db_name, embeddings=embeddings, allow_dangerous_deserialization=True)
    
    def delete_all_files(self):
        directory = "./uploaded_files"
        # List all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"{filename} has been deleted.")
            else:
                print(f"{filename} is not a file and was skipped.")


    async def handle_upload(self, uploaded_files:list[rx.UploadFile]):
        self.delete_all_files()
        print("ELIMINADOS # # #")
        for file in uploaded_files:
            print("file", file)
            await self.save_uploaded_file(file)
        self.process_files(uploaded_files)
        print("SELF DOCS", self.docs[0])
        chunks = self.generate_chunks(self.docs)
        print("CHUNS", chunks[0])
        self.generate_vector_embedding()
        print("vector embedding creado")
        self.create_vector_db(chunks)
        print("vector db creado")
        self.upload_status = "¡Se procesaron y agregaron a la base de datos!"

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
        """Cambiar el nombre del chat."""
        self.current_chat = chat_name

    def format_docs(self):
        """Para transformar el contexto en cadena de texto"""
        docs = self.docs
        return "\n\n".join([doc.page_content for doc in docs])

    @rx.var
    def chat_titles(self) -> list[str]:
        """Lisado de nombres de los chats."""
        return list(self.chats.keys())
    
    def ollama_process_question(self,question):
        global vector_store
        print(question)
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)            # Agrego la pregunta/prompt al listado de preguntas
        self.processing = True
        yield
        #relevant_data = self.vector_store.search(query=question, search_type='similarity')
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs = {'k': 3, 'fetch_k': 100,'lambda_mult': 1})
        #retriever.invoke(question)
        model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")
        #prompt = hub.pull("rlm/rag-prompt")
        prompt = """
            Eres un asistente para tareas de preguntas y respuestas. Utiliza los siguientes fragmentos de contexto para responde a la pregunta.
            Si no sabes la respuesta, simplemente di que no la sabes.
            Responde en viñetas.
            Asegúrate de que tu respuesta sea relevante para la pregunta y de que sea respondida en base al contexto provisto.
            Manten la respuesta concisa.
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

    
    def process_question(self,form_data: dict[str, str]):
        print(form_data)
        question = form_data["question"]
        print(question)
        if question == "":
            return
        self.ollama_process_question(question)
        

    
    










# import tempfile
# from embedchain import App

# from langchain.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
# from langchain.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from sentence_transformers import SentenceTransformer, util
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain.chains import create_history_aware_retriever
# from langchain_huggingface import HuggingFaceEmbeddings
# from typing import Optional



# conversational_chain = None

# class State(rx.State):
#     """Estado de la aplicación."""

#     chats: dict[str, list[QA]] = DEFAULT_CHATS
#     current_chat = "Chat #1"
#     question: str
#     processing: bool = False
#     new_chat_name: str = ""
#     #db_path: str = tempfile.mkdtemp()
#     #pdf_filename: str = ""
#     knowledge_base_files: list[str] = []            # Listado con los nombres de los archivos procesados
#     upload_status: str = ""
#     #conversational_chain: Optional[RunnableWithMessageHistory] = None

#     ########## ARCHIVOS ##########

#     async def split_documents(self, files: list[rx.UploadFile]):
#         """Carga y procesamiento de archivos."""
#         if not files:
#             self.upload_status = "No hay archivos cargados"
#             return

#         # file = files[0]
#         # upload_data = await file.read()
#         # outfile = rx.get_upload_dir() / file.filename
#         # self.pdf_filename = file.filename

#         # with outfile.open("wb") as file_object:
#         #     file_object.write(upload_data)

#         # app = self.get_app_instance()
#         # app.add(str(outfile), data_type="pdf_file")

#         # self.upload_status = f"¡Se procesó y agregó {self.pdf_filename} a la base de datos!"

#         split_docs = []
#         for pdf in files:
#             upload_data = await pdf.read()
#             with open(pdf.filename, "wb") as f:
#                 f.write(upload_data)
#             loader = PyPDFLoader(pdf.filename)
#             documents = loader.load()
#             splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=256, disallowed_special=(), separators=["\n\n", "\n"," "])
#             split_docs.extend(splitter.split_documents(documents))
#             self.knowledge_base_files.append(pdf.filename)
        
#         return split_docs
    
#     async def upload_to_vectordb(self, split_docs):
#         #print("SPLIT DOCS", split_docs)
#         print("TIPO",type(split_docs))
#         # i=0
#         # while i < 5:
#         #     print(split_docs[i], type(split_docs[i]))
#         #     i+=1
#         # texts = []
#         # for doc in split_docs:
#         #     print("CONTENT", doc.page_content)

#         #     texts.append(doc.page_content)
#         # print("TEXTS", texts)
#         #texts = [doc.page_content if hasattr(doc, 'page_content') else doc for doc in split_docs]
#         #texts = [" ".join(doc) if isinstance(doc, list) else doc for doc in texts]
#         print("SPLIT", split_docs)
#         embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
#         db = FAISS.from_documents(split_docs, embeddings)
#         db.save_local('vectorstore/db_faiss')
        
#         return db

#     def get_conversation_chain(self, retriever):
#         global conversational_chain

#         llm = Ollama(model="llama3.2")
#         contextualize_q_system_prompt = ("Dado el historial del chat y la última pregunta, respondé directamente a la pregunta basándote en los documentos provistos.")
#         contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
#         history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
#         system_prompt = ("Como asistente personal, provea información certera y relevante, basándose en el conocimiento almacenado. No haga preguntas como respuesta. Si no sabe la respuesta según sus documentos, responda que no cuenta con esa información."
#                          "{context}")
#         qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
#         question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#         rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#         store = {}

#         def get_session_history(session_id):
#             if session_id not in store:
#                 store[session_id] = ChatMessageHistory()
#             return store[session_id]
        
#         conversational_chain = RunnableWithMessageHistory( rag_chain, get_session_history, input_messages_key="input", history_msessages_key="chat_history", output_messages_key="answer")

#         #return conversational_rag_chain
    
#     async def handle_upload(self, files:list[rx.UploadFile]):
#         split_documents = await self.split_documents(files)
#         vector_database = await self.upload_to_vectordb(split_documents)
#         retriever = vector_database.as_retriever()
#         self.upload_status = f"¡Se procesaron y agregaron los archivos a la base de datos!"
#         self.get_conversation_chain(retriever)

#     # REVISAR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     def delete_file(self, file_name:str):
#         self.knowledge_base_files = [file for file in self.knowledge_base_files if file!=file_name]
#         self.remove_embeddings(file_name)

#     def get_database_embeddings(self):
#         app_instance = self.get_app_instance()
#         database = app_instance.vectordb
#         return database
    
#     def remove_embeddings(self, file_name):
#         """Eliminar los datos del archivo de la base de datos de vectores"""
#         database = self.get_database_embeddings()
#         database.delete(conditions={"file_name": file_name})


#     ########## CHATS ##########

#     def create_chat(self):
#         """Crear chat nuevo."""
#         self.current_chat = self.new_chat_name
#         self.chats[self.new_chat_name] = []

#     def delete_chat(self):
#         """Eliminar chat."""
#         del self.chats[self.current_chat]
#         if len(self.chats) == 0:
#             self.chats = DEFAULT_CHATS
#         self.current_chat = list(self.chats.keys())[0]

#     def set_chat(self, chat_name: str):
#         """Cambiar el nombre del chat."""
#         self.current_chat = chat_name

#     @rx.var
#     def chat_titles(self) -> list[str]:
#         """Lisado de nombres de los chats."""
#         return list(self.chats.keys())

#     async def process_question(self, form_data: dict[str, str]):
#         """Procesar prompt."""
#         global conversational_chain
#         print("CONVEERSATIONAL CHAIN", conversational_chain)

#         question = form_data["question"]
#         print("QUESTION", question)

#         if question == "":
#             return
        
#         qa = QA(question=question, answer="")
#         self.chats[self.current_chat].append(qa)
#         print("QA", qa)
        
#         self.processing = True
#         yield
        
#         chat_rag = conversational_chain
#         response = chat_rag.invoke({"input": question}, config={"configurable": {"session_id": self.current_chat}})
#         print("RESPONSE", response)
#         context_docs = response.get('context', [])

#         self.chats[self.current_chat][-1].answer += context_docs
#         yield
#         self.chats = self.chats
#         yield
        
#         self.processing = False
        
#         # model = self.ollama_process_question

#         # async for value in model(question):
#         #     yield value

#     # async def ollama_process_question(self, question: str):
#     #     """Respuesta del LLM."""

#     #     qa = QA(question=question, answer="")
#     #     self.chats[self.current_chat].append(qa)            # Agrego la pregunta/prompt al listado de preguntas

#     #     self.processing = True
#     #     yield

#     #     app_instance = self.get_app_instance()              # Instancia de la aplicación.
#     #     prompt = question

#     #     answer_llm = app_instance.chat(prompt)              # Respuesta LLM

#     #     self.chats[self.current_chat][-1].answer += answer_llm
#     #     yield
#     #     self.chats = self.chats
#     #     yield

#     #     self.processing = False

#     # def get_app_instance(self):
#     #     """Configurar la aplicación con Ollama y Chroma."""
#     #     return App.from_config(
#     #         config={
#     #             "llm": {
#     #                 "provider": "ollama",
#     #                 "config": {
#     #                     "model": "llama3.2:1b",
#     #                     "max_tokens": 250,
#     #                     "temperature": 0.5,
#     #                     "stream": True,
#     #                     "base_url": 'http://localhost:11434'
#     #                 }
#     #             },
#     #             "vectordb": {
#     #                 "provider": "chroma",
#     #                 "config": {"dir": self.db_path}
#     #             },
#     #             "embedder": {
#     #                 "provider": "ollama",
#     #                 "config": {
#     #                     "model": "llama3.2:1b",
#     #                     "base_url": 'http://localhost:11434'
#     #                 }
#     #             }
#     #         }
#     #     )
        