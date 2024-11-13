import os
import reflex as rx
import tempfile
from embedchain import App

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
    db_path: str = tempfile.mkdtemp()
    pdf_filename: str = ""
    knowledge_base_files: list[str] = []
    upload_status: str = ""

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Carga y procesamiento de archivos."""
        if not files:
            self.upload_status = "No hay archivos cargados"
            return

        file = files[0]
        upload_data = await file.read()
        outfile = rx.get_upload_dir() / file.filename
        self.pdf_filename = file.filename

        with outfile.open("wb") as file_object:
            file_object.write(upload_data)

        app = self.get_app_instance()
        app.add(str(outfile), data_type="pdf_file")  # Procesar PDF y agregarlo a Chroma
        self.knowledge_base_files.append(self.pdf_filename)

        self.upload_status = f"¡Se procesó y agregó {self.pdf_filename} a la base de datos!"

    def delete_file(self, file_name: str):
        self.knowledge_base_files = [file for file in self.knowledge_base_files if file != file_name]
        self.remove_embeddings(file_name)

    def get_database_embeddings(self):
        app_instance = self.get_app_instance()
        database = app_instance.vectordb
        return database
    
    def remove_embeddings(self, file_name):
        """Eliminar los datos del archivo de la base de datos de vectores"""
        database = self.get_database_embeddings()
        database.delete(conditions={"file_name": file_name})

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
        question = form_data["question"]

        if question == "":
            return

        model = self.ollama_process_question

        async for value in model(question):
            yield value

    async def ollama_process_question(self, question: str):
        """Respuesta del LLM con contexto RAG."""
        
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)  # Agrego la pregunta/prompt al listado de preguntas

        self.processing = True
        yield

        app_instance = self.get_app_instance()  # Instancia de la aplicación.
        database = app_instance.vectordb

        # Realizar consulta en la base de datos de vectores
        relevant_docs = database.query(question)  # Realiza la búsqueda por similitud

        # Si encontramos documentos relevantes, los usamos como contexto
        if relevant_docs:
            context = "\n".join([doc["text"] for doc in relevant_docs])  # Ajusta según el formato de tus documentos
            prompt = f"Contexto relevante:\n{context}\nPregunta: {question}"
        else:
            prompt = question

        # Obtener la respuesta del modelo LLM utilizando el contexto
        answer_llm = app_instance.chat(prompt)

        self.chats[self.current_chat][-1].answer += answer_llm
        yield
        self.chats = self.chats
        yield

        self.processing = False

    def get_app_instance(self):
        """Configurar la aplicación con Ollama y Chroma."""
        return App.from_config(
            config={
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": "llama3.2:1b",
                        "max_tokens": 250,
                        "temperature": 0.5,
                        "stream": True,
                        "base_url": 'http://localhost:11434'
                    }
                },
                "vectordb": {
                    "provider": "chroma",
                    "config": {"dir": self.db_path}  # Ruta de la base de datos de vectores
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": "llama3.2:1b",
                        "base_url": 'http://localhost:11434'
                    }
                }
            }
        )
