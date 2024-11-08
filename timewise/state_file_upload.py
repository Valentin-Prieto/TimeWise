import reflex as rx
import tempfile
from embedchain import App

class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str

DEFAULT_CHATS = {
    "Chat #1": [],
}

class State(rx.State):
    """The app state."""
    #messages: list[dict] = []
    db_path: str = tempfile.mkdtemp()
    pdf_filename: str = ""
    knowledge_base_files: list[str] = []
    #user_question: str = ""
    upload_status: str = ""

    chats: dict[str, list[QA]] = DEFAULT_CHATS  # Un diccionario con: nombre del chat (clave) y listado de preguntas y respuestas (valor)
    current_chat = "Chat #1"                    # Nombre del chat actual
    question: str                               # Pregunta del chat actual
    processing: bool = False                    # Si se procesa la pregunta o no
    new_chat_name: str = ""                     # Nombre del chat nuevo

    ######### ARCHIVO #########
    def get_app(self):
        return App.from_config(
            config={
                "llm": {"provider": "ollama",
                        "config": {"model": "llama3.2:latest", "max_tokens": 250, "temperature": 0.5, "stream": True,
                                   "base_url": 'http://localhost:11434'}},
                "vectordb": {"provider": "chroma", "config": {"dir": self.db_path}},
                "embedder": {"provider": "ollama",
                             "config": {"model": "llama3.2:latest", "base_url": 'http://localhost:11434'}},
            }
        )

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the file upload and processing."""
        if not files:
            self.upload_status = "¡No hay archivos cargados!"
            return

        file = files[0]
        upload_data = await file.read()
        outfile = rx.get_upload_dir() / file.filename
        self.pdf_filename = file.filename

        # Save the file
        with outfile.open("wb") as file_object:
            file_object.write(upload_data)

        # Process and add to knowledge base
        app = self.get_app()
        app.add(str(outfile), data_type="pdf_file")
        self.knowledge_base_files.append(self.pdf_filename)

        self.upload_status = f"¡Procesado y agregado {self.pdf_filename} a la base de datos!"


    ######### CHAT #########

    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def request(self, form_data: dict[str, str]):
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return
        
        self.process_question()
    
    async def process_question(self, question: str):
        
        app = self.get_app()

        qa = QA(question=question, answer="")           # Agregar la pregunta al listado
        self.chats[self.current_chat].append(qa)        # de preguntas.
        
        self.processing = True                          # Comenzó el procesamiento y limpiar el input.
        yield

        #self.messages.append({"role": "user", "content": self.question})
        response = app.chat(self.question)
        self.chats[self.current_chat].append(response)
        #self.messages.append({"role": "assistant", "content": response})
        #self.user_question = ""  # Clear the question after sending

    # def clear_chat(self):
    #     self.messages = []
        self.processing = False