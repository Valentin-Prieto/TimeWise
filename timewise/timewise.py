"""The main Chat app."""

import reflex as rx
from timewise.components import chat, navbar

def file_upload_page() -> rx.Component:
    """Page for uploading files to be retrieved by the RAG."""
    return rx.vstack(
        #rx.heading("Sube tu archivo para ser procesado por RAG"),
        rx.image(src="assets/TIMEWISE CHAT.jpg",width="100px"),
        rx.input(type="file"),  # Componente de carga de archivos
        rx.button("Ir al chat", on_click=lambda: rx.redirect("/chat")),  # To go to the chat
        background_color=rx.color("mauve", 1),
        color=rx.color("mauve", 12),
        min_height="100vh",
        align_items="stretch",
        spacing="0",
    )

def chat_page() -> rx.Component:
    """The chat page."""
    return rx.chakra.vstack(
        navbar(),
        chat.chat(),
        chat.action_bar(),
        background_color=rx.color("mauve", 1),
        color=rx.color("mauve", 12),
        min_height="100vh",
        align_items="stretch",
        spacing="0",
    )

# Add state and page to the app.
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="violet",
    ),
    static_dir="assets",
)
app.add_page(file_upload_page, route="/")
app.add_page(chat_page, route="/chat")
