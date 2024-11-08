"""The main Chat app."""

import reflex as rx
#from timewise.components import chat, navbar
from timewise.pages import chat, file_upload

# Add state and page to the app.
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="violet",
    ),
)
app.add_page(file_upload.file_upload_page(), route="/")
app.add_page(chat.chat_page(), route="/chat")
