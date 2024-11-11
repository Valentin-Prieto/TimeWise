"""TIMEWISE"""

import reflex as rx
from timewise.pages import chat

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="violet",
    ),
)
app.add_page(chat.chat_page(), route="/")