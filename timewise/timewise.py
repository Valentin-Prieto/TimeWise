"""TIMEWISE"""

import reflex as rx
from timewise.pages import chat

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="violet",
    ),
)
app.add_page(chat.chat_page(), route="/chat")
app.run(host="0.0.0.0", port=3000)