import reflex as rx
from timewise.components import chat, navbar

def chat_page() -> rx.Component:
    """Página principal (chat)."""
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