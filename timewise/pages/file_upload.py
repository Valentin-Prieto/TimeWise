import reflex as rx
from timewise.components import chat, navbar
from timewise.state_file_upload import State

color = "violet"

def file_upload_page() -> rx.Component:
    """Page for uploading files to be retrieved by the RAG."""
    return rx.vstack(
        #rx.heading("Sube tu archivo para ser procesado por RAG"),
        #rx.image(src="assets/TIMEWISE CHAT.jpg",width="100px"),
        rx.heading("TIMEWISE", style={"font-size": "40px", "font-weight": "bold"}),
        rx.text("This app allows you to chat with a PDF using Llama 3.2 running locally with Ollama!"),
        rx.hstack(
            rx.vstack(
                rx.heading("PDF Upload", size="md"),
                rx.upload(
                    rx.vstack(
                        rx.button(
                            "Select PDF File",
                            color=color,
                            bg="white",
                            border=f"1px solid {color}",
                        ),
                        rx.text("Drag and drop PDF file here or click to select"),
                    ),
                    id="pdf_upload",
                    multiple=False,
                    accept={".pdf": "application/pdf"},
                    max_files=1,
                    border=f"1px dotted {color}",
                    padding="2em",
                ),
                rx.hstack(rx.foreach(rx.selected_files("pdf_upload"), rx.text)),
                rx.hstack(
                    rx.button(
                        "Upload and Process",
                        on_click=State.handle_upload(rx.upload_files(upload_id="pdf_upload")),
                        align_self="flex-start",
                    ),
                    rx.button(
                        "Clear",
                        on_click=rx.clear_selected_files("pdf_upload"),
                        align_self="flex-start",
                    ),
                    rx.button(
                        "Ir al chat",
                        on_click=lambda: rx.redirect("/chat"),  # To go to the chat
                        align_self="flex-end",
                        margin_left="auto",
                    ),
                    justify="space-between",  # Espaciar los botones en los extremos
                    width="100%",
                    padding="2em",
                ),
                rx.text(State.upload_status),  # Display upload status
                width="50%",
            ),
            #rx.vstack(
                # rx.foreach(
                #     State.messages,
                #     lambda message, index: rx.cond(
                #         message["role"] == "user",
                #         rx.box(
                #             rx.text(message["content"]),
                #             background_color="rgb(0,0,0)",
                #             padding="10px",
                #             border_radius="10px",
                #             margin_y="5px",
                #             width="100%",
                #         ),
                #         rx.box(
                #             rx.text(message["content"]),
                #             background_color="rgb(0,0,0)",
                #             padding="10px",
                #             border_radius="10px",
                #             margin_y="5px",
                #             width="100%",
                #         ),
                #     )
                #),
                # rx.hstack(
                #     rx.input(
                #         placeholder="Ask a question about the PDF",
                #         id="user_question",
                #         value=State.user_question,
                #         on_change=State.set_user_question,
                #         **message_style,
                #     ),
                #     rx.button("Send", on_click=State.chat),
                # ),
            #     rx.button("Clear Chat History", on_click=State.clear_chat),
            #     width="50%",
            #     height="100vh",
            #     overflow="auto",
            # ),
            width="100%",
        ),
        align_items="stretch",
        padding="2em",
    )
        #background_color=rx.color("mauve", 1),
        #color=rx.color("mauve", 12),
        #min_height="100vh",
        #spacing="0",
    #)