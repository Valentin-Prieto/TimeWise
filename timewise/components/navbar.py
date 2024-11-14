import reflex as rx
from timewise.state import State

def chats(chat: str) -> rx.Component:
    """Diseño del listado de chats."""
    return  rx.drawer.close(rx.hstack(
        rx.button(
            chat, on_click=lambda: State.set_chat(chat), width="80%", variant="surface"
        ),
        rx.button(
            rx.icon(
                tag="trash",
                on_click=State.delete_chat,
                stroke_width=1,
            ),
            width="20%",
            variant="surface",
            color_scheme="red",
        ),
        width="100%",
    ))


def sidebar_chats(trigger) -> rx.Component:
    """Barra izquierda para ver los chats creados, elegir o borrar uno."""
    return rx.drawer.root(
        rx.drawer.trigger(trigger),
        rx.drawer.overlay(),
        rx.drawer.portal(
            rx.drawer.content(
                rx.vstack(
                    rx.heading("Chats", color=rx.color("mauve", 11)),
                    rx.divider(),
                    rx.foreach(State.chat_titles, lambda chat: chats(chat)),
                    align_items="stretch",
                    width="100%",
                ),
                top="auto",
                right="auto",
                height="100%",
                width="20em",
                padding="2em",
                background_color=rx.color("mauve", 2),
                outline="none",
            )
        ),
        direction="left",
    )


def files(file: str) -> rx.Component:
    """Diseño del listado de archivos."""
    return  rx.drawer.close(rx.hstack(
        rx.button(
            file, width="80%", variant="surface"
        ),
        rx.button(
            rx.icon(
                tag="trash",
                on_click=State.delete_file(file),
                stroke_width=1,
            ),
            width="20%",
            variant="surface",
            color_scheme="red",
        ),
        width="100%",
    ))


def sidebar_files(trigger) -> rx.Component:
    """Barra izquierda para cargar archivos y procesarlos, ver los archivos cargados y con la posibilidad de borrarlos."""
    return rx.drawer.root(
        rx.drawer.trigger(trigger),
        rx.drawer.overlay(),
        rx.drawer.portal(
            rx.drawer.content(
                rx.vstack(
                    rx.heading("Administración de archivos", color=rx.color("mauve", 11)),
                    rx.divider(),
                    rx.upload(
                        rx.vstack(
                            rx.button(
                                "Elegí el archivo",
                                color="#8870b9",
                                bg="white",
                                border=f"1px solid {"#8870b9"}",
                            ),
                            rx.text("Arrastrá un archivo PDF acá o hacé click para seleccionarlo."),
                        ),
                        id="pdf_upload",
                        multiple=False,
                        accept={".pdf": "application/pdf"},
                        max_files=1,
                        border=f"1px dotted {"#8870b9"}",
                        padding="2em",
                    ),
                    rx.foreach(rx.selected_files("pdf_upload"), rx.text),
                    rx.foreach(State.knowledge_base_files, lambda file: files(file)),
                    rx.hstack(
                        rx.button(
                            "Procesar",
                            on_click=State.handle_upload(rx.upload_files(upload_id="pdf_upload")),
                            align_self="flex-start",
                        ),
                        rx.button(
                            "Limpiar",
                            on_click=rx.clear_selected_files("pdf_upload"),
                            align_self="flex-start",
                        ),
                        justify="space-between",
                        width="100%",
                        padding="2em",
                    ),
                    rx.text(State.upload_status),
                    align_items="stretch",
                    width="100%",
                ),
                top="auto",
                right="auto",
                height="100%",
                width="20em",
                padding="2em",
                background_color=rx.color("mauve", 2),
                outline="none",
            ),
        ),
        direction="left",
    )


def modal(trigger) -> rx.Component:
    """PopUp para asginarle el nombre a un chat nuevo."""
    return rx.dialog.root(
        rx.dialog.trigger(trigger),
        rx.dialog.content(
            rx.hstack(
                rx.input(
                    placeholder="Nombre del chat...",
                    on_blur=State.set_new_chat_name,
                    width=["15em", "20em", "30em", "30em", "30em", "30em"],
                ),
                rx.dialog.close(
                    rx.button(
                        "Crear chat",
                        on_click=State.create_chat,
                    ),
                ),
                background_color=rx.color("mauve", 1),
                spacing="2",
                width="100%",
            ),
        ),
    )


def navbar():
    """Diseño de la barra superior."""
    return rx.box(
        rx.hstack(
            rx.hstack(
                #rx.image(src="assets/TimeWise.jpg", width="100px", height="auto"),
                rx.heading("TimeWise"),
                rx.desktop_only(
                    rx.badge(
                    State.current_chat,
                    rx.tooltip(rx.icon("info", size=12), content="Chat actual"),
                    variant="soft"
                    )
                ),
                align_items="center",
            ),
            rx.hstack(
                modal(rx.button("+ Nuevo chat")),
                sidebar_chats(
                    rx.button(
                        rx.icon(
                            tag="messages-square",
                            color=rx.color("mauve", 12),
                        ),
                        background_color=rx.color("mauve", 6),
                    )
                ),
                sidebar_files(
                    rx.button(
                        rx.icon(
                            tag="files",
                            color=rx.color("mauve", 12),
                        ),
                        background_color=rx.color("mauve", 6),
                    )
                ),
                align_items="center",
            ),
            justify_content="space-between",
            align_items="center",
        ),
        backdrop_filter="auto",
        backdrop_blur="lg",
        padding="12px",
        border_bottom=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        position="sticky",
        top="0",
        z_index="100",
        align_items="center",
    )
