"""
ui.py — Gradio chat interface for the multi-agent legal RAG system.
Run:
    python ui.py
"""

import logging
import os
import inspect
import threading

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "Apikey.env")
DEBUG_MODE = os.environ.get("LEGAL_RAG_DEBUG", "false").strip().lower() == "true"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

import gradio as gr
from agents import SupervisorAgent
import config

print("Gradio version:", gr.__version__)

logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# INITIALISE SYSTEM
# ---------------------------------------------------------------------------

print("Building system ...")
supervisor = SupervisorAgent()
supervisor_lock = threading.RLock()
print("System ready.")


# ---------------------------------------------------------------------------
# CHAT HISTORY HELPERS
# ---------------------------------------------------------------------------

def _session_choices():
    """
    (label, value) pairs for the sidebar list, most recently updated first.
    Label shows the title and turn count, value is the session_id used to
    reload it.
    """
    with supervisor_lock:
        sessions = supervisor.history.list_sessions()
    return [
        (f"{s['title']}  ({s['turn_count']} turns)", s["session_id"])
        for s in sessions
    ]


def _turns_to_chatbot(turns_data):
    """Convert ChatHistoryStore turns into Gradio's messages format."""
    history = []
    for t in turns_data:
        history.append({"role": "user", "content": t["query"]})
        history.append({"role": "assistant", "content": t["answer"]})
    return history


# ---------------------------------------------------------------------------
# CHAT LOGIC
# ---------------------------------------------------------------------------

def bot_logic(user_input, history):

    # Work on a fresh list so a yielded Gradio value is not mutated later.
    history = list(history or [])

    query = user_input.strip()

    if not query:
        yield history
        return

    # Add user message
    history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    # Temporary assistant message
    history.append(
        {
            "role": "assistant",
            "content": "⏳ Thinking...",
        }
    )

    yield history

    # Ask the system. The Supervisor owns mutable session/STM state, so UI
    # operations are serialized even though the selected specialist agents
    # run concurrently inside SupervisorAgent.ask().
    try:
        with supervisor_lock:
            answer = supervisor.ask(query)
    except Exception:
        logger.exception("Unable to answer UI request")
        answer = (
            "A temporary error prevented the request from completing. "
            "Please try again."
        )

    # Replace temporary answer
    history[-1] = {
        "role": "assistant",
        "content": answer,
    }

    yield history


def handle_submit(user_input, history):

    for updated_history in bot_logic(user_input, history):

        stm_text = f"{len(supervisor.stm)} turns in session"
        ltm_text = str(supervisor.ltm)

        yield (
            updated_history,
            "",
            stm_text,
            ltm_text,
            gr.update(choices=_session_choices(), value=supervisor.session_id),
        )


def new_session():
    """'New Session' button: really starts a new chat_history session
    (not just an in-RAM stm reset), so the previous chat stays in the
    sidebar instead of being overwritten."""
    try:
        with supervisor_lock:
            supervisor.new_session()
    except Exception:
        logger.exception("Unable to start a new chat session")
        gr.Warning("Could not start a new chat session.")
        return (
            gr.update(),
            gr.update(),
            f"{len(supervisor.stm)} turns in session",
            str(supervisor.ltm),
            gr.update(),
        )

    return (
        [],                 # chatbot history
        "",                 # textbox
        "0 turns in session",
        str(supervisor.ltm),
        gr.update(choices=_session_choices(), value=supervisor.session_id),
    )


def load_selected_session(session_id):
    """Sidebar click: reload a past session verbatim and make it active,
    so any follow-up question continues that conversation."""
    if not session_id:
        return gr.update(), "", ""

    try:
        with supervisor_lock:
            turns_data = supervisor.load_session(session_id)
    except (ValueError, OSError):
        logger.exception("Unable to load chat session %s", session_id)
        gr.Warning("The selected chat session is unavailable.")
        return (
            gr.update(),
            f"{len(supervisor.stm)} turns in session",
            str(supervisor.ltm),
        )

    return (
        _turns_to_chatbot(turns_data),
        f"{len(supervisor.stm)} turns in session",
        str(supervisor.ltm),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Legal RAG",
) as demo:

    gr.Markdown("# ⚖️ Multi-Agent Legal RAG")

    gr.Markdown(
        "Ask questions about family law and inheritance in "
        "**Italy, Estonia, and Slovenia**."
    )

    with gr.Row():

        # ---------------------------------------------------------------
        # Sidebar
        # ---------------------------------------------------------------

        with gr.Column(scale=1):

            gr.Markdown("### 💬 Chat history")

            session_list = gr.Radio(
                choices=_session_choices(),
                value=supervisor.session_id,
                label=None,
                show_label=False,
                interactive=True,
            )

            new_session_btn = gr.Button("🆕 New chat")

            gr.Markdown("---")
            gr.Markdown("### 🧠 System Status")

            stm_status = gr.Textbox(
                label="Short-term memory",
                value="0 turns in session",
                interactive=False,
            )

            ltm_status = gr.Textbox(
                label="Long-term memory",
                value=str(supervisor.ltm),
                interactive=False,
            )

            gr.Markdown("---")

            gr.Markdown(
                "**Agents available:**\n"
                + "\n".join(
                    f"- {a.agent_id}"
                    for a in config.AGENT_REGISTRY
                )
            )

        # ---------------------------------------------------------------
        # Chat
        # ---------------------------------------------------------------

        with gr.Column(scale=3):

            chatbot_options = {
                "value": [],
                "label": "Conversation",
                "height": 520,
            }
            # Gradio 4/5 needs type="messages" for role/content dictionaries;
            # Gradio 6 only supports messages and removed the type parameter.
            if "type" in inspect.signature(gr.Chatbot).parameters:
                chatbot_options["type"] = "messages"
            chatbot = gr.Chatbot(**chatbot_options)

            msg_input = gr.Textbox(
                label="Your question",
                placeholder="Ask a legal question...",
                lines=2,
            )

            with gr.Row():

                submit_btn = gr.Button(
                    "Send",
                    variant="primary",
                )

                clear_btn = gr.Button(
                    "🧹 New Session",
                )

    # -------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------

    submit_outputs = [
        chatbot,
        msg_input,
        stm_status,
        ltm_status,
        session_list,
    ]

    submit_btn.click(
        fn=handle_submit,
        inputs=[msg_input, chatbot],
        outputs=submit_outputs,
        concurrency_id="supervisor",
        concurrency_limit=1,
    )

    msg_input.submit(
        fn=handle_submit,
        inputs=[msg_input, chatbot],
        outputs=submit_outputs,
        concurrency_id="supervisor",
        concurrency_limit=1,
    )

    clear_btn.click(
        fn=new_session,
        outputs=[
            chatbot,
            msg_input,
            stm_status,
            ltm_status,
            session_list,
        ],
        concurrency_id="supervisor",
        concurrency_limit=1,
    )

    new_session_btn.click(
        fn=new_session,
        outputs=[
            chatbot,
            msg_input,
            stm_status,
            ltm_status,
            session_list,
        ],
        concurrency_id="supervisor",
        concurrency_limit=1,
    )

    session_list.select(
        fn=load_selected_session,
        inputs=[session_list],
        outputs=[
            chatbot,
            stm_status,
            ltm_status,
        ],
        concurrency_id="supervisor",
        concurrency_limit=1,
    )

    demo.queue(default_concurrency_limit=1)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    demo.launch(
        # This UI uses one in-process Supervisor/session and is intentionally
        # local-only. A public deployment needs per-user state and auth.
        share=False,
        debug=DEBUG_MODE,
        theme=gr.themes.Soft(),
    )
