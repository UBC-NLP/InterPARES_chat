import gradio as gr
import pandas as pd
import logging
import asyncio
import os
import time
from uuid import uuid4
from datetime import datetime, timedelta
from pathlib import Path

from modules.vectordb import  get_local_qdrant
from modules.retriever import get_context
from modules.reader import  inf_provider
from modules.utils import make_html_source, parse_output_llm_with_sources, save_logs, get_message_template, get_client_location, get_client_ip, get_platform_info, getconfig
from dotenv import load_dotenv
load_dotenv()
from threading import Lock
from gradio.routes import Request
import json


model_config = getconfig("model_params.cfg")

# create the local logs repo
JSON_DATASET_DIR = Path("json_dataset")
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)
JSON_DATASET_PATH = JSON_DATASET_DIR / f"logs-{uuid4()}.json"





# Metadata filter options
languages = ['All', 'en', 'it', 'de', 'unknown', 'fr', 'es', 'ca', 'nl', 'pt', 'ru',
             'pl', 'hu', 'ko', 'tr', 'vi', 'cs', 'no', 'ur', 'sl', 'id', 'hr', 'ja']

categories = ['All', 'dissemination', 'archiv', 'book', 'policy', 'seminar',
              'report', 'summary', 'research', 'presentation', 'legislation',
              'questions', 'proposal', 'appendix', 'abstract', 'module', 'webinar']

phases = ['All', 'p1', 'p2', 'p3', 'p4']

QUESTIONS = {
    "Sample Questions for any individual report": [
        "What are the benchmark requirements supporting the presumption of authenticity of electronic records in InterPARES 1?",
        "How does InterPARES define the concepts of identity and integrity in assessing authenticity of electronic records?",
        "What procedures are required by InterPARES to maintain the authenticity of electronic records after their transfer to the preserver?",
        "How do the baseline requirements differ from the benchmark requirements in InterPARES authenticity assessment?",
        "According to InterPARES, what evidence supports a presumption of authenticity for electronic records before transfer to the preserver?",
        "According to the InterPARES Appraisal Task Force, at what stage in the records' life cycle should electronic records be appraised?",
        "How does the concept of authenticity influence appraisal decisions in the InterPARES framework?",
        "What are the main activities involved in the InterPARES selection function for authentic electronic records?",
        "How does the Appraisal Task Force define the relationship between appraisal and long-term preservation within InterPARES 1?",
        "What definitions and perspectives of digital preservation emerged from the InterPARES survey on preservation strategies, and how did they differ between archivists and librarians?"
    ],
    "Exploratory Questions": [
        "What are the common archival issues identified across various sectors, entities, or programs?"
    ]
}

# Flatten all questions into a single list
ALL_QUESTIONS = []
for category_questions in QUESTIONS.values():
    ALL_QUESTIONS.extend(category_questions)
                              
new_files = {'InterPARES Phase 1': ['Report 1',
 'Report 2',
 ]
 }

#####--------------- VECTOR STORE -------------------------------------------------


vectorstores = get_local_qdrant()



#####---------------------CHAT-----------------------------------------------------
def start_chat(query,history):
    history = history + [(query,None)]
    history = [tuple(x) for x in history]
    return (gr.update(interactive = False),history)

def finish_chat():
    return (gr.update(interactive = True,value = ""))

def submit_feedback(feedback, logs_data):
    """Handle feedback submission"""
    try:
        if logs_data is None:
            return gr.update(visible=False), gr.update(visible=True)
            
        session_id = logs_data.get("session_id")
        if session_id:
            # Update session last_activity to now
            session_manager.update_session(session_id)
            # Compute duration from the session manager and update the log.
            logs_data["session_duration_seconds"] = session_manager.get_session_duration(session_id)
            
        # Add feedback to logs_data
        logs_data["feedback"] = feedback
        # Now save the (feedback) log record - only locally
        # Function signature: save_logs(json_path, logs, feedback=None)
        save_logs(JSON_DATASET_PATH, logs_data, feedback)
        return gr.update(visible=False), gr.update(visible=True)
    except Exception as e:
        logging.error(f"Error submitting feedback: {e}")
        return gr.update(visible=False), gr.update(visible=True)


# Session Manager added (track session duration, location, and platform)
class SessionManager:
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, client_ip, user_agent):
        session_id = str(uuid4())
        self.sessions[session_id] = {
            'start_time': datetime.now(),
            'last_activity': datetime.now(),
            'client_ip': client_ip,
            'location_info': get_client_location(client_ip),
            'platform_info': get_platform_info(user_agent)
        }
        return session_id
    
    def update_session(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id]['last_activity'] = datetime.now()
    
    def get_session_duration(self, session_id):
        if session_id in self.sessions:
            start = self.sessions[session_id]['start_time']
            last = self.sessions[session_id]['last_activity']
            return (last - start).total_seconds()
        return 0
    
    def get_session_data(self, session_id):
        return self.sessions.get(session_id)

# Initialize session manager
session_manager = SessionManager()
    
async def chat(query,history, language_filter, category_filter, phase_filter, client_ip=None, session_id = None, request:gr.Request = None):
    """taking a query and a message history, use a pipeline (reformulation, retriever, answering) 
       to yield a tuple of:(messages in gradio format/messages in langchain format, source documents)
    """

    if not session_id:
        user_agent =  request.headers.get('User-Agent','') if request else ''
        session_id = session_manager.create_session(client_ip, user_agent)
    else:
        session_manager.update_session(session_id)

    # Get session id
    session_data = session_manager.get_session_data(session_id)
    session_duration = session_manager.get_session_duration(session_id)

    print(f">> NEW QUESTION : {query}")
    print(f"history:{history}")
    print(f"language_filter:{language_filter}")
    print(f"category_filter:{category_filter}")
    print(f"phase_filter:{phase_filter}")
    docs_html = ""
    output_query = ""

    ##------------------------fetch collection from vectorstore------------------------------
    vectorstore = vectorstores["ip"]

    ##------------------------------get context---------------------------------------------- 

    ### adding for assessing computation time
    start_time = time.time()
    
    # Convert 'All' or empty lists to None for no filtering
    # Handle lists from multiselect dropdowns
    lang = None if not language_filter or 'All' in language_filter else language_filter
    cat = None if not category_filter or 'All' in category_filter else category_filter
    ph = None if not phase_filter or 'All' in phase_filter else phase_filter
    doc = None  # Document filter removed
    
    context_retrieved = get_context(
        vectorstore=vectorstore,
        query=query,
        language=lang,
        categories=cat,
        phase=ph,
        filename_org=doc
    )
    end_time = time.time()
    print("Time for retriever:",end_time - start_time)

        
    if not context_retrieved or len(context_retrieved) == 0:
        warning_message = "⚠️ **No relevant information was found in InterPARES documents pertaining your query.** Please try rephrasing your question or selecting different report filters."
        history[-1] = (query, warning_message)
        # Update logs with the warning instead of answer
        logs_data = {
            "record_id": str(uuid4()),
            "session_id": session_id,
            "session_duration_seconds": session_duration,
            "client_location": session_data['location_info'],
            "platform": session_data['platform_info'],
            "question": query,
            "retriever": model_config.get('retriever','MODEL'),
            "endpoint_type": model_config.get('reader','TYPE'),
            "reader": model_config.get('reader','NVIDIA_MODEL'),
            "answer": warning_message,
            "no_results": True  # Flag to indicate no results were found
        }
        yield [tuple(x) for x in history], "", logs_data, session_id
        # Save log for the warning response
        save_logs(JSON_DATASET_PATH, logs_data, None)
        return
    context_retrieved_formatted = "||".join(doc.page_content for doc in context_retrieved)
    context_retrieved_lst = [doc.page_content for doc in context_retrieved]
    
    ##------------------- -------------Define Prompt-------------------------------------------
    SYSTEM_PROMPT = """
        You are InterPARES Q&A, an AI Assistant. \
            You are given a question and extracted passages of the document.\
            Provide a clear and structured answer based on the passages/context provided and the guidelines.
        Guidelines:
        - Passeges are provided as comma separated list of strings
        - If the passages have useful facts or numbers, use them in your answer.
        - When you use information from a passage, mention where it came from by using [Doc i] at the end of the sentence. i stands for the number of the document.
        - Do not use the sentence 'Doc i says ...' to say where information came from.
        - If the same thing is said in more than one document, you can mention all of them like this: [Doc i, Doc j, Doc k]
        - Always use commas to separate document numbers when citing multiple documents.
        - Do not just summarize each passage one by one. Group your summaries to highlight the key parts in the explanation.
        - If it makes sense, use bullet points and lists to make your answers easier to understand.
        - You do not need to use every passage. Only use the ones that help answer the question.
        - If the documents do not have the information needed to answer the question, just say you do not have enough information.
        """
    
    USER_PROMPT = """Passages:
        {context}
        -----------------------
        Question: {question}  - Explained to archivist expert
        Answer in english with the passages citations:
        """.format(context = context_retrieved_lst, question=query)
    
    ##-------------------- apply message template ------------------------------
    messages = get_message_template(model_config.get('reader','TYPE'),SYSTEM_PROMPT,USER_PROMPT)

    ## -----------------Prepare HTML for displaying source documents --------------
    docs_html = []
    for i, d in enumerate(context_retrieved, 1):
        docs_html.append(make_html_source(d, i))
    docs_html = "".join(docs_html)

    ##-----------------------get answer from endpoints------------------------------
    answer_yet = ""

    logs_data = {
        "record_id": str(uuid4()),  # Add unique record ID
        "session_id": session_id,
        "session_duration_seconds": session_duration,
        "client_location": session_data['location_info'],
        "platform": session_data['platform_info'], 
        "system_prompt": SYSTEM_PROMPT, 
        "language_filter": language_filter,
        "category_filter": category_filter,
        "phase_filter": phase_filter,
        "question": query,
        "retriever": model_config.get('retriever','MODEL'),
        "endpoint_type": model_config.get('reader','TYPE'),
        "reader": model_config.get('reader','NVIDIA_MODEL'),
        "docs": [doc.page_content for doc in context_retrieved],
    }


    if model_config.get('reader','TYPE') == 'INF_PROVIDERS':
        chat_model = inf_provider()
        start_time = time.time()
        ai_prefix = "**AI-Generated Response:**\n\n"
        async def process_stream():
            nonlocal answer_yet
            answer_yet += ai_prefix
            # Use LangChain's astream method for vLLM client
            async for chunk in chat_model.astream(messages):
                token = chunk.content
                if token:
                    answer_yet += token
                    parsed_answer = parse_output_llm_with_sources(answer_yet)
                    history[-1] = (query, parsed_answer)
                    logs_data["answer"] = parsed_answer
                    yield [tuple(x) for x in history], docs_html, logs_data, session_id
                    await asyncio.sleep(0.05)

        # Stream the response updates
        async for update in process_stream():
            yield update
            
    else:
        raise ValueError(f"Unsupported reader type: {model_config.get('reader','TYPE')}")
    

    # logging the event
    try:
        save_logs(JSON_DATASET_PATH, logs_data, None)
    except Exception as e:
        logging.error(f"Error saving logs: {e}")
        raise




#####-------------------------- Gradio App--------------------------------------####

# Set up Gradio Theme - Modern Light Theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono")
).set(
    # Main background colors
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_50",
    background_fill_primary="white",
    background_fill_primary_dark="white",
    background_fill_secondary="*neutral_50",
    background_fill_secondary_dark="*neutral_50",
    
    # Block and component backgrounds
    block_background_fill="white",
    block_background_fill_dark="white",
    
    # Input field styling
    input_background_fill="white",
    input_background_fill_dark="white",
    input_background_fill_focus="white",
    input_background_fill_focus_dark="white",
    
    # Button styling
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_dark="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_background_fill_hover_dark="*primary_600",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    
    button_secondary_background_fill="white",
    button_secondary_background_fill_dark="white",
    button_secondary_background_fill_hover="*neutral_100",
    button_secondary_background_fill_hover_dark="*neutral_100",
    button_secondary_text_color="*neutral_800",
    button_secondary_text_color_dark="*neutral_800",
    
    # Panel and container backgrounds
    panel_background_fill="white",
    panel_background_fill_dark="white",
    
    # Text colors
    body_text_color="*neutral_800",
    body_text_color_dark="*neutral_800",
    block_label_text_color="*neutral_800",
    block_label_text_color_dark="*neutral_800",
)

# Modern CSS with extensive light theme styling
css = """
/* Modern color palette */
:root {
    --primary-color: #2563eb;
    --primary-hover: #1d4ed8;
    --secondary-color: #8b5cf6;
    --accent-color: #06b6d4;
    --success-color: #10b981;
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-tertiary: #f1f5f9;
    --border-color: #e2e8f0;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    color-scheme: light !important;
}

* {
    color-scheme: light !important;
}

/* Clean gradient background */
.gradio-container {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #faf5ff 100%) !important;
    color-scheme: light !important;
}

/* Force all components to light theme */
.gradio-container * {
    color-scheme: light !important;
}

/* Override dark mode */
.dark .gradio-container,
[data-theme="dark"] .gradio-container,
.gradio-container[data-theme="dark"] {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #faf5ff 100%) !important;
    color-scheme: light !important;
}

/* Modern card style */
.box-style {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 16px !important;
    border: 1px solid var(--border-color) !important;
    padding: 24px !important;
    box-shadow: var(--shadow-md) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(10px) !important;
}

.box-style:hover {
    box-shadow: var(--shadow-lg) !important;
    border-color: #cbd5e1 !important;
}

/* Modern primary button */
button.primary {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.35) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button.primary:hover {
    box-shadow: 0 8px 20px 0 rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-2px) !important;
}

/* Feedback buttons */
.feedback-button {
    padding: 8px 20px !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--border-color) !important;
}

.feedback-button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Chatbot styling */
#chatbot {
    border-radius: 16px !important;
    box-shadow: var(--shadow-md) !important;
    background: var(--bg-primary) !important;
}

#chatbot .message {
    background: var(--bg-primary) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 8px 0 !important;
}

/* Input textbox */
#input-textbox textarea {
    background: var(--bg-primary) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}

#input-textbox textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* Right panel styling */
#right-panel {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 16px !important;
    border: 1px solid var(--border-color) !important;
    padding: 20px !important;
    box-shadow: var(--shadow-md) !important;
}

/* Tab styling */
.gradio-container .tab-nav button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 12px 20px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.gradio-container .tab-nav button:hover {
    color: var(--primary-color) !important;
    background: var(--bg-secondary) !important;
}

.gradio-container .tab-nav button.selected {
    background: transparent !important;
    color: var(--primary-color) !important;
    border-bottom-color: var(--primary-color) !important;
    font-weight: 600 !important;
}

/* Dropdown styling */
.gradio-container select,
.gradio-container .dropdown {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}

/* Examples styling */
#dropdown-samples {
    background: var(--bg-primary) !important;
    border-radius: 12px !important;
}

/* Sources textbox */
#sources-textbox {
    background: var(--bg-primary) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    border: 1px solid var(--border-color) !important;
}

/* Source document styling */
.source-box a {
    font-size: 0.75rem !important;
    color: var(--primary-color) !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    transition: color 0.2s ease !important;
}

.source-box a:hover {
    color: var(--primary-hover) !important;
    text-decoration: underline !important;
}

/* Example buttons - remove spacing */
#tab-examples .block {
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

#tab-examples .form {
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

#tab-examples > div {
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

#tab-examples .block > * {
    margin: 0 !important;
    padding: 0 !important;
}

#tab-examples button {
    margin: 0 !important;
    margin-bottom: 1px !important;
    border-radius: 8px !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    padding: 10px 14px !important;
    line-height: 1.4 !important;
    width: 100% !important; /* Make button full width */
    display: block !important; /* Ensure it takes up the full line */
}

/* NEW: Full-length sample question buttons (no truncation) */
#examples_list {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}
#tab-examples .example-btn,
#examples_list .example-btn {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    height: auto !important;
    line-height: 1.4 !important;
    text-align: left !important;
    display: block !important;
}
#tab-examples .example-btn * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

/* NEW: Make example question font normal weight/size */
#tab-examples .example-btn,
#examples_list .example-btn,
#tab-examples .example-btn * {
    font-weight: 400 !important;      /* normal (not bold) */
    font-size: inherit !important;    /* same size as other text */
    text-transform: none !important;  /* ensure no automatic casing */
}

footer {
    visibility: hidden;
}
"""

js_code = """
function() {
    // Force light mode on document
    document.documentElement.style.colorScheme = 'light';
    document.body.style.colorScheme = 'light';
    
    // Remove any dark mode classes
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    document.documentElement.classList.add('light');
    document.body.classList.add('light');
    
    // Set data theme attribute
    document.documentElement.setAttribute('data-theme', 'light');
    document.body.setAttribute('data-theme', 'light');
    
    // Force gradio app container to light mode
    const gradioApp = document.querySelector('gradio-app');
    if (gradioApp) {
        gradioApp.style.colorScheme = 'light';
        gradioApp.setAttribute('data-theme', 'light');
        gradioApp.classList.remove('dark');
        gradioApp.classList.add('light');
    }
    
    // Monitor for any theme changes and override them
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && 
                (mutation.attributeName === 'class' || mutation.attributeName === 'data-theme')) {
                const target = mutation.target;
                if (target.classList.contains('dark')) {
                    target.classList.remove('dark');
                    target.classList.add('light');
                }
                if (target.getAttribute('data-theme') === 'dark') {
                    target.setAttribute('data-theme', 'light');
                }
            }
        });
    });
    
    observer.observe(document.documentElement, { attributes: true });
    observer.observe(document.body, { attributes: true });
    if (gradioApp) {
        observer.observe(gradioApp, { attributes: true });
    }
}
"""

init_prompt =  """
Hello, I am InterPARES Q&A, an AI-powered conversational assistant designed to help you understand InterPARES documents. I will answer your questions by using **InterPARES documents**.

"""


with gr.Blocks(title="InterPARES chat Q&A", css=css, theme=theme, elem_id="main-component", js=js_code) as demo:
    # Add file serving for PDFs
    demo.allow_flagging = "never"
    
    # Modern title with gradient
    gr.HTML("""
        <div style="background: linear-gradient(135deg, #2563eb 0%, #8b5cf6 50%, #06b6d4 100%); 
                    padding: 48px 32px; border-radius: 20px; margin-bottom: 32px; 
                    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                    position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                        background: url('data:image/svg+xml,<svg width=&quot;100&quot; height=&quot;100&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot;><rect width=&quot;100&quot; height=&quot;100&quot; fill=&quot;none&quot;/><circle cx=&quot;50&quot; cy=&quot;50&quot; r=&quot;40&quot; fill=&quot;rgba(255,255,255,0.05)&quot;/></svg>'); 
                        opacity: 0.3;"></div>
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 16px; position: relative;">
                <span style="font-size: 3em; margin-right: 16px;">💬</span>
                <h1 style="margin: 0; font-size: 3em; color: white; font-weight: 800; letter-spacing: -1px;">
                    InterPARES-Chat
                </h1>
            </div>
            <div style="text-align: center; color: rgba(255, 255, 255, 0.95); font-size: 1.15em; 
                        font-weight: 500; letter-spacing: 0.3px; position: relative;">
                <span style="background: rgba(255, 255, 255, 0.15); padding: 8px 20px; border-radius: 20px; 
                            backdrop-filter: blur(10px); display: inline-block;">
                    ✨ AI-Powered Conversational Assistant for InterPARES Documents
                </span>
            </div>
        </div>
    """)
    
    #----------------------------------------------------------------------------------------------
    # main layout where chat interaction happens
    # ---------------------------------------------------------------------------------------------
    
    with gr.Row(elem_id="chatbot-row"):
        # LEFT COLUMN: chatbot output, input, examples, and filters
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                value=[(None,init_prompt)],
                show_copy_button=True,
                show_label=False,
                elem_id="chatbot",
                layout="panel",
                avatar_images=(None,"ip_icon.png"),
                type="tuples"  # Explicitly set to avoid deprecation warning
            )

            # feedback UI
            with gr.Column(elem_id="feedback-container"):
                with gr.Row(visible=False) as feedback_row:
                    gr.Markdown("Was this response helpful?")
                    with gr.Row():
                        okay_btn = gr.Button("👍 Okay", elem_classes="feedback-button")
                        not_okay_btn = gr.Button("👎 Not to expectations", elem_classes="feedback-button")
                feedback_thanks = gr.Markdown("Thanks for the feedback!", visible=False)
                feedback_state = gr.State()

            with gr.Row(elem_id = "input-message"):
                gr.Image(
                    value="chat_icon.png",
                    show_label=False,
                    container=False,
                    width=40,
                    height=40,
                    interactive=False,
                    show_download_button=False,
                    show_share_button=False,
                    show_fullscreen_button=False
                )
                textbox=gr.Textbox(
                    placeholder="Ask me anything here!",
                    show_label=False,
                    scale=7,
                    lines=1,
                    interactive=True,
                    elem_id="input-textbox",
                    container=False
                )

            # EXAMPLES AND FILTERS TABS - Below the input on left side
            with gr.Tabs() as left_tabs:
                ############### tab for Question selection ###############
                with gr.TabItem("Examples",elem_id = "tab-examples",id = 0):
                    examples_hidden = gr.Textbox(visible = False)
                    
                    gr.Markdown("### Sample Questions")
                    gr.Markdown("*Click on any question to use it*")
                    
                    # REPLACED: use full-text buttons instead of gr.Examples to avoid truncation
                    example_buttons = []
                    with gr.Column(elem_id="examples_list"):
                        for q in ALL_QUESTIONS:
                            example_buttons.append(
                                gr.Button(q, elem_classes=["example-btn"], variant="secondary")
                            )
                    # Wire buttons to set the hidden textbox (triggers existing .change flow)
                    for btn, q in zip(example_buttons, ALL_QUESTIONS):
                        btn.click(lambda q=q: q, inputs=None, outputs=examples_hidden)

                #---------------- tab for FILTERS ----------------------
                with gr.Tab("Filters",elem_id = "tab-config",id = 2):
                    #---------------- METADATA-BASED FILTERS ------------
                    gr.Markdown("### Filter Documents by Metadata")
                    
                    dropdown_language = gr.Dropdown(
                        languages,
                        label="Language",
                        value=["All"],
                        interactive=True,
                        multiselect=True,
                    )
                    
                    dropdown_category = gr.Dropdown(
                        categories,
                        label="Category",
                        value=["All"],
                        interactive=True,
                        multiselect=True,
                    )
                    
                    dropdown_phase = gr.Dropdown(
                        phases,
                        label="Phase",
                        value=["All"],
                        interactive=True,
                        multiselect=True,
                    )
                    
                    gr.Markdown("*Select 'All' to include all options for that filter*")

        # RIGHT COLUMN: Sources panel
        with gr.Column(scale=1, variant="panel",elem_id = "right-panel"):
            gr.Markdown("### Sources")
            sources_textbox = gr.HTML(show_label=False, elem_id="sources-textbox")
            docs_textbox = gr.State("")

    def change_sample_questions(key):
        # update the questions list based on key selected
        index = list(QUESTIONS.keys()).index(key)
        visible_bools = [False] * len(samples)
        visible_bools[index] = True
        return [gr.update(visible=visible_bools[i]) for i in range(len(samples))]

    # Remove the dropdown_samples change event handler (no longer needed)
    # dropdown_samples.change(change_sample_questions,dropdown_samples,samples)


    #-------------------- Feedback handling -------------------------

    

    def show_feedback(logs):
        """Show feedback buttons and store logs in state"""
        return gr.update(visible=True), gr.update(visible=False), logs 

    def submit_feedback_okay(logs_data):
        """Handle 'okay' feedback submission"""
        return submit_feedback("okay", logs_data)

    def submit_feedback_not_okay(logs_data):
        """Handle 'not okay' feedback submission"""
        return submit_feedback("not_okay", logs_data)

    okay_btn.click(
        submit_feedback_okay,
        [feedback_state],
        [feedback_row, feedback_thanks]
    )
    
    not_okay_btn.click(
        submit_feedback_not_okay,
        [feedback_state],
        [feedback_row, feedback_thanks]
    )

    #-------------------- Session Management + Geolocation -------------------------

    # Add these state components at the top level of the Blocks
    session_id = gr.State(None)
    client_ip = gr.State(None)
    
    @demo.load(api_name="get_client_ip")
    def get_client_ip_handler(dummy_input="", request: gr.Request = None):
        """Handler for getting client IP in Gradio context"""
        return get_client_ip(request)
    

    #-------------------- Gradio voodoo -------------------------
    
    # Update the event handlers - remove document_filter parameter
    (textbox
        .submit(get_client_ip_handler, [textbox], [client_ip], api_name="get_ip_textbox")
        .then(start_chat, [textbox, chatbot], [textbox, chatbot], queue=False, api_name="start_chat_textbox")
        .then(chat, 
            [textbox, chatbot, dropdown_language, dropdown_category, dropdown_phase, client_ip, session_id], 
            [chatbot, sources_textbox, feedback_state, session_id], 
            queue=True, concurrency_limit=8, api_name="chat_textbox")
        .then(show_feedback, [feedback_state], [feedback_row, feedback_thanks, feedback_state], api_name="show_feedback_textbox")
        .then(finish_chat, None, [textbox], api_name="finish_chat_textbox"))

    (examples_hidden
        .change(start_chat, [examples_hidden, chatbot], [textbox, chatbot], queue=False, api_name="start_chat_examples")
        .then(get_client_ip_handler, [examples_hidden], [client_ip], api_name="get_ip_examples")
        .then(chat, 
            [examples_hidden, chatbot, dropdown_language, dropdown_category, dropdown_phase, client_ip, session_id], 
            [chatbot, sources_textbox, feedback_state, session_id], 
            concurrency_limit=8, api_name="chat_examples")
        .then(show_feedback, [feedback_state], [feedback_row, feedback_thanks, feedback_state], api_name="show_feedback_examples")
        .then(finish_chat, None, [textbox], api_name="finish_chat_examples"))

    demo.queue()

# Add routes BEFORE launch - Access FastAPI app through demo's internal app
from fastapi import Response
from fastapi.responses import FileResponse
import mimetypes

@demo.app.get("/download_pdf/{filepath:path}")
async def download_pdf(filepath: str):
    """Serve PDF files for download"""
    try:
        import urllib.parse
        decoded_path = urllib.parse.unquote(filepath)
        base_path = "."
        full_path = os.path.join(base_path, decoded_path)
        
        # Security check: normalize paths to prevent directory traversal
        base_path_abs = os.path.abspath(base_path)
        full_path_abs = os.path.abspath(full_path)
        
        if not full_path_abs.startswith(base_path_abs):
            logging.warning(f"Access denied for path: {full_path}")
            return Response(content="Access denied", status_code=403)
        
        if not os.path.exists(full_path_abs):
            logging.warning(f"File not found: {full_path_abs}")
            return Response(content="File not found", status_code=404)
        
        logging.info(f"Serving PDF: {full_path_abs}")
        
        return FileResponse(
            path=full_path_abs,
            media_type="application/pdf",
            filename=os.path.basename(full_path_abs)
        )
    except Exception as e:
        logging.error(f"Error serving PDF: {e}")
        return Response(content=f"Error serving file: {str(e)}", status_code=500)

# Launch with basic configuration
demo.launch(share=True)