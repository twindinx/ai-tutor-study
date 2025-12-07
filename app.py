import streamlit as st
from groq import Groq

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Study Partner", page_icon="🧬", layout="centered")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        div[data-testid="stForm"] {border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: #f9f9f9;}
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTION: STREAM PARSER ---
def parse_groq_stream(stream):
    """Parses the raw stream from Groq into clean text."""
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# --- HELPER FUNCTION: CENTRALIZED API CALLER ---
def get_ai_response(client, messages):
    """Handles the API call and streaming display to ensure consistent behavior."""
    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True
        )
        # This is the critical fix applied centrally:
        return st.write_stream(parse_groq_stream(stream))
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# --- HELPER FUNCTION: SMART ROUTER ---
def is_new_topic(client, history, new_prompt):
    """Decides if the user is switching topics."""
    if not history: return True
    context = history[-3:] 
    router_prompt = f"""
    Analyze conversation. Context: {context}. New Prompt: "{new_prompt}".
    Is this a CONTINUATION/FOLLOW-UP (e.g. "Why?", "Example?") or a NEW topic?
    Respond ONLY with "NEW" or "CONTINUATION".
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0, max_tokens=10
        )
        return "NEW" in completion.choices[0].message.content.strip().upper()
    except:
        return True

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔐 Researcher Controls")
    # API Key Loading
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("Key Loaded")
    else:
        api_key = st.text_input("Enter Groq API Key:", type="password")

    condition = st.radio("Condition:", ["Standard GenAI", "Planning-Intervention GenAI"])
    
    if st.button("Reset Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your AI Study Partner. Ask me anything about the reading."}]
        st.session_state.planning_active = False
        st.session_state.pending_question = ""
        st.rerun()

# --- STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your AI Study Partner. Ask me anything about the reading."}]
if "planning_active" not in st.session_state:
    st.session_state.planning_active = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# --- CHAT UI ---
st.title("🧬 Evolution Study Partner")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- PLANNING FORM INTERFACE ---
if st.session_state.planning_active:
    with st.chat_message("assistant"):
        st.markdown(f"### Detected New Topic: *'{st.session_state.pending_question}'*")
        st.info("To give you the best explanation, let's clarify your learning goal.")
        
        with st.form("planning_form"):
            st.markdown("#### 1. Focus Area (Scope)")
            scope_selection = []
            if st.checkbox("Core Definitions"): scope_selection.append("Core Definitions")
            if st.checkbox("The Step-by-Step Process"): scope_selection.append("Step-by-Step Process")
            if st.checkbox("A Concrete Example"): scope_selection.append("Concrete Example")
            if st.checkbox("Common Misconceptions"): scope_selection.append("Common Misconceptions")
            other_text = st.text_input("Other:", placeholder="Type here...", label_visibility="collapsed")
            if other_text: scope_selection.append(f"Other: {other_text}")

            st.markdown("---")
            st.markdown("#### 2. Learning Goal (Depth)")
            depth = st.radio("Select one:", ["Quick Summary", "Detailed Explanation", "Application Practice"], label_visibility="collapsed")
            
            st.markdown("---")
            st.markdown("#### 3. Prior Knowledge")
            prior = st.text_input("Any confusion?", placeholder="e.g., I don't get how...", label_visibility="collapsed")
            
            submitted = st.form_submit_button("Generate Response", use_container_width=True)
            
            if submitted:
                # Build System Prompt for Planning
                scope_str = ", ".join(scope_selection) if scope_selection else "General Overview"
                sys_instruction = (
                    f"User Question: '{st.session_state.pending_question}'. "
                    f"Constraints -> Scope: {scope_str}. Depth: {depth}. Confusion: {prior}. "
                    "Answer strictly based on this plan."
                )
                
                if api_key:
                    client = Groq(api_key=api_key)
                    # Prepare messages
                    msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    msgs.append({"role": "user", "content": sys_instruction})
                    
                    # CALL CENTRALIZED FUNCTION
                    with st.chat_message("assistant"):
                        response = get_ai_response(client, msgs)
                    
                    if response:
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.planning_active = False
                        st.session_state.pending_question = ""
                        st.rerun()
                else:
                    st.error("Please enter API Key.")

# --- MAIN INPUT HANDLER ---
if not st.session_state.planning_active:
    if prompt := st.chat_input("Type your question here..."):
        # 1. Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Logic based on Condition
        if api_key:
            client = Groq(api_key=api_key)
            
            if condition == "Planning-Intervention GenAI":
                # Check for new topic
                if is_new_topic(client, st.session_state.messages[:-1], prompt):
                    st.session_state.pending_question = prompt
                    st.session_state.planning_active = True
                    st.rerun()
                else:
                    # Follow-up (Direct Answer)
                    with st.chat_message("assistant"):
                        response = get_ai_response(client, st.session_state.messages)
                    if response:
                        st.session_state.messages.append({"role": "assistant", "content": response})

            else:
                # Standard Mode (Always Direct Answer)
                with st.chat_message("assistant"):
                    response = get_ai_response(client, st.session_state.messages)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Please enter API Key in sidebar.")