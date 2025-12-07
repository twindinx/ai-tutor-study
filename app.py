import streamlit as st
from groq import Groq

# --- HELPER FUNCTION: STREAM PARSER ---
def parse_groq_stream(stream):
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# --- HELPER FUNCTION: SMART ROUTER (The Magic Logic) ---
def is_new_topic(client, history, new_prompt):
    """
    Uses a small, fast AI call to decide if the user is changing topics.
    Returns: True (Show Form) or False (Skip Form)
    """
    # If no history, it's definitely new
    if not history: 
        return True
    
    # Get the last few messages for context
    context = history[-3:] 
    
    router_prompt = f"""
    Analyze the conversation context and the new user prompt.
    
    [CONTEXT]: {context}
    
    [NEW PROMPT]: "{new_prompt}"
    
    Task: Determine if the [NEW PROMPT] is a continuation/follow-up of the current topic, 
    or if it initiates a NEW line of inquiry (new topic, new concept, or restart).
    
    - "Give me an example", "Explain simpler", "Why?", "What about birds?" (if discussing animals) -> CONTINUATION
    - "What is Speciation?", "Move on to the next section", "Define Mutation" (if discussing something else) -> NEW TOPIC
    
    Respond ONLY with the word "NEW" or "CONTINUATION".
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0,
            max_tokens=10
        )
        decision = completion.choices[0].message.content.strip().upper()
        return "NEW" in decision
    except:
        # If API fails, default to showing the form to be safe
        return True

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Study Partner", page_icon="🧬", layout="centered")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        div[data-testid="stForm"] {border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: #f9f9f9;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔐 Researcher Controls")
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("API Key loaded")
    else:
        api_key = st.text_input("Enter Groq API Key:", type="password")

    condition = st.radio("Condition:", ["Standard GenAI", "Planning-Intervention GenAI"])
    
    if st.button("Reset Chat"):
        st.session_state.messages = []
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
st.title("🧬 Study Partner")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- PLANNING FORM ---
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
                scope_str = ", ".join(scope_selection) if scope_selection else "General Overview"
                sys_instruction = (
                    f"User Question: '{st.session_state.pending_question}'. "
                    f"Constraints -> Scope: {scope_str}. Depth: {depth}. Confusion: {prior}. "
                    "Answer strictly based on this plan."
                )
                
                if api_key:
                    client = Groq(api_key=api_key)
                    # Add instruction to history momentarily for the call
                    msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    msgs.append({"role": "user", "content": sys_instruction})
                    
                    try:
                        stream = client.chat.completions.create(
                            model="llama-3.3-70b-versatile", messages=msgs, stream=True
                        )
                        response = st.write_stream(parse_groq_stream(stream))
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.planning_active = False
                        st.session_state.pending_question = ""
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

# --- INPUT HANDLER ---
if not st.session_state.planning_active:
    if prompt := st.chat_input("Type your question here..."):
        # 1. Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Logic based on Condition
        if condition == "Planning-Intervention GenAI":
            if api_key:
                client = Groq(api_key=api_key)
                # SMART ROUTER CHECK
                # We check if we should interrupt or just answer
                should_plan = is_new_topic(client, st.session_state.messages[:-1], prompt)
                
                if should_plan:
                    st.session_state.pending_question = prompt
                    st.session_state.planning_active = True
                    st.rerun()
                else:
                    # It's a follow-up, just answer normally
                    stream = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        stream=True
                    )
                    response = st.write_stream(parse_groq_stream(stream))
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Add API Key")
        
        else:
            # Standard Mode (Direct)
            if api_key:
                client = Groq(api_key=api_key)
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    stream=True
                )
                response = st.write_stream(parse_groq_stream(stream))
                st.session_state.messages.append({"role": "assistant", "content": response})