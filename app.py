import streamlit as st
from groq import Groq

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Study Partner", page_icon="🤖", layout="centered")

# --- CSS FOR CLEAN UI ---
# Hides the Streamlit hamburger menu and footer for a cleaner look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# --- 1. SETUP & STATE ---
# Initialize Session State variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "planning_active" not in st.session_state:
    st.session_state.planning_active = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# --- 2. RESEARCHER SIDEBAR (THE "FLAG") ---
with st.sidebar:
    st.header("🔐 Researcher Controls")
    st.info("Configure this before the participant sits down.")
    
    # API Key Input (Uses Secrets if available, otherwise asks)
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("API Key loaded from Secrets")
    else:
        api_key = st.text_input("Enter Groq API Key:", type="password")

    # THE FEATURE FLAG
    condition = st.radio(
        "Select Condition:",
        ["Standard GenAI", "Planning-Intervention GenAI"],
        help="Standard = Normal Chat. Planning = Triggers Form."
    )
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.planning_active = False
        st.rerun()

# --- 3. MAIN CHAT INTERFACE ---
st.title("💬 AI Study Partner")
st.caption("Ask questions to help you understand the reading material.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. PLANNING INTERVENTION UI ---
# This block ONLY appears if the flag is ON and the user just asked a question
if st.session_state.planning_active:
    with st.chat_message("assistant"):
        st.warning("Wait! To give you the best explanation, let's plan your learning goal.")
        
        with st.form("planning_form"):
            st.write(f"**Your Question:** *{st.session_state.pending_question}*")
            
            st.subheader("1. Focus Area")
            scope = st.multiselect("What specific part do you want to cover?",
                ["Core Definitions", "Step-by-Step Mechanism", "Concrete Examples", "Common Misconceptions"])
            
            st.subheader("2. Learning Goal")
            depth = st.radio("How deep do you want to go?",
                ["Quick Summary", "Deep Explanation", "Quiz Me"])
            
            st.subheader("3. Prior Knowledge")
            prior = st.text_input("Do you have a specific confusion?", placeholder="e.g., I don't get how...")
            
            submitted = st.form_submit_button("Generate Response")
            
            if submitted:
                # CONSTRUCT SYSTEM PROMPT (The "Backend" Logic)
                sys_instruction = (
                    f"The user asked: '{st.session_state.pending_question}'. "
                    f"They planned this constraints: Scope={scope}, Depth={depth}, Confusion={prior}. "
                    "Answer the question strictly following this plan."
                )
                
                # CALL API
                if api_key:
                    try:
                        client = Groq(api_key=api_key)
                        # We send the history + the constrained instruction
                        messages_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                        messages_payload.append({"role": "user", "content": sys_instruction})

                        stream = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=messages_payload,
                            stream=True
                        )
                        response = st.write_stream(stream)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Reset Planning State
                        st.session_state.planning_active = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"API Error: {e}")

# --- 5. CHAT INPUT HANDLER ---
# Only show input if we aren't currently planning
if not st.session_state.planning_active:
    if prompt := st.chat_input("Type your question here..."):
        # 1. Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. CHECK CONDITION FLAG
        if condition == "Planning-Intervention GenAI":
            # Trigger Intervention Mode
            st.session_state.pending_question = prompt
            st.session_state.planning_active = True
            st.rerun()
        
        else:
            # Standard Mode (Direct Response)
            if api_key:
                client = Groq(api_key=api_key)
                try:
                    stream = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        stream=True
                    )
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"API Error: {e}")