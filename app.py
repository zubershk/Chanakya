import streamlit as st
import spacy
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np
import os
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import settings as fe_settings
import google.generativeai as genai # For the LLM Voice
import json
import re
import ast
import random

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------

# --- Model Paths (Local) ---
NER_MODEL_PATH = "chanakya_ner_model"
RL_POLICY_PATH = "chanakya_ppo_policy.zip"

# --- API Payloads ---
JSON_OUTPUT_FORMAT = '{"context_input": "...", "numerical_action": "...", "policy_recommendation": "..."}'
SYSTEM_PROMPT = f"""You are an expert academic in the Arthashastra. Your task is to analyze the contemporary situation (CONTEXT) and the recommended numerical decision (ACTION).
Generate a formal policy document (Niti) grounded in the principles of the Arthashastra.
You MUST ONLY output a valid JSON object. Do not include any explanation, <think> tags, markdown, or any text before or after the JSON object.
Your output must follow this format EXACTLY: {JSON_OUTPUT_FORMAT}.

Use ONLY these options for the 'numerical_action' key:
- "A_1: Resource Reserve Adjustment: HIGH"
- "A_1: Resource Reserve Adjustment: LOW"
- "A_2: Infrastructure Investment Level: HIGH"
- "A_2: Infrastructure Investment Level: MEDIUM"
- "A_2: Infrastructure Investment Level: LOW"
- "A_3: Tax Policy: RAISE"
- "A_3: Tax Policy: STABLE"
- "A_4: Diplomatic Stance Shift: ALLY"
- "A_4: Diplomatic Stance Shift: NEUTRAL"
- "A_4: Diplomatic Stance Shift: ADVERSARIAL"
- "A_5: Unknown Action"
"""

# ----------------------------------------------------------------------
# 2. MODEL LOADING (Cached for performance)
# ----------------------------------------------------------------------

@st.cache_resource
def load_all_models():
    """Loads all models into memory: NER, RL, and API client."""
    print("Loading all models...")
    
    # Load "The Eyes" (NER Model)
    try:
        ner_model = spacy.load(NER_MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load NER model from '{NER_MODEL_PATH}'. Ensure the folder is in your GitHub repo. Error: {e}")
        return None, None, None
        
    # Load "The Brain" (RL Policy)
    try:
        rl_model = PPO.load(RL_POLICY_PATH, device="cpu")
    except Exception as e:
        st.error(f"Failed to load RL model from '{RL_POLICY_PATH}'. Ensure the file is in your GitHub repo. Error: {e}")
        return None, None, None
        
    # Load "The Voice" (Gemini API Client)
    try:
        # Check for Streamlit Secrets
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("GEMINI_API_KEY is not set in Streamlit Secrets. Please add it.")
            raise ValueError("GEMINI_API_KEY is not set in Streamlit Secrets. Please add it.")
            
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
            st.error("GEMINI_API_KEY is set but empty.")
            raise ValueError("GEMINI_API_KEY is set but empty.")
            
        genai.configure(api_key=api_key)
        llm_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            system_instruction=SYSTEM_PROMPT
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini API. Check your secrets. Error: {e}")
        return None, None, None

    print("All models loaded successfully.")
    return ner_model, rl_model, llm_model

# ----------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_real_time_state_features():
    """
    Fetches real-time data and extracts tsfresh features.
    We will *simulate* the GDELT/BigQuery data fetch to stay zero-cost
    and avoid complex service account setup.
    """
    print("Simulating real-time data fetch (GDELT/Commodities)...")
    
    # --- SIMULATED DATA ---
    data = {
        'date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=30, freq='D')),
        'global_avg_tone': np.random.uniform(-5, 5, 30),
        'high_conflict_count': np.random.randint(1000, 3000, 30),
        'total_events': np.random.randint(10000, 50000, 30)
    }
    geopolitical_features_df = pd.DataFrame(data)
    # --- END SIMULATED DATA ---

    # --- TSFRESH FEATURE EXTRACTION ---
    ts_data_long = pd.melt(
        geopolitical_features_df,
        id_vars=['date'],
        value_vars=['global_avg_tone', 'high_conflict_count', 'total_events'],
        var_name='kind',
        value_name='value'
    )
    ts_data_long['id'] = 0 
    
    custom_settings = {
        "global_avg_tone": {"standard_deviation": None},
        "high_conflict_count": {"approximate_entropy": [{"m": 2, "r": 0.5}]},
        "total_events": {"linear_trend": [{"attr": "slope"}]}
    }
    
    # Supress tsfresh warnings
    try:
        extracted_features = extract_features(
            ts_data_long,
            column_id='id',
            column_sort='date',
            column_kind='kind',
            column_value='value',
            kind_to_fc_parameters=custom_settings,
            impute_function=lambda x: x.fillna(0),
            disable_progressbar=True
        )
    except Exception as e:
        print(f"TSFresh feature extraction failed: {e}. Returning empty features.")
        extracted_features = pd.DataFrame() # Return empty
    
    # For this demo, we simulate the S_t vector for the RL agent
    # [Kosa (wealth), Resource_Buffer, Geopolitical_Entropy]
    simulated_S_t = np.array([
        random.uniform(20, 80), # Kosa
        random.uniform(20, 80), # Buffer
        random.uniform(0, 1)    # Entropy
    ], dtype=np.float32)

    return simulated_S_t, extracted_features.to_dict('records')


def call_llm_oracle(llm_model, context, action_label):
    """Calls the Gemini API to generate the Niti (policy)."""
    
    # The SYSTEM_PROMPT is already in the model, so we just send the user prompt.
    user_prompt = f"Analyze this block:\nCONTEXT: {context}\nACTION: {action_label}"
    
    raw_text_json = "" # For error logging

    try:
        # Define generation config for deterministic JSON output
        generation_config = genai.types.GenerationConfig(
            temperature=0.01,
            max_output_tokens=300
        )
        
        # Make the API call
        response = llm_model.generate_content(
            user_prompt,
            generation_config=generation_config
        )
        
        raw_text = response.text

        # Clean up and parse the JSON/Python dict string
        start_index = raw_text.find('{')
        end_index = raw_text.rfind('}')
        
        if start_index != -1 and end_index != -1:
            raw_text_json = raw_text[start_index:end_index+1]
        else:
            st.error(f"LLM Parsing Error: Could not find JSON in model output: {raw_text[:100]}...")
            return None
        
        # Use ast.literal_eval to safely parse a Python dictionary string (with single quotes)
        # Gemini usually returns valid JSON, but this is safer.
        return ast.literal_eval(raw_text_json)

    except Exception as e:
        st.error(f"An unexpected error occurred during Gemini API call: {e}. Raw text was: {raw_text_json}")
        return None

# ----------------------------------------------------------------------
# 4. STREAMLIT UI APPLICATION
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Augmented Strategic Oracle")
st.subheader("*Fusing Chanakyan Wisdom with Real-Time AI*")

# --- Load Models ---
# This runs only once at the start.
ner_model, rl_model, llm_model = load_all_models()

if ner_model and rl_model and llm_model:

    st.sidebar.header("Raw Policy Analysis (NER)")
    text_input = st.sidebar.text_area("Test the 'Eyes' model:", "The king shall protect agriculture from oppressive fines and fill the Kosa.")
    if st.sidebar.button("Analyze Text"):
        doc = ner_model(text_input)
        st.sidebar.subheader("Entities Found (The 'Eyes')")
        if not doc.ents:
            st.sidebar.write("No entities found.")
        for ent in doc.ents:
            st.sidebar.markdown(f"- **{ent.text}** `({ent.label_})`")

    st.divider()

    # --- Main Oracle ---
    st.header("Generate Chanakyan Policy")
    
    if st.button("Query the Oracle"):
        
        with st.spinner("Analyzing current world state (S_t)..."):
            # 1. Get current world state
            S_t, features = get_real_time_state_features()
            
            st.subheader("1. Current State Vector (S_t)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Kosa (Treasury)", f"{S_t[0]:.2f} / 100")
            col2.metric("Resource Buffer", f"{S_t[1]:.2f} / 100")
            col3.metric("Geopolitical Entropy", f"{S_t[2]:.2f} / 1.0")
            
            with st.expander("Show simulated GDELT feature analysis"):
                st.json(features)

        with st.spinner("Deliberating on optimal policy (A_t)..."):
            # 2. Get numerical action from RL "Brain"
            numerical_action, _ = rl_model.predict(S_t, deterministic=True)
            
            ACTION_LABELS = {
                0: "A_3: Tax Policy: RAISE",
                1: "A_3: Tax Policy: STABLE",
                2: "A_2: Infrastructure Investment Level: HIGH",
                3: "A_2: Infrastructure Investment Level: MEDIUM",
                4. "A_2: Infrastructure Investment Level: LOW"
            }
            action_label = ACTION_LABELS.get(int(numerical_action), "A_5: Unknown Action")

            st.subheader("2. Optimal Numerical Action (A_t)")
            st.info(f"**Recommended Action:** {action_label}")

        with st.spinner("Translating policy into Niti... (Calling Gemini API)"):
            # 3. Call LLM "Voice" to get the final policy
            context_summary = f"Current Kosa is {S_t[0]:.2f}, Buffer is {S_t[1]:.2f}, and Geopolitical Entropy is {S_t[2]:.2f}."
            
            # Use the LLM to generate the final output
            generated_policy = call_llm_oracle(llm_model, context_summary, action_label)
            
            if generated_policy:
                st.subheader("3. Final Policy Recommendation (Niti)")
                
                # We use the keys from the JSON *our LLM* generated
                st.markdown(f"**Modern Context:** `{generated_policy.get('context_input', 'N/A')}`")
                st.markdown(f"**Chanakyan Policy:**")
                st.success(f"{generated_policy.get('policy_recommendation', 'No policy generated.')}")
            else:
                st.error("The Oracle LLM failed to generate a response.")

else:
    st.error("Models could not be loaded. Please ensure you are on a CPU runtime and all model files are in the repository.")