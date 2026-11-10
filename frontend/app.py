import os
import requests
import streamlit as st
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Stotra Recitation Assistant", layout="centered")

st.title("📿 Stotra Recitation & Verification")
st.caption("Recite Jain/Sanskrit verses to verify pronunciation and detect pauses.")

# Text input for reference stotra
default_verse = "णमो अरिहंताणं णमो सिद्धाणं णमो आयरियाणं णमो उवज्झायाणं णमो लोए सव्वसाहूणं"
reference_text = st.text_area("Reference Verse (Devanagari):", value=default_verse, height=100)

st.write("### Record Recitation")
audio_data = mic_recorder(
    start_prompt="🔴 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key="recorder"
)

# API endpoint (reads from Docker network or falls back to localhost)
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if audio_data is not None:
    audio_bytes = audio_data['bytes']
    st.audio(audio_bytes, format='audio/wav')
    
    if st.button("🔍 Verify Recitation"):
        with st.spinner("Analyzing recitation with Sushrota & VAD..."):
            files = {
                "file": ("recitation.wav", audio_bytes, "audio/wav")
            }
            data = {
                "reference_text": reference_text
            }
            
            try:
                response = requests.post(f"{API_URL}/api/v1/analyze", files=files, data=data)
                if response.status_code == 200:
                    res = response.json()
                    
                    st.success(f"Analysis Completed in {res.get('latency_seconds', 0)}s")
                    st.metric("Similarity Score", f"{res.get('similarity_score', 0)}%")
                    
                    st.subheader("Recognized Speech:")
                    st.info(res.get("transcription", ""))
                    
                    # Pause Detection Feedback
                    st.subheader("Pause Analysis:")
                    pauses = res.get("long_pauses", [])
                    if pauses:
                        for p in pauses:
                            st.warning(f"⚠️ Long pause detected around {p['after_second']}s (duration: {p['pause_duration_seconds']}s)")
                    else:
                        st.write("✅ Rhythm looks good! No unnatural pauses detected.")
                        
                    # Syllable / Discrepancy feedback
                    st.subheader("Pronunciation Discrepancies:")
                    discrepancies = res.get("discrepancies", [])
                    if discrepancies:
                        for d in discrepancies:
                            st.error(f"Expected **'{d['expected']}'**, but heard **'{d['heard']}'**")
                    else:
                        st.write("✅ Accurate pronunciation matching the reference text.")
                else:
                    st.error(f"Server Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
