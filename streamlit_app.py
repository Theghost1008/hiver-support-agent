import streamlit as st
from src.pipeline import process_message

st.set_page_config(page_title="AppleSupport AI Agent",page_icon="🍎")
st.title("AppleSupport AI Agent - Demo")
st.write("Type a customer support message to see the full pipeline in action")

message = st.text_area("Customer message", placeholder="e.g. my phone screen went blank and won't respond")

if st.button("Run") and message:
    with st.spinner("Processing"):
        result = process_message(message)
    st.subheader("Intent")
    st.write(result["intent"])
    st.subheader("Escalation decision")
    if result["escalation_decision"] == "escalate":
        st.error(f"🚨 Escalate — {result['escalation_reason']}")
    else:
        st.success(f"✅ Auto-handle — {result['escalation_reason']}")
    st.subheader("Generated reply")
    st.info(result["reply"])