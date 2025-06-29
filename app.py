# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jSgHelHA9HaRcxMHxPf54bPy6csOg_ob
"""

# app.py

import streamlit as st
import tempfile
from metadata_backend import extract_text_from_pdf_with_ocr, generate_all_metadata
import json

# App configuration
st.set_page_config(page_title="Document Metadata Generator", layout="centered")
st.title("📄 Document Metadata Generator")

# Upload section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("🔍 Extracting and generating metadata..."):
        # Extract text and page count
        result = extract_text_from_pdf_with_ocr(tmp_path)

        # Generate metadata
        metadata = generate_all_metadata(result["text"], result["page_count"])

    st.success("✅ Metadata generated successfully!")

    # Display metadata
    st.subheader("📌 Metadata Output")
    st.json(metadata)

    # Download button
    st.download_button(
        label="📥 Download Metadata as JSON",
        data=json.dumps(metadata, indent=2),
        file_name="metadata.json",
        mime="application/json"
    )