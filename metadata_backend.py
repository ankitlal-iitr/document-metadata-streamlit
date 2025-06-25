
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langdetect import detect
import spacy

# Lightweight spaCy model load
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text_with_ocr(page):
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img)

def extract_text_from_pdf_with_ocr(file_path):
    doc = fitz.open(file_path)
    full_text = ""

    for page in doc:
        text = page.get_text().strip()
        if not text:
            text = extract_text_with_ocr(page)
        full_text += text + "\n\n"

    return {
        "text": full_text.strip(),
        "page_count": len(doc)
    }

def generate_basic_metadata(text, page_count):
    lines = text.split('\n')
    title = next((line.strip() for line in lines if line.strip()), "Unknown Title")
    language = detect(text)
    return {
        "title": title,
        "language": language,
        "page_count": page_count
    }

def generate_summary(text, max_input_len=1024):
    return "Demo summary: summarization is disabled in cloud mode."

def extract_keywords(text, top_n=5):
    # Just return first few unique words as mock keywords
    words = list({w.strip('.,') for w in text.split() if len(w) > 4})
    return words[:top_n]

def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        label = ent.label_
        if label not in entities:
            entities[label] = set()
        entities[label].add(ent.text)
    for label in entities:
        entities[label] = list(entities[label])
    return entities

def classify_document_category(text):
    return "Demo category: classification is disabled in cloud mode."

def generate_all_metadata(text, page_count):
    basic = generate_basic_metadata(text, page_count)
    summary = generate_summary(text)
    keywords = extract_keywords(text)
    entities = extract_entities(text)
    category = classify_document_category(text)

    metadata = {
        "title": basic["title"],
        "language": basic["language"],
        "page_count": basic["page_count"],
        "summary": summary,
        "keywords": keywords,
        "entities": entities,
        "category": category
    }
    return metadata
