!python -m spacy download en_core_web_sm
import streamlit as st
from PIL import Image
import io
import os
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF
import camelot
from pdf2image import convert_from_path
from pytesseract import image_to_string
import re
import spacy

# Load environment variables
os.environ['GENAI_API_KEY'] = 'AIzaSyDJuDtnzp3dEvttkr4oia3pGvmTQuDT-E0'
genai.configure(api_key=os.environ['GENAI_API_KEY'])

# Load environment variables from a .env file
load_dotenv()
genai.configure(api_key=os.getenv('GENAI_API_KEY'))

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

class ResearchPaperAnalyzer:
    def __init__(self, model_name: str):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.image_descriptions = {}
        self.tables_data = {}
        self.extracted_text = ""

    def extract_text(self, pdf_path):
        pages = convert_from_path(pdf_path)
        text = ""
        for page in pages:
            text += image_to_string(page)
        return text

    def segment_text(self, text):
        heading_pattern = r"^\d+\.\s.+"
        paragraphs = text.split("\n\n")
        sections = []
        for para in paragraphs:
            para = para.strip()
            if re.match(heading_pattern, para):
                sections.append({"type": "heading", "content": para})
            else:
                sections.append({"type": "paragraph", "content": para})
        return sections

    def rebuild_structure(self, sections):
        formatted_text = ""
        for section in sections:
            if section['type'] == 'heading':
                formatted_text += f"\n\n### {section['content']} ###\n\n"
            else:
                formatted_text += f"\n{section['content']}\n"
        return formatted_text

    def analyze_text_with_nlp(self, text):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        pos_tags = [(token.text, token.pos_) for token in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return tokens, pos_tags, entities

    def process_text(self, text):
        sections = self.segment_text(text)
        structured_text = self.rebuild_structure(sections)

        analyzed_data = []
        for section in sections:
            tokens, pos_tags, entities = self.analyze_text_with_nlp(section['content'])
            analyzed_data.append({
                "section_type": section['type'],
                "content": section['content'],
                "tokens": tokens,
                "pos_tags": pos_tags,
                "entities": entities
            })

        return structured_text, analyzed_data

    def extract_images(self, pdf_path):
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image['image']
                images.append(image_data)

        return images

    def analyze_image(self, image):
        max_size = (300, 300)
        image.thumbnail(max_size)
        prompt = "Given the following image, what can you infer from it?"
        response = self.model.generate_content([image, prompt])
        return response.text

    def extract_tables(self, pdf_path):
        tables = camelot.read_pdf(pdf_path, pages='all')
        return tables

    def process_paper(self, pdf_path):
        # Step 1: Extract and process text
        self.extracted_text = self.extract_text(pdf_path)
        structured_text, _ = self.process_text(self.extracted_text)

        # Step 2: Extract and analyze images
        images = self.extract_images(pdf_path)
        for idx, img_data in enumerate(images):
            image = Image.open(io.BytesIO(img_data))
            description = self.analyze_image(image)
            self.image_descriptions[f"image_{idx + 1}.jpg"] = description

        # Step 3: Extract and process tables
        tables = self.extract_tables(pdf_path)
        for i, table in enumerate(tables):
            csv_data = table.df.to_csv(index=False)
            self.tables_data[f"table_{i + 1}"] = csv_data

    def handle_query(self, query):
        # Concatenate all information
        combined_info = f"{self.extracted_text}\n\nImage Descriptions:\n{self.image_descriptions}\n\nTables Data:\n{self.tables_data}"

        # Generate a response based on the combined information
        prompt = f"Based on the following information, answer the query:\n\n{combined_info}\n\nQuery: {query}"
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Initialize Streamlit app
st.title("Research Paper Analyzer")

# Create an instance of ResearchPaperAnalyzer
analyzer = ResearchPaperAnalyzer("gemini-1.5-pro-latest")

# Upload PDF file
pdf_file = st.file_uploader("Upload PDF", type="pdf")

# Define functions for processing
def process_text():
    if pdf_file:
        with open(pdf_file.name, "wb") as f:
            f.write(pdf_file.getbuffer())
        analyzer.process_paper(pdf_file.name)
        st.success("Text processed successfully.")

def process_images():
    if pdf_file:
        with open(pdf_file.name, "wb") as f:
            f.write(pdf_file.getbuffer())
        analyzer.process_paper(pdf_file.name)
        st.success("Images processed successfully.")
        st.write("Image Descriptions:")
        for img_name, desc in analyzer.image_descriptions.items():
            st.write(f"{img_name}: {desc}")

def process_tables():
    if pdf_file:
        with open(pdf_file.name, "wb") as f:
            f.write(pdf_file.getbuffer())
        analyzer.process_paper(pdf_file.name)
        st.success("Tables processed successfully.")
        st.write("Tables Data:")
        for table_name, csv_data in analyzer.tables_data.items():
            st.write(f"Table {table_name}:")
            st.text(csv_data)

# Add buttons for processing
if st.button("Process Text"):
    process_text()

if st.button("Process Images"):
    process_images()

if st.button("Process Tables"):
    process_tables()

# Chatbot interaction
st.subheader("Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def add_message(message, is_user=True):
    st.session_state.messages.append({"message": message, "is_user": is_user})

def display_messages():
    for msg in st.session_state.messages:
        if msg['is_user']:
            st.markdown(f"**You:** {msg['message']}")
        else:
            st.markdown(f"**Bot:** {msg['message']}")

# Display existing messages
display_messages()

# Text area input for user query
user_input = st.text_area("You:", "")

if st.button("Send") and user_input.strip():
    # Add user message to chat
    add_message(user_input.strip(), is_user=True)

    # Get bot response
    bot_response = analyzer.handle_query(user_input.strip())
    
    # Add bot response to chat
    add_message(bot_response, is_user=False)

    # Rerun to update the chat with new messages
    st.rerun()
