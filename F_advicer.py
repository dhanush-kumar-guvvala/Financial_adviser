import streamlit as st
# Add these imports at the top of your file
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import streamlit as st
import pytesseract
import os
import tempfile
import fitz  # PyMuPDF for PDF processing
import pytesseract
from PIL import Image
import io
import os
import tempfile
import google.generativeai as genai
from PIL import Image
import time
import pandas as pd
from io import StringIO
import json
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Attempt to import PDF libraries with error handling
try:
    import pdf2image
    import pytesseract

    PDF_IMPORTS_AVAILABLE = True
except ImportError:
    PDF_IMPORTS_AVAILABLE = False

# Configure page settings
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    /* Improved CSS for Enhanced UI/UX */
    /* General Page Styling */
    body {
        background-color: #FFD580;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Header Styling */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }

    .sub-header {
        font-size: 1.6rem;
        color: #2563EB;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        text-align: center;
    }

    /* Card Styling */
    .card {
        background-color:#FFD580;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 24px;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
        background-color:##000000;
    }

    /* Button Styling */
    .styled-btn {
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }

    .styled-btn:hover {
        background-color: #1D4ED8;
        transform: scale(1.05);
    }

    /* Chat Interface Styling */
    .chat-user {
        background-color: #E5EDFF;
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 12px;
        border-left: 4px solid #2563EB;
    }

    .chat-ai {
        background-color: #F3F4F6;
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 12px;
        border-left: 4px solid #4B5563;
    }

    /* Sidebar Styling */
    .sidebar {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
        padding: 20px;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }

    .sidebar .stRadio > div {
        flex-direction: column;
    }

    .sidebar .stRadio label {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        background-color: #F3F4F6;
        transition: background-color 0.3s ease;
    }

    .sidebar .stRadio label:hover {
        background-color: #E5EDFF;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.9rem;
        padding: 16px 0;
        background-color: #FFFFFF;
        border-top: 1px solid #E5E7EB;
        margin-top: 24px;
    }

    /* Hover Effects for Buttons and Cards */
    .hover-effect {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .hover-effect:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }

    /* Sequential Button Arrangement */
    .button-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 24px;
    }

    .button-container .styled-btn {
        flex: 1 1 calc(33% - 24px);
        max-width: calc(33% - 24px);
    }

    @media (max-width: 768px) {
        .button-container .styled-btn {
            flex: 1 1 100%;
            max-width: 100%;
        }
    }

    /* Input Field Styling */
    .input-field {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }

    .input-field:focus {
        border-color: #2563EB;
        box-shadow: 0 0 8px rgba(37, 99, 235, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Replace with your actual API key
GOOGLE_API_KEY = "AIzaSyAeNQ69G1M5ZHkZy2h231u_fvdTYgPcuhI"  # Replace this with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'document_analysis' not in st.session_state:
    st.session_state.document_analysis = None
if 'risk_assessment' not in st.session_state:
    st.session_state.risk_assessment = None
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'page' not in st.session_state:
    st.session_state.page = "Document Analysis"
if 'response_editable' not in st.session_state:
    st.session_state.response_editable = {}
if 'chat_input_counter' not in st.session_state:
    st.session_state.chat_input_counter = 0


def speech_to_text(language_code="en-US"):
    """
    Captures speech from microphone and converts to text
    language_code: Language code for speech recognition (e.g., "en-US", "es-ES", "fr-FR")
    """
    language_mapping = {
        "English": "en-US",
        "Spanish": "es-ES",
        "French": "fr-FR",
        "Arabic": "ar-SA",
        "Chinese": "zh-CN",
        "Hindi": "hi-IN",
        "Telugu": "te-IN"
    }

    # Use the mapping if available, otherwise use the provided code
    lang_code = language_mapping.get(language_code, language_code)

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(f"Listening... (Language: {language_code})")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        st.info("Processing speech...")
        text = recognizer.recognize_google(audio, language=lang_code)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return ""
    except sr.RequestError as e:
        st.error(f"Error with speech recognition service: {e}")
        return ""


def text_to_speech(text, language_code="en"):
    """
    Converts text to speech and plays it
    language_code: Language code for text-to-speech (e.g., "en", "es", "fr")
    """
    language_mapping = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "Arabic": "ar",
        "Chinese": "zh-CN",
        "Hindi": "hi",
        "Telugu": "te"
    }

    # Use the mapping if available, otherwise use the provided code
    lang_code = language_mapping.get(language_code, language_code)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(fp.name)

        audio_file = open(fp.name, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')

        # Clean up the temporary file
        audio_file.close()
        os.unlink(fp.name)
    except Exception as e:
        st.error(f"Error generating speech: {e}")


# Helper functions
def check_dependencies():
    """Check if required dependencies are installed and provide detailed guidance if not"""
    missing_deps = []

    if not PDF_IMPORTS_AVAILABLE:
        missing_deps.extend(['pdf2image', 'pytesseract'])

    # Check for poppler specifically
    try:
        # Try to convert a simple PDF to test if poppler is available
        import pdf2image
        test_pdf = pdf2image.convert_from_bytes(
            b"%PDF-1.0\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 3 3]>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n149\n%EOF\n")
        poppler_available = len(test_pdf) > 0
    except Exception:
        poppler_available = False

    if not poppler_available:
        st.error("❌ Poppler is not installed or not in PATH! PDF processing will not work.")

        # Platform-specific installation instructions
        import platform
        system = platform.system().lower()

        if system == "windows":
            st.markdown("""
            ### Windows Poppler Installation:

            1. Download Poppler for Windows from: [https://github.com/oschwartz10612/poppler-windows/releases/](https://github.com/oschwartz10612/poppler-windows/releases/)
            2. Extract the downloaded ZIP file to a folder (e.g., `C:\\Program Files\\poppler`)
            3. Add the `bin` directory to your PATH:
               - Search for "Environment Variables" in Windows search
               - Click "Edit the system environment variables"
               - Click "Environment Variables" button
               - Under "System variables", find "Path" and click "Edit"
               - Click "New" and add the path to the bin folder (e.g., `C:\\Program Files\\poppler\\bin`)
               - Click "OK" on all dialogs
            4. Restart your application or computer
            """)
        elif system == "darwin":  # macOS
            st.markdown("""
            ### macOS Poppler Installation:

            Install using Homebrew:
            ```
            brew install poppler
            ```
            """)
        elif system == "linux":
            st.markdown("""
            ### Linux Poppler Installation:

            For Ubuntu/Debian:
            ```
            sudo apt-get update
            sudo apt-get install -y poppler-utils
            ```

            For Fedora/RHEL/CentOS:
            ```
            sudo dnf install poppler-utils
            ```
            """)
        else:
            st.markdown("""
            ### Poppler Installation:
            Please install poppler-utils for your operating system.
            """)

    if missing_deps:
        st.warning(f"""
        Some Python dependencies are missing: {', '.join(missing_deps)}

        To fix this, run the following command in your terminal:
        ```
        pip install {' '.join(missing_deps)}
        ```

        For pytesseract, you also need to install Tesseract OCR:
        - Windows: https://github.com/UB-Mannheim/tesseract/wiki
        - Mac: brew install tesseract
        - Linux: sudo apt-get install tesseract-ocr
        """)

    return len(missing_deps) == 0 and poppler_available


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using OCR if needed"""
    if not check_dependencies():
        return "Error: Missing dependencies for PDF processing. Please install them using the instructions above."

    try:
        # Create a temporary directory to store the PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = os.path.join(temp_dir, "temp.pdf")

            # Save the uploaded PDF to the temporary path
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())

            # Convert PDF to images with detailed error handling
            try:
                st.info("Attempting to convert PDF to images...")
                images = pdf2image.convert_from_path(
                    temp_pdf_path,
                    dpi=200,
                    fmt="jpeg",
                    output_folder=temp_dir,
                    thread_count=1,
                    grayscale=True,
                    use_cropbox=True,
                    strict=False
                )
                st.success(f"Successfully converted PDF to {len(images)} images.")
            except Exception as pdf_error:
                st.error(f"Error converting PDF to images: {str(pdf_error)}")

                # Provide more diagnostic information
                import shutil
                poppler_path = shutil.which("pdftoppm") or shutil.which("pdftocairo")
                if poppler_path:
                    st.info(f"Poppler binary found at: {poppler_path}")
                else:
                    st.error("No poppler binaries (pdftoppm, pdftocairo) found in PATH!")

                st.warning("""
                Please install the poppler-utils package and make sure it's in your PATH.
                See the installation instructions in the "Check dependencies" section above.
                Alternative solution: Try converting your PDF to images manually, then upload the images instead.
                """)
                return ""

            # Extract text from each image
            text = ""
            for i, img in enumerate(images):
                progress_bar = st.progress((i + 1) / len(images))
                try:
                    page_text = pytesseract.image_to_string(img)
                    text += f"--- Page {i + 1} ---\n{page_text}\n"
                except Exception as ocr_error:
                    st.error(f"Error performing OCR on page {i + 1}: {str(ocr_error)}")

            return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def extract_text_from_image(image_file):
    """Extract text from an image using OCR"""
    try:
        # Convert the uploaded file to a PIL Image
        image = Image.open(image_file)

        # Perform some basic image preprocessing to improve OCR accuracy
        image = image.convert('L')  # Convert to grayscale

        # Optional: Resize image if it's too large (helps with processing speed)
        max_size = 3000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple([int(dim * ratio) for dim in image.size])
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Perform OCR using pytesseract
        with st.spinner("Processing image with OCR..."):
            try:
                text = pytesseract.image_to_string(image)
                text = text.strip()

                # Remove any non-standard characters or encoding issues
                text = ''.join(char for char in text if ord(char) < 128)

                if not text:
                    st.warning(
                        "No text was detected in the image. Please ensure the image contains clear, readable text.")
                    return ""

                return text
            except pytesseract.TesseractError as e:
                st.error(f"OCR Error: {str(e)}")
                st.info("Please ensure Tesseract OCR is properly installed on your system.")
                return ""
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return ""


def format_content_for_notepad(content):
    """Clean text content to be compatible with basic text editors like Notepad"""
    # Replace special/smart quotes with standard ones
    content = content.replace('"', '"').replace('"', '"')
    content = content.replace(''', "'").replace(''', "'")

    # Replace em/en dashes with hyphens
    content = content.replace('—', '-').replace('–', '-')

    # Replace other special characters
    content = content.replace('•', '*')
    content = content.replace('…', '...')

    # Remove any zero-width spaces or other invisible unicode
    content = re.sub(r'[\u200B-\u200D\uFEFF]', '', content)

    return content


def analyze_document(text, query_type="general_analysis"):
    """Analyze document with Gemini AI based on query type"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')  # Updated model name

        if query_type == "general_analysis":
            prompt = f"""
            You are an expert legal AI assistant. Analyze the following legal document:

            {text}

            Provide a comprehensive analysis including:
            1. Document type and purpose
            2. Key parties involved
            3. Main clauses and their implications
            4. Summary of rights and obligations
            5. Important dates and deadlines

            Format your response in a well-structured manner with markdown headings.
            Use simple characters that are supported by normal text editors like notepad.
            Avoid special characters, smart quotes, em-dashes, or any other characters that might not display correctly in basic text editors.
            """

        elif query_type == "risk_assessment":
            prompt = f"""
            You are an expert legal AI assistant. Conduct a thorough risk assessment of the following legal document:

            {text}

            Provide:
            1. Identification of high-risk clauses (with clause number/reference)
            2. Missing important clauses or protections
            3. Ambiguous or vague language that could create legal uncertainties
            4. Compliance issues with common regulations
            5. Overall risk score (1-10, where 10 is highest risk)
            6. Specific recommendations to mitigate identified risks

            Format your response in a well-structured manner with markdown headings.
            Use simple characters that are supported by normal text editors like notepad.
            Avoid special characters, smart quotes, em-dashes, or any other characters that might not display correctly in basic text editors.
            """

        response = model.generate_content(prompt)
        return format_content_for_notepad(response.text)

    except Exception as e:
        st.error(f"Error analyzing document: {str(e)}")
        return "Unable to analyze the document. Please try again."


def generate_document(doc_type, parameters):
    """Generate a legal document based on parameters"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')  # Updated model name

        prompt = f"""
        You are an expert legal document generator. Create a professional {doc_type} with the following parameters:

        {parameters}

        The document should:
        1. Follow standard legal formatting and structure
        2. Include all necessary clauses for this type of agreement
        3. Be compliant with common legal requirements
        4. Use clear, precise legal language
        5. Include proper signature blocks and date fields

        Format your response in a well-structured manner with markdown headings.
        Use simple characters that are supported by normal text editors like notepad.
        Avoid special characters, smart quotes, em-dashes, or any other characters that might not display correctly in basic text editors.
        """

        response = model.generate_content(prompt)
        return format_content_for_notepad(response.text)

    except Exception as e:
        st.error(f"Error generating document: {str(e)}")
        return "Unable to generate the document. Please try again."


def get_chatbot_response(query, language="English", context=""):
    """Get response from legal chatbot"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')  # Updated model name

        chat_history = "\n".join([f"User: {q}\nAI: {a}" for q, a in st.session_state.chat_history[-5:]])

        prompt = f"""
        You are an expert legal AI assistant specialized in contract law, compliance, and regulations.

        Previous conversation:
        {chat_history}

        Context: {context}

        User query: {query}

        Provide a helpful, accurate response about the legal query. If you're unsure, state clearly what you don't know.
        If the question is about specific jurisdiction laws that you're not confident about, indicate that limitation.

        Use simple characters that are supported by normal text editors like notepad.
        Avoid special characters, smart quotes, em-dashes, or any other characters that might not display correctly in basic text editors.

        Provide your response in {language}.
        """

        response = model.generate_content(prompt)
        return format_content_for_notepad(response.text)

    except Exception as e:
        st.error(f"Error getting chatbot response: {str(e)}")
        return f"Sorry, I encountered an error processing your query. Please try again."


def search_legal_precedents(query):
    """Search for legal precedents based on query"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')  # Updated model name

        prompt = f"""
        You are an expert legal research assistant. The user is looking for legal precedents and case law related to:

        "{query}"

        Provide:
        1. Most relevant case names and citations (at least 3-5 if available)
        2. Brief summary of each case's ruling and significance
        3. How these precedents might apply to similar situations
        4. Any conflicting rulings or jurisdictional differences

        Format your response in a well-structured manner with markdown.
        Use simple characters that are supported by normal text editors like notepad.
        Avoid special characters, smart quotes, em-dashes, or any other characters that might not display correctly in basic text editors.
        """

        response = model.generate_content(prompt)
        return format_content_for_notepad(response.text)

    except Exception as e:
        st.error(f"Error searching legal precedents: {str(e)}")
        return "Unable to search for legal precedents. Please try again."


def convert_markdown_to_pdf(markdown_text, title="Document"):
    """Convert markdown text to PDF"""
    try:
        # Create a buffer for the PDF
        buffer = io.BytesIO()

        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        # Create a custom style for the text
        custom_style = ParagraphStyle(
            'CustomStyle',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
        )

        # Create a list to hold the PDF elements
        elements = []

        # Add title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=1,  # Center alignment
            spaceAfter=12
        )
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 12))

        # Convert markdown to paragraphs
        # This is a simple conversion - for more complex markdown, you'd need a proper parser
        lines = markdown_text.split('\n')
        current_paragraph = ""

        for line in lines:
            line = line.strip()

            # Handle headings
            if line.startswith('# '):
                if current_paragraph:
                    elements.append(Paragraph(current_paragraph, custom_style))
                    elements.append(Spacer(1, 6))
                    current_paragraph = ""

                heading_text = line[2:]
                heading_style = ParagraphStyle(
                    'Heading1',
                    parent=styles['Heading1'],
                    fontSize=14,
                    spaceAfter=10,
                    spaceBefore=10
                )
                elements.append(Paragraph(heading_text, heading_style))

            elif line.startswith('## '):
                if current_paragraph:
                    elements.append(Paragraph(current_paragraph, custom_style))
                    elements.append(Spacer(1, 6))
                    current_paragraph = ""

                heading_text = line[3:]
                heading_style = ParagraphStyle(
                    'Heading2',
                    parent=styles['Heading2'],
                    fontSize=12,
                    spaceAfter=8,
                    spaceBefore=8
                )
                elements.append(Paragraph(heading_text, heading_style))

            # Handle blank lines as paragraph breaks
            elif not line:
                if current_paragraph:
                    elements.append(Paragraph(current_paragraph, custom_style))
                    elements.append(Spacer(1, 6))
                    current_paragraph = ""

            # Add line to current paragraph
            else:
                if current_paragraph:
                    current_paragraph += " " + line
                else:
                    current_paragraph = line

        # Add any remaining paragraph
        if current_paragraph:
            elements.append(Paragraph(current_paragraph, custom_style))

        # Build the PDF
        doc.build(elements)

        # Get the PDF from the buffer
        buffer.seek(0)
        return buffer

    except Exception as e:
        st.error(f"Error converting to PDF: {str(e)}")
        return None


def get_image_download_link(img, filename="document.png", text="Download as Image"):
    """Generate a download link for a PIL Image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def render_text_as_image(text, title="Document", width=800, height=1000):
    """Render text as an image for download"""
    try:
        # Create a new image with white background
        img = Image.new('RGB', (width, height), color='white')
        from PIL import ImageDraw, ImageFont

        # Create a draw object
        draw = ImageDraw.Draw(img)

        # Try to use a common font or default to the built-in font
        try:
            # Try to get a TrueType font
            title_font = ImageFont.truetype("Arial.ttf", 24)
            body_font = ImageFont.truetype("Arial.ttf", 14)
        except IOError:
            # If not available, use the default font
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()

        # Draw the title
        draw.text((20, 20), title, fill='black', font=title_font)

        # Draw the text
        y_position = 60
        lines = text.split('\n')

        for line in lines:
            # Check if we need to add a page (not implemented here)
            if y_position > height - 40:
                break

            # Add the line
            draw.text((20, y_position), line, fill='black', font=body_font)
            y_position += 20

        return img

    except Exception as e:
        st.error(f"Error rendering text as image: {str(e)}")
        return None


# Sidebar with business-style template
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h1 style="color: #1E3A8A; font-size: 1.8rem; margin-bottom: 0;">⚖️ AI Legal Assistant</h1>
    <p style="color: #6B7280; font-size: 0.9rem;">Your Intelligent Legal Partner</p>
</div>
""", unsafe_allow_html=True)

# Add a professional logo or placeholder
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQRa9vqZLs4_b-a0gR6iXNs69vQBdNL9q3lDA&s",
                 width=100)
st.sidebar.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

# Language selection with improved UI
st.sidebar.markdown("""
<p style="color: #4B5563; font-weight: 500; margin-bottom: 5px;">Language Settings</p>
""", unsafe_allow_html=True)
languages = ["English", "Spanish", "French", "Arabic", "Chinese", "Hindi", "telugu"]
selected_language = st.sidebar.selectbox(
    "Select Language",
    languages,
    index=languages.index(st.session_state.language)
)
st.session_state.language = selected_language

# Navigation with improved styles
st.sidebar.markdown("""
<p style="color: #4B5563; font-weight: 500; margin-bottom: 5px; margin-top: 20px;">Navigate Services</p>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["Document Analysis", "Legal Chatbot", "Risk Assessment", "Document Generator", "Legal Research"],
    index=["Document Analysis", "Legal Chatbot", "Risk Assessment", "Document Generator", "Legal Research"].index(
        st.session_state.page)
)

# Update the current page in session state
st.session_state.page = page

st.sidebar.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
st.sidebar.info(
    "This AI Legal Assistant helps with contract analysis, compliance guidance, risk assessment, and document generation."
)


def display_chatbot():
    """Display a collapsible chatbot widget on every page."""
    with st.sidebar.expander("💬 Legal Chatbot", expanded=False):
        st.markdown("""
        <div class="card">
            <p>Ask any legal or compliance-related questions. The AI assistant can help with:
            <ul>
                <li>Contract interpretation</li>
                <li>Regulatory compliance</li>
                <li>Business legalities</li>
                <li>General legal guidance</li>
            </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize states for voice recording
        if "is_recording_chat" not in st.session_state:
            st.session_state.is_recording_chat = False
        if "voice_chat_query" not in st.session_state:
            st.session_state.voice_chat_query = ""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        chat_history_text = "\n".join([f"You: {q}\nAI: {a}\n" for q, a in st.session_state.chat_history])
        st.text_area("Chat History", chat_history_text, height=300, disabled=True)

        # Voice input controls
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎤 Start Recording", key="start_chat_voice"):
                st.session_state.is_recording_chat = True
                with st.spinner("Listening..."):
                    st.session_state.voice_chat_query = speech_to_text(st.session_state.language)
                    if st.session_state.voice_chat_query:
                        st.success(f"Recognized: {st.session_state.voice_chat_query}")

        with col2:
            if st.session_state.is_recording_chat and st.button("⏹️ Stop Recording", key="stop_chat_voice"):
                st.session_state.is_recording_chat = False

        with col3:
            if st.session_state.chat_history:
                if st.button("🔊 Read Last Response", key="read_chat_response"):
                    last_response = st.session_state.chat_history[-1][1]
                    text_to_speech(last_response, st.session_state.language)

        # Chat input field with voice recognition text
        user_query = st.text_input(
            "Ask a legal question:",
            value=st.session_state.voice_chat_query,
            key=f"chat_input_{st.session_state.chat_input_counter}"
        )
        st.session_state.chat_input_counter += 1

        # Submit button
        if st.button("Send", key="send_chat_button"):
            if user_query.strip():
                with st.spinner("Getting response..."):
                    response = get_chatbot_response(user_query, st.session_state.language)
                    st.session_state.chat_history.append((user_query, response))
                    st.session_state.voice_chat_query = ""  # Clear voice input after sending
                st.rerun()

        # Clear chat history button
        if st.button("Clear Chat History", key="clear_chat_history"):
            st.session_state.chat_history = []
            st.success("Chat history cleared successfully!")
            st.rerun()


# Main content area with improved UI for each page
if page == "Document Analysis":
    st.markdown('<h1 class="main-header">📄 Contract & Document Analysis</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p>Upload a legal document (contract, policy, agreement) for AI-powered analysis.
        The system will extract text, identify key clauses, and provide a comprehensive summary.</p>
    </div>
    """, unsafe_allow_html=True)

    # Check for dependencies first
    check_dependencies()

    uploaded_file = st.file_uploader("Upload a document (PDF, Image, or Text file)",
                                     type=["pdf", "png", "jpg", "jpeg", "txt"])

    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file is not None:
            st.markdown('<h3 class="sub-header">Document Processing</h3>', unsafe_allow_html=True)

            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                with st.spinner("Extracting text from PDF..."):
                    st.info("Processing PDF document. This may take a moment...")
                    text = extract_text_from_pdf(uploaded_file)
                    if text:
                        st.success("PDF text extraction complete!")
                    else:
                        st.error("PDF extraction failed. Check if dependencies are installed correctly.")

            elif uploaded_file.type.startswith("image/"):
                with st.spinner("Extracting text from image..."):
                    st.info("Processing image with OCR. This may take a moment...")
                    text = extract_text_from_image(uploaded_file)
                    if text:
                        st.success("Image text extraction complete!")
                    else:
                        st.error("Image extraction failed. Check if dependencies are installed correctly.")

            else:  # Assume text file
                text = uploaded_file.getvalue().decode("utf-8")
                st.success("Text file loaded successfully!")

            st.session_state.document_text = text

            # Display extracted text with option to edit
            st.markdown('<h3 class="sub-header">Extracted Text</h3>', unsafe_allow_html=True)
            text_area = st.text_area(
                "Review and edit the extracted text if needed",
                st.session_state.document_text,
                height=400
            )
            st.session_state.document_text = text_area

    with col2:
        if st.session_state.document_text:
            st.markdown('<h3 class="sub-header">Analysis Controls</h3>', unsafe_allow_html=True)

            analyze_button = st.button("🔍 Analyze Document", use_container_width=True)

            if analyze_button:
                with st.spinner("Analyzing document with AI..."):
                    analysis = analyze_document(st.session_state.document_text)
                    st.session_state.document_analysis = analysis
                    # Initialize editable response
                    st.session_state.response_editable["document_analysis"] = analysis

            if "document_analysis" in st.session_state and st.session_state.document_analysis:
                st.markdown('<h3 class="sub-header">Document Analysis</h3>', unsafe_allow_html=True)

                # Allow editing the analysis
                edited_analysis = st.text_area(
                    "Edit analysis if needed:",
                    st.session_state.response_editable["document_analysis"],
                    height=400
                )
                st.session_state.response_editable["document_analysis"] = edited_analysis

                # Display the analysis with markdown formatting
                with st.expander("View Formatted Analysis", expanded=True):
                    st.markdown(edited_analysis)

                # Download options
                st.markdown('<h4 style="color: #4B5563; margin-top: 15px;">Download Options</h4>',
                            unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    # Download as text
                    txt = edited_analysis
                    st.download_button(
                        label="📄 Download as Text",
                        data=txt,
                        file_name="document_analysis.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                with col2:
                    # Download as PDF
                    pdf_buffer = convert_markdown_to_pdf(edited_analysis, "Document Analysis")
                    if pdf_buffer:
                        st.download_button(
                            label="📑 Download as PDF",
                            data=pdf_buffer,
                            file_name="document_analysis.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                # Image download option
                img = render_text_as_image(edited_analysis, "Document Analysis")
                if img:
                    st.markdown(get_image_download_link(img, "document_analysis.png", "📊 Download as Image"),
                                unsafe_allow_html=True)
    display_chatbot()
elif page == "Legal Chatbot":
    st.markdown('<h1 class="main-header">💬 Legal Compliance Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("""
      <div class="card">
          <p>Ask any legal or compliance-related questions. The AI assistant can help with:
          <ul>
              <li>Contract interpretation</li>
              <li>Regulatory compliance</li>
              <li>Business legalities</li>
              <li>General legal guidance</li>
          </ul>
          </p>
      </div>
      """, unsafe_allow_html=True)

    # Initialize states for voice recording
    if "is_recording_chat" not in st.session_state:
        st.session_state.is_recording_chat = False
    if "voice_chat_query" not in st.session_state:
        st.session_state.voice_chat_query = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    chat_history_text = "\n".join([f"You: {q}\nAI: {a}\n" for q, a in st.session_state.chat_history])
    st.text_area("Chat History", chat_history_text, height=300, disabled=True)

    # Voice input controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎤 Start Recording", key="start_chat_voice"):
            st.session_state.is_recording_chat = True
            with st.spinner("Listening..."):
                st.session_state.voice_chat_query = speech_to_text(st.session_state.language)
                if st.session_state.voice_chat_query:
                    st.success(f"Recognized: {st.session_state.voice_chat_query}")

    with col2:
        if st.session_state.is_recording_chat and st.button("⏹️ Stop Recording", key="stop_chat_voice"):
            st.session_state.is_recording_chat = False

    with col3:
        if st.session_state.chat_history:
            if st.button("🔊 Read Last Response", key="read_chat_response"):
                last_response = st.session_state.chat_history[-1][1]
                text_to_speech(last_response, st.session_state.language)

    # Chat input field with voice recognition text
    user_query = st.text_input(
        "Ask a legal question:",
        value=st.session_state.voice_chat_query,
        key=f"chat_input_{st.session_state.chat_input_counter}"
    )
    st.session_state.chat_input_counter += 1

    # Submit button
    if st.button("Send", key="send_chat_button"):
        if user_query.strip():
            with st.spinner("Getting response..."):
                response = get_chatbot_response(user_query, st.session_state.language)
                st.session_state.chat_history.append((user_query, response))
                st.session_state.voice_chat_query = ""  # Clear voice input after sending
            st.rerun()

    # Clear chat history button
    if st.button("Clear Chat History", key="clear_chat_history"):
        st.session_state.chat_history = []
        st.success("Chat history cleared successfully!")
        st.rerun()

    display_chatbot()

elif page == "Risk Assessment":
    st.markdown('<h1 class="main-header">🔍 Risk & Compliance Assessment</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p>Get a detailed risk assessment of your legal documents.
        The system evaluates contracts for potential risks, missing clauses, and compliance issues.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.document_text:
        st.warning("⚠️ Please upload and analyze a document first in the Document Analysis section.")
        if st.button("Go to Document Analysis", use_container_width=True):
            st.session_state.page = "Document Analysis"
            st.rerun()
    else:
        if st.button("🔍 Perform Risk Assessment", use_container_width=True):
            with st.spinner("Conducting risk assessment..."):
                risk_assessment = analyze_document(st.session_state.document_text, "risk_assessment")
                st.session_state.risk_assessment = risk_assessment
                # Initialize editable response
                st.session_state.response_editable["risk_assessment"] = risk_assessment

        if st.session_state.risk_assessment:
            st.markdown('<h3 class="sub-header">Risk Assessment Results</h3>', unsafe_allow_html=True)

            # Allow editing the risk assessment
            edited_risk = st.text_area(
                "Edit risk assessment if needed:",
                st.session_state.response_editable["risk_assessment"],
                height=400
            )
            st.session_state.response_editable["risk_assessment"] = edited_risk

            # Display with markdown formatting
            with st.expander("View Formatted Risk Assessment", expanded=True):
                st.markdown(edited_risk)

            # Download options
            st.markdown('<h4 style="color: #4B5563; margin-top: 15px;">Download Options</h4>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # Download as text
                txt = edited_risk
                st.download_button(
                    label="📄 Download as Text",
                    data=txt,
                    file_name="risk_assessment.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col2:
                # Download as PDF
                pdf_buffer = convert_markdown_to_pdf(edited_risk, "Risk Assessment")
                if pdf_buffer:
                    st.download_button(
                        label="📑 Download as PDF",
                        data=pdf_buffer,
                        file_name="risk_assessment.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            # Image download option
            img = render_text_as_image(edited_risk, "Risk Assessment")
            if img:
                st.markdown(get_image_download_link(img, "risk_assessment.png", "📊 Download as Image"),
                            unsafe_allow_html=True)
    display_chatbot()
elif page == "Document Generator":
    st.markdown('<h1 class="main-header">📝 Smart Legal Document Generator</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p>Generate customized legal documents based on your requirements.
        The AI assistant can create contracts, NDAs, MOUs, and more.</p>
    </div>
    """, unsafe_allow_html=True)

    # Add "Others" option to the document types list
    doc_types = [
        "Non-Disclosure Agreement (NDA)",
        "Memorandum of Understanding (MOU)",
        "Employment Contract",
        "Service Agreement",
        "Partnership Agreement",
        "Lease Agreement",
        "Sales Contract",
        "Privacy Policy",
        "Terms of Service",
        "Others"  # New option for custom document generation
    ]

    selected_doc = st.selectbox("Select Document Type", doc_types)

    if selected_doc != "Others":
        # Existing logic for predefined document types
        st.markdown('<h3 class="sub-header">Document Parameters</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if selected_doc == "Non-Disclosure Agreement (NDA)":
                party1 = st.text_input("Disclosing Party Name")
                party2 = st.text_input("Receiving Party")
                receiving_party = st.text_input("Receiving Party Name")
                purpose = st.text_area("Purpose of Disclosure")
                duration = st.number_input("Duration (in years)", min_value=1, max_value=10, value=2)
                state_law = st.text_input("Governing Law (State/Country)")

                parameters = f"""
                            - Disclosing Party: {party1}
                            - Receiving Party: {receiving_party}
                            - Purpose of Disclosure: {purpose}
                            - Duration: {duration} years
                            - Governing Law: {state_law}
                            """

            elif selected_doc == "Employment Contract":
                employer = st.text_input("Employer Name")
                employee = st.text_input("Employee Name")
                position = st.text_input("Job Position")
                start_date = st.date_input("Start Date")
                salary = st.number_input("Annual Salary", min_value=0, value=50000)
                benefits = st.text_area("Benefits")

                parameters = f"""
                                - Employer: {employer}
                                - Employee: {employee}
                                - Position: {position}
                                - Start Date: {start_date}
                                - Salary: ${salary}
                                - Benefits: {benefits}
                                """

            # ... (Add similar logic for other predefined document types)

        if st.button("📝 Generate Document", use_container_width=True):
            with st.spinner("Generating document..."):
                generated_doc = generate_document(selected_doc, parameters)
                st.session_state.generated_doc = generated_doc
                st.session_state.response_editable["generated_doc"] = generated_doc

    else:
        # Logic for "Others" option
        st.markdown('<h3 class="sub-header">Custom Document Requirements</h3>', unsafe_allow_html=True)

        # Chatbot-like interface for custom document generation
        st.markdown("""
        <div class="card">
            <p>Describe your document requirements in detail. The AI will generate a custom document based on your input.</p>
        </div>
        """, unsafe_allow_html=True)

        custom_query = st.text_area("Describe your document requirements:", height=150,
                                    placeholder="e.g., I need a custom agreement for a freelance project...")

        if st.button("📝 Generate Custom Document", use_container_width=True):
            if custom_query.strip():
                with st.spinner("Generating custom document..."):
                    # Use the custom query as parameters for document generation
                    generated_doc = generate_document("Custom Document", custom_query)
                    st.session_state.generated_doc = generated_doc
                    st.session_state.response_editable["generated_doc"] = generated_doc
            else:
                st.error("Please provide details about your document requirements.")

    # Display the generated document (common for both predefined and custom documents)
    if "generated_doc" in st.session_state:
        st.markdown('<h3 class="sub-header">Generated Document</h3>', unsafe_allow_html=True)

        edited_doc = st.text_area(
            "Edit document if needed:",
            st.session_state.response_editable["generated_doc"],
            height=400
        )
        st.session_state.response_editable["generated_doc"] = edited_doc

        with st.expander("View Formatted Document", expanded=True):
            st.markdown(edited_doc)

        st.markdown('<h4 style="color: #4B5563; margin-top: 15px;">Download Options</h4>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            txt = edited_doc
            st.download_button(
                label="📄 Download as Text",
                data=txt,
                file_name=f"{selected_doc.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:
            pdf_buffer = convert_markdown_to_pdf(edited_doc, selected_doc)
            if pdf_buffer:
                st.download_button(
                    label="📑 Download as PDF",
                    data=pdf_buffer,
                    file_name=f"{selected_doc.lower().replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        img = render_text_as_image(edited_doc, selected_doc)
        if img:
            st.markdown(
                get_image_download_link(
                    img,
                    f"{selected_doc.lower().replace(' ', '_')}.png",
                    "📊 Download as Image"
                ),
                unsafe_allow_html=True
            )
elif page == "Legal Research":
    st.markdown('<h1 class="main-header">🔎 Legal Research Assistant</h1>', unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
            <p>Research legal precedents, case law, and regulatory information. 
            The system helps find relevant cases and precedents to strengthen your legal position.</p>
        </div>
        """, unsafe_allow_html=True)

    # Initialize states for voice recording
    if "is_recording_research" not in st.session_state:
        st.session_state.is_recording_research = False
    if "voice_research_query" not in st.session_state:
        st.session_state.voice_research_query = ""

    # Research query input with voice recognition text
    research_query = st.text_area(
        "Describe the legal issue or question you want to research:",
        value=st.session_state.voice_research_query,
        height=100
    )

    jurisdiction = st.selectbox(
        "Jurisdiction",
        ["United States (Federal)", "United States (State)", "European Union",
         "United Kingdom", "Canada", "Australia", "International", "india"]
    )

    if jurisdiction == "india":
        state = st.selectbox(
            "Select State",
            ["Andhra Pradesh", "Arunachal Pradesh", "Bihar", "Assam", "Chhattisgarh",
             "Gujarat", "Goa", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
             "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
             "Tamil Nadu", "Tripura", "Telangana", "Uttar Pradesh", "Uttarakhand", "West Bengal"])

    timeframe = st.selectbox(
        "Timeframe",
        ["All Time", "Last 5 Years", "Last 10 Years", "Last 20 Years"]
    )

    # Voice input controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎤 Start Recording", key="start_research_voice"):
            st.session_state.is_recording_research = True
            with st.spinner("Listening for research query..."):
                st.session_state.voice_research_query = speech_to_text(st.session_state.language)
                if st.session_state.voice_research_query:
                    st.success(f"Recognized: {st.session_state.voice_research_query}")
                    research_query = st.session_state.voice_research_query

    with col2:
        if st.session_state.is_recording_research and st.button("⏹️ Stop Recording", key="stop_research_voice"):
            st.session_state.is_recording_research = False

    with col3:
        if "research_results" in st.session_state:
            if st.button("🔊 Read Results", key="read_research_results"):
                text_to_speech(st.session_state.research_results[:500], st.session_state.language)

    if st.button("🔎 Search Legal Precedents", use_container_width=True):
        if research_query:
            with st.spinner("Researching legal precedents..."):
                detailed_query = f"Research legal precedents for: {research_query}\n"
                detailed_query += f"Jurisdiction: {jurisdiction}\n"

                if jurisdiction == "United States (State)":
                    detailed_query += f"State: {state}\n"

                detailed_query += f"Timeframe: {timeframe}"

                research_results = search_legal_precedents(detailed_query)
                st.session_state.research_results = research_results
                st.session_state.response_editable["research_results"] = research_results
                st.session_state.voice_research_query = ""  # Clear voice input after searching
        else:
            st.error("Please enter a research query.")

    if "research_results" in st.session_state:
        st.markdown('<h3 class="sub-header">Research Results</h3>', unsafe_allow_html=True)

        edited_research = st.text_area(
            "Edit research results if needed:",
            st.session_state.response_editable["research_results"],
            height=400
        )
        st.session_state.response_editable["research_results"] = edited_research

        with st.expander("View Formatted Research Results", expanded=True):
            st.markdown(edited_research)

        # Download options section remains the same...

    display_chatbot()

    # Check if this is the first run
    if 'first_run' not in st.session_state:
        st.session_state.first_run = False

    # Show a welcome message on first run
    st.success("Welcome to AI Legal Assistant! Choose a service from the sidebar to get started.")