# test_pdf_ingestion.py
# Usage: python test_pdf_ingestion.py <path_to_pdf_file>
#
# Imports
import logging
import os
import re
import sys
#
# 3rd-Party Imports
import pymupdf
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

#from App_Function_Libraries.PDF_Ingestion_Lib import extract_text_and_format_from_pdf, extract_metadata_from_pdf
def extract_text_and_format_from_pdf(pdf_path):
    """
    Extract text from a PDF file and convert it to Markdown, preserving formatting.
    """
    try:
        markdown_text = ""
        with pymupdf.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                markdown_text += f"## Page {page_num}\n\n"
                blocks = page.get_text("dict")["blocks"]
                current_paragraph = ""
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                text = span["text"]
                                font_size = span["size"]
                                font_flags = span["flags"]

                                # Apply formatting based on font size and flags
                                if font_size > 20:
                                    text = f"# {text}"
                                elif font_size > 16:
                                    text = f"## {text}"
                                elif font_size > 14:
                                    text = f"### {text}"

                                if font_flags & 2 ** 0:  # Bold
                                    text = f"**{text}**"
                                if font_flags & 2 ** 1:  # Italic
                                    text = f"*{text}*"

                                line_text += text + " "

                            # Remove hyphens at the end of lines
                            line_text = line_text.rstrip()
                            if line_text.endswith('-'):
                                line_text = line_text[:-1]
                            else:
                                line_text += " "

                            current_paragraph += line_text

                        # End of block, add paragraph
                        if current_paragraph:
                            # Remove extra spaces
                            current_paragraph = re.sub(r'\s+', ' ', current_paragraph).strip()
                            markdown_text += current_paragraph + "\n\n"
                            current_paragraph = ""
                    elif block["type"] == 1:  # Image block
                        markdown_text += "[Image]\n\n"
                markdown_text += "\n---\n\n"  # Page separator

        # Clean up hyphenated words
        markdown_text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', markdown_text)

        return markdown_text
    except Exception as e:
        logging.error(f"Error extracting text and formatting from PDF: {str(e)}")
        raise

def extract_metadata_from_pdf(pdf_path):
    """
    Extract metadata from a PDF file using PyMuPDF.
    """
    try:
        with pymupdf.open(pdf_path) as doc:
            metadata = doc.metadata
        return metadata
    except Exception as e:
        logging.error(f"Error extracting metadata from PDF: {str(e)}")
        return {}


logging.basicConfig(level=logging.INFO)


def test_pdf_ingestion(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return

    try:
        # Extract metadata
        metadata = extract_metadata_from_pdf(pdf_path)
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        print("\n" + "=" * 50 + "\n")

        # Extract and convert text to Markdown
        markdown_text = extract_text_and_format_from_pdf(pdf_path)

        # Print the first 1000 characters of the Markdown text
        print("Extracted Markdown (first 1000 characters):")
        print(markdown_text[:1000])
        print("...\n")

        # Save the full Markdown text to a file
        output_file = os.path.splitext(pdf_path)[0] + "_converted.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        print(f"Full Markdown content saved to: {output_file}")

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pdf_ingestion.py <path_to_pdf_file>")
    else:
        pdf_path = sys.argv[1]
        test_pdf_ingestion(pdf_path)

#
# End of test_pdf_ingestion.py
####################################################################################################
