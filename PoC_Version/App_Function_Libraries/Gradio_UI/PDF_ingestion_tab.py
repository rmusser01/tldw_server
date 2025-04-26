# PDF_ingestion_tab.py
# Gradio UI for ingesting PDFs into the database
import os
import shutil
import tempfile

# Imports
#
# External Imports
import gradio as gr
import pymupdf4llm
from docling.document_converter import DocumentConverter

#
# Local Imports
from PoC_Version.App_Function_Libraries.PDF.PDF_Ingestion_Lib import extract_metadata_from_pdf, extract_text_and_format_from_pdf, \
    process_and_cleanup_pdf
#
#
########################################################################################################################
#
# Functions:

def create_pdf_ingestion_tab():
    with gr.TabItem("PDF Ingestion", visible=True):
        gr.Markdown("# Ingest PDF(/pptx) Files and Extract Metadata (Docling Supports parsing of pptx files)")
        with gr.Row():
            with gr.Column():
                # Changed to support multiple files
                pdf_file_input = gr.File(
                    label="Uploaded PDF Files",
                    file_types=[".pdf"],
                    visible=True,
                    file_count="multiple"
                )
                pdf_upload_button = gr.UploadButton(
                    "Click to Upload PDFs",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                parser_selection = gr.Radio(
                    choices=["pymupdf", "pymupdf4llm", "docling"],
                    label="Select Parser",
                    value="pymupdf"  # default value
                )
                # Common metadata for all files
                pdf_keywords_input = gr.Textbox(label="Keywords (Optional, comma-separated)")
                pdf_ingest_button = gr.Button("Ingest PDFs")

                # Update the upload button handler for multiple files
                pdf_upload_button.upload(
                    fn=lambda files: files,
                    inputs=pdf_upload_button,
                    outputs=pdf_file_input
                )

            with gr.Column():
                pdf_result_output = gr.DataFrame(
                    headers=["Filename", "Status", "Message"],
                    label="Processing Results"
                )

            # Define a new function to handle multiple PDFs
            def process_multiple_pdfs(pdf_files, keywords, custom_prompt_checkbox_value, custom_prompt_text, system_prompt_text):
                results = []
                if pdf_files is None:
                    return [["No files", "Error", "No files uploaded"]]

                for pdf_file in pdf_files:
                    try:
                        # Extract metadata from PDF
                        metadata = extract_metadata_from_pdf(pdf_file.name)

                        # Use custom or system prompt if checkbox is checked
                        if custom_prompt_checkbox_value:
                            prompt = custom_prompt_text
                            system_prompt = system_prompt_text
                        else:
                            prompt = None
                            system_prompt = None

                        # Process the PDF with prompts
                        result = process_and_cleanup_pdf(
                            pdf_file,
                            metadata.get('title', os.path.splitext(os.path.basename(pdf_file.name))[0]),
                            metadata.get('author', 'Unknown'),
                            keywords,
                            #prompt=prompt,
                            #system_prompt=system_prompt
                        )

                        results.append([
                            pdf_file.name,
                            "Success" if "successfully" in result else "Error",
                            result
                        ])
                    except Exception as e:
                        results.append([
                            pdf_file.name,
                            "Error",
                            str(e)
                        ])

                return results

            # Update the ingest button click handler
            pdf_ingest_button.click(
                fn=process_multiple_pdfs,
                inputs=[
                    pdf_file_input,
                    pdf_keywords_input,
                    parser_selection,
                    #custom_prompt_checkbox,
                    #custom_prompt_input,
                    #system_prompt_input
                ],
                outputs=pdf_result_output
            )


def test_pymupdf4llm_pdf_ingestion(pdf_file):
    if pdf_file is None:
        return "No file uploaded", ""

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a path for the temporary PDF file
            temp_path = os.path.join(temp_dir, "temp.pdf")

            # Copy the contents of the uploaded file to the temporary file
            shutil.copy(pdf_file.name, temp_path)

            # Extract text and convert to Markdown
            markdown_text = pymupdf4llm.to_markdown(temp_path)

            # Extract metadata from PDF
            metadata = extract_metadata_from_pdf(temp_path)

            # Use metadata for title and author if not provided
            title = metadata.get('title', os.path.splitext(os.path.basename(pdf_file.name))[0])
            author = metadata.get('author', 'Unknown')

        result = f"PDF '{title}' by {author} processed successfully by pymupdf4llm."
        return result, markdown_text
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}", ""


def test_pymupdf_pdf_ingestion(pdf_file):
    if pdf_file is None:
        return "No file uploaded", ""

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a path for the temporary PDF file
            temp_path = os.path.join(temp_dir, "temp.pdf")

            # Copy the contents of the uploaded file to the temporary file
            shutil.copy(pdf_file.name, temp_path)

            # Extract text and convert to Markdown
            markdown_text = extract_text_and_format_from_pdf(temp_path)

            # Extract metadata from PDF
            metadata = extract_metadata_from_pdf(temp_path)

            # Use metadata for title and author if not provided
            title = metadata.get('title', os.path.splitext(os.path.basename(pdf_file.name))[0])
            author = metadata.get('author', 'Unknown')

        result = f"PDF '{title}' by {author} processed successfully by pymupdf."
        return result, markdown_text
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}", ""


def test_docling_pdf_ingestion(pdf_file):
    if pdf_file is None:
        return "No file uploaded", ""

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a path for the temporary PDF file
            temp_path = os.path.join(temp_dir, "temp.pdf")

            # Copy the contents of the uploaded file to the temporary file
            shutil.copy(pdf_file.name, temp_path)

            # Extract text and convert to Markdown
            converter = DocumentConverter()
            parsed_pdf = converter.convert(temp_path)
            markdown_text = parsed_pdf.document.export_to_markdown()
            # Extract metadata from PDF
            metadata = extract_metadata_from_pdf(temp_path)

            # Use metadata for title and author if not provided
            title = metadata.get('title', os.path.splitext(os.path.basename(pdf_file.name))[0])
            author = metadata.get('author', 'Unknown')

        result = f"PDF '{title}' by {author} processed successfully by Docling."
        return result, markdown_text
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}", ""

def create_pdf_ingestion_test_tab():
    with gr.TabItem("Test PDF Ingestion", visible=True):
        with gr.Row():
            with gr.Column():
                pdf_file_input = gr.File(label="Upload PDF for testing")
                test_button = gr.Button("Test pymupdf PDF Ingestion")
                test_button_2 = gr.Button("Test pymupdf4llm PDF Ingestion")
                test_button_3 = gr.Button("Test Docling PDF Ingestion")
            with gr.Column():
                test_output = gr.Textbox(label="Test Result")
                pdf_content_output = gr.Textbox(label="PDF Content", lines=200, elem_id="scrollable-textbox")
        test_button.click(
            fn=test_pymupdf_pdf_ingestion,
            inputs=[pdf_file_input],
            outputs=[test_output, pdf_content_output]
        )
        test_button_2.click(
            fn=test_pymupdf4llm_pdf_ingestion,
            inputs=[pdf_file_input],
            outputs=[test_output, pdf_content_output]
        )
        test_button_3.click(
            fn=test_docling_pdf_ingestion,
            inputs=[pdf_file_input],
            outputs=[test_output, pdf_content_output]
        )

