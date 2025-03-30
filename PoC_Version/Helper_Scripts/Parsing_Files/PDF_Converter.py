# pdf_converter.py

import sys
import marker_pdf


def convert_pdf_to_markdown(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        markdown_content = marker_pdf.convert(pdf_file)
    return markdown_content


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_converter.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    try:
        markdown = convert_pdf_to_markdown(pdf_path)
        print(markdown)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)