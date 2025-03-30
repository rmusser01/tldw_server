# html_to_markdown/main.py
# Usage: python -m html_to_markdown.main input.html output.md --extract-main --refify-urls --include-meta extended --debug
# Arguments:
#     input.html: Path to your input HTML file.
#     output.md: Desired path for the output Markdown file.
#     --extract-main: (Optional) Extracts the main content from the HTML.
#     --refify-urls: (Optional) Refactors URLs to reference-style.
#     --include-meta: (Optional) Includes metadata. Choose between basic or extended.
#     --debug: (Optional) Enables debug logging for detailed trace.

from html_to_markdown import convert_html_to_markdown
from conversion_options import ConversionOptions

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert HTML to Markdown.")
    parser.add_argument('input_file', help="Path to the input HTML file.")
    parser.add_argument('output_file', help="Path to the output Markdown file.")
    parser.add_argument('--extract-main', action='store_true', help="Extract main content.")
    parser.add_argument('--refify-urls', action='store_true', help="Refify URLs.")
    parser.add_argument('--include-meta', choices=['basic', 'extended'], default=False, help="Include metadata.")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging.")

    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    options = ConversionOptions(
        extract_main_content=args.extract_main,
        refify_urls=args.refify_urls,
        include_meta_data=args.include_meta if args.include_meta else False,
        debug=args.debug
    )

    markdown = convert_html_to_markdown(html_content, options)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"Conversion complete. Markdown saved to {args.output_file}")

if __name__ == "__main__":
    main()
