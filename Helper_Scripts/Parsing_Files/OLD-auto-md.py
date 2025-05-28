# Script taken from https://github.com/tegridydev/auto-md
# Description: automd~ is a mini python based tool designed to convert various types of files and GitHub repositories into LLM-ready Markdown documents

import os
import re
import zipfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import subprocess
import stat
from datetime import datetime
import logging
import tempfile
import uuid
import threading
import queue


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


# file~~extensions~~configuration
TEXT_EXTENSIONS = [
    '.txt', '.md', '.html', '.css', '.py', '.js', '.yaml', '.yml',
    '.json', '.xml', '.csv', '.rst', '.ini', '.cfg', '.log', '.conf'
]

SKIP_EXTENSIONS = [
    '.exe', '.dll', '.bin', '.img', '.iso', '.tar', '.gz', '.zip',
    '.rar', '.7z', '.mp3', '.mp4', '.wav', '.flac', '.mov', '.avi',
    '.mkv', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico'
]


def clean_text(text):
    """
    Perform basic cleaning of the text.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()


def format_as_markdown(text, title, file_path, all_files):
    """
    Format the cleaned text into an optimized Markdown structure for LLM indexing.
    """
    lines = text.split('\n')
    formatted_text = f"# {title}\n\n"
    formatted_text += "## Metadata\n"
    formatted_text += f"- **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    if 'github.com' in file_path.lower():

        repo_parts = file_path.split('/')
        source = f"{repo_parts[-2]}/{repo_parts[-1]}"
    else:

        source = os.path.basename(os.path.dirname(file_path))

    formatted_text += f"- **Source:** {source}\n\n"

    formatted_text += "## Table of Contents\n"
    for file in all_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        formatted_text += f"- [{file_name}](#{file_name.lower().replace(' ', '-')})\n"
    formatted_text += "\n"

    formatted_text += f"# {title}\n\n"
    for line in lines:
        if line.strip():
            formatted_text += f"{line.strip()}\n\n"

    return formatted_text


def process_file(file_path, output_dir, single_file, all_files):
    """
    Process a single text file, clean, format, and either save to individual file or return content.
    """
    logging.info(f"Processing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        if not text.strip():
            logging.warning(f"File is empty: {file_path}")
            return None
        title = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').replace('-', ' ')
        cleaned_text = clean_text(text)
        markdown_text = format_as_markdown(cleaned_text, title, file_path, all_files)

        if single_file:
            return markdown_text
        else:
            output_file = os.path.join(output_dir, f"{title}.md")
            with open(output_file, 'w', encoding='utf-8') as out_file:
                out_file.write(markdown_text)
            logging.info(f"Saved markdown to: {output_file}")
        return markdown_text
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None


def process_folder(folder_path, output_dir, single_file, combined_content, all_files):
    """
    Process all text files in a given folder (and its subfolders).
    """
    logging.info(f"Processing folder: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file.endswith(ext) for ext in TEXT_EXTENSIONS):
                all_files.append(file_path)
                result = process_file(file_path, output_dir, single_file, all_files)
                if result:
                    combined_content.append(result)
            elif file.endswith('.zip'):
                temp_extract_to = os.path.join(root, f"temp_{os.path.basename(file)}")
                extract_zip(file_path, temp_extract_to)
                process_folder(temp_extract_to, output_dir, single_file, combined_content, all_files)
                shutil.rmtree(temp_extract_to, ignore_errors=True)
            elif any(file.endswith(ext) for ext in SKIP_EXTENSIONS):
                logging.info(f"Skipping file: {file_path}")


def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to the specified directory.
    """
    logging.info(f"Extracting zip file: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info(f"Extracted to: {extract_to}")
    except Exception as e:
        logging.error(f"Error extracting zip file {zip_path}: {e}")


def clone_git_repo(repo_url, temp_folder, depth=None):
    """
    Clone a GitHub repository to the specified directory.
    """
    logging.info(f"Cloning GitHub repository: {repo_url}")
    try:
        cmd = ["git", "clone"]
        if depth is not None:
            cmd.extend(["--depth", str(depth)])
        cmd.extend([repo_url, temp_folder])
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(result.stdout)
        logging.info(f"Cloned to: {temp_folder}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error cloning GitHub repository {repo_url}: {e}")
        logging.error(e.stderr)


def process_input(input_paths, output_path, temp_folder, single_file, repo_depth):
    """
    Process each item in the input paths: directories, text files, zip files, and GitHub repos.
    """
    combined_content = []
    all_files = []

    if single_file:
        output_dir = os.path.dirname(output_path)
    else:
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)

    for item_path in input_paths:
        if os.path.isdir(item_path):
            process_folder(item_path, output_dir, single_file, combined_content, all_files)
        elif any(item_path.endswith(ext) for ext in TEXT_EXTENSIONS):
            all_files.append(item_path)
            result = process_file(item_path, output_dir, single_file, all_files)
            if result:
                combined_content.append(result)
        elif item_path.endswith('.zip'):
            extract_to = os.path.join(temp_folder, os.path.basename(item_path).replace('.zip', ''))
            extract_zip(item_path, extract_to)
            process_folder(extract_to, output_dir, single_file, combined_content, all_files)
            shutil.rmtree(extract_to, ignore_errors=True)
        elif item_path.startswith("https://github.com"):
            repo_name = os.path.basename(item_path).replace('.git', '')
            repo_temp_folder = os.path.join(temp_folder, repo_name)
            clone_git_repo(item_path, repo_temp_folder, depth=repo_depth)
            process_folder(repo_temp_folder, output_dir, single_file, combined_content, all_files)
            shutil.rmtree(repo_temp_folder, ignore_errors=True)
        elif any(item_path.endswith(ext) for ext in SKIP_EXTENSIONS):
            logging.info(f"Skipping file: {item_path}")

    if single_file and combined_content:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write("\n---\n\n".join(combined_content))
        logging.info(f"Combined content saved to: {output_path}")
    elif single_file and not combined_content:
        logging.warning("No content was processed. Output file not created.")
    else:
        logging.info(f"Individual Markdown files saved in: {output_dir}")

    return output_dir if not single_file else os.path.dirname(output_path)


class AutoMDApp:
    def __init__(self, master):
        self.master = master
        master.title("automd~~")
        master.geometry("800x600")
        master.resizable(False, False)

        self.temp_dir = os.path.join(tempfile.gettempdir(), f"automd_temp_{uuid.uuid4().hex}")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(self.queue_handler)

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        ttk.Label(main_frame, text="Input Files or GitHub Repos:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_files_entry = ttk.Entry(main_frame, width=60)
        self.input_files_entry.grid(row=0, column=1, pady=5, padx=(0, 5))
        ttk.Button(main_frame, text="Browse", command=self.browse_input_files).grid(row=0, column=2, pady=5)

        ttk.Label(main_frame, text="Output:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_entry = ttk.Entry(main_frame, width=60)
        self.output_entry.grid(row=1, column=1, pady=5, padx=(0, 5))
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=5)

        self.single_file_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Output to single file", variable=self.single_file_var,
                        command=self.toggle_output_mode).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)

        ttk.Label(main_frame, text="Repository clone depth:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.repo_depth_var = tk.StringVar(value="Full")
        repo_depth_combo = ttk.Combobox(main_frame, textvariable=self.repo_depth_var,
                                        values=["Full", "1", "5", "10", "20", "50", "100"])
        repo_depth_combo.grid(row=3, column=1, sticky=tk.W, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)

        self.console_output = scrolledtext.ScrolledText(main_frame, height=15)
        self.console_output.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        self.console_output.config(state=tk.DISABLED)

        ttk.Button(main_frame, text="Start Processing", command=self.start_processing).grid(row=7, column=1, pady=20)

        self.master.after(100, self.check_queue)

    def check_queue(self):
        while True:
            try:
                message = self.log_queue.get_nowait()
                self.console_output.config(state=tk.NORMAL)
                self.console_output.insert(tk.END, message + '\n')
                self.console_output.see(tk.END)
                self.console_output.config(state=tk.DISABLED)
            except queue.Empty:
                break
        self.master.after(100, self.check_queue)

    def browse_input_files(self):
        files_selected = filedialog.askopenfilenames(filetypes=[("All files", "*.*"), ("Zip files", "*.zip")])
        self.input_files_entry.delete(0, tk.END)
        self.input_files_entry.insert(0, " ".join(files_selected))
        if files_selected:
            output_name = os.path.splitext(os.path.basename(files_selected[0]))[0]
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, output_name + (".md" if self.single_file_var.get() else ""))

    def browse_output(self):
        if self.single_file_var.get():
            file_selected = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown files", "*.md")])
        else:
            file_selected = filedialog.askdirectory()
        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, file_selected)

    def toggle_output_mode(self):
        current_output = self.output_entry.get()
        if self.single_file_var.get():
            if not current_output.endswith('.md'):
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, current_output + '.md')
        else:
            if current_output.endswith('.md'):
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, os.path.splitext(current_output)[0])

    def start_processing(self):
        input_files = self.input_files_entry.get().split()
        output_path = self.output_entry.get()
        single_file = self.single_file_var.get()
        repo_depth = None if self.repo_depth_var.get() == "Full" else int(self.repo_depth_var.get())

        if not input_files:
            messagebox.showerror("Error", "Please select input files or repositories.")
            return

        if not output_path:
            messagebox.showerror("Error", "Please specify an output path.")
            return

        self.status_label.config(text="Processing...")
        self.progress_var.set(0)
        self.master.update()

        def process_thread():
            try:
                output_dir = process_input(input_files, output_path, self.temp_dir, single_file, repo_depth)
                self.master.after(0, lambda: self.progress_var.set(100))
                self.master.after(0, lambda: self.status_label.config(text="Processing complete!"))
                self.master.after(0, lambda: self.show_completion_dialog(output_dir))
            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
                self.master.after(0, lambda: self.status_label.config(text="Processing failed."))
            finally:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

        threading.Thread(target=process_thread, daemon=True).start()

    def show_completion_dialog(self, output_dir):
        result = messagebox.askquestion("Success", "Processing complete. Would you like to open the containing folder?",
                                        icon='info')
        if result == 'yes':
            self.open_output_folder(output_dir)

    def open_output_folder(self, path):
        if os.path.isfile(path):
            path = os.path.dirname(path)
        os.startfile(path)


if __name__ == "__main__":
    root = tk.Tk()
    app = AutoMDApp(root)
    root.mainloop()
