import os

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_compiled_prompt(output_path, title, system_content, user_content):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(f"### TITLE ###\n{title}\n\n")
        file.write("### AUTHOR ###\nfabric project\n\n")
        file.write(f"### SYSTEM ###\n{system_content.strip()}\n\n")
        file.write(f"### USER ###\n{user_content.strip()}\n\n")
        file.write("### KEYWORDS ###\n")

def compile_prompts(base_folder):
    patterns_folder = os.path.join(base_folder, 'patterns')
    
    for prompt_folder in os.listdir(patterns_folder):
        prompt_path = os.path.join(patterns_folder, prompt_folder)
        
        if os.path.isdir(prompt_path):
            system_file = os.path.join(prompt_path, 'system.md')
            user_file = os.path.join(prompt_path, 'user.md')
            
            if os.path.exists(system_file) and os.path.exists(user_file):
                system_content = read_file_content(system_file)
                user_content = read_file_content(user_file)
                
                title = prompt_folder.replace(' ', '_')
                output_file = f"{title}.md"
                output_path = os.path.join(base_folder, output_file)
                
                write_compiled_prompt(output_path, prompt_folder, system_content, user_content)
                print(f"Compiled prompt: {output_file}")

if __name__ == "__main__":
    base_folder = input("Enter the path to the base folder: ")
    compile_prompts(base_folder)
