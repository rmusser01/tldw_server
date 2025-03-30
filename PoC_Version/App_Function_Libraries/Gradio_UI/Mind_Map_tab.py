# Mind_Map_tab.py
# Description: File contains functions for generation of PlantUML mindmaps for the gradio tab
#
# Imports
import re
#
# External Libraries
import gradio as gr
#
######################################################################################################################
#
# Functions:

def parse_plantuml_mindmap(plantuml_text: str) -> dict:
    """Parse PlantUML mindmap syntax into a nested dictionary structure"""
    lines = [line.strip() for line in plantuml_text.split('\n')
             if line.strip() and not line.strip().startswith('@')]

    root = None
    nodes = []
    stack = []

    for line in lines:
        level_match = re.match(r'^([+\-*]+|\*+)', line)
        if not level_match:
            continue
        level = len(level_match.group(0))
        text = re.sub(r'^([+\-*]+|\*+)\s*', '', line).strip('[]').strip('()')
        node = {'text': text, 'children': []}

        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            stack[-1][1]['children'].append(node)
        else:
            root = node

        stack.append((level, node))

    return root

def create_mindmap_html(plantuml_text: str) -> str:
    """Convert PlantUML mindmap to HTML visualization with collapsible nodes using CSS only"""
    # Parse the mindmap text into a nested structure
    root_node = parse_plantuml_mindmap(plantuml_text)
    if not root_node:
        return "<p>No valid mindmap content provided.</p>"

    html = "<style>"
    html += """
    details {
        margin-left: 20px;
    }
    summary {
        cursor: pointer;
        padding: 5px;
        border: 1px solid #333;
        border-radius: 3px;
        background-color: #e6f3ff;
    }
    .mindmap-node {
        margin-left: 20px;
        padding: 5px;
        border: 1px solid #333;
        border-radius: 3px;
    }
    """
    html += "</style>"

    colors = ['#e6f3ff', '#f0f7ff', '#f5f5f5', '#fff0f0', '#f0fff0']

    def create_node_html(node, level):
        bg_color = colors[(level - 1) % len(colors)]
        if node['children']:
            children_html = ''.join(create_node_html(child, level + 1) for child in node['children'])
            return f"""
            <details open>
                <summary style="background-color: {bg_color};">{node['text']}</summary>
                {children_html}
            </details>
            """
        else:
            return f"""
            <div class="mindmap-node" style="background-color: {bg_color}; margin-left: {level * 20}px;">
                {node['text']}
            </div>
            """

    html += create_node_html(root_node, level=1)
    return html

# Create Gradio interface
def create_mindmap_tab():
    with gr.TabItem("PlantUML Mindmap"):
        gr.Markdown("# Collapsible PlantUML Mindmap Visualizer")
        gr.Markdown("Convert PlantUML mindmap syntax to a visual mindmap with collapsible nodes.")
        plantuml_input = gr.Textbox(
            lines=15,
            label="Enter PlantUML mindmap",
            placeholder="""@startmindmap
    * Project Planning
    ** Requirements
    *** Functional Requirements
    **** User Interface
    **** Backend Services
    *** Technical Requirements
    **** Performance
    **** Security
    ** Timeline
    *** Phase 1
    *** Phase 2
    ** Resources
    *** Team
    *** Budget
    @endmindmap"""
        )
        submit_btn = gr.Button("Generate Mindmap")
        mindmap_output = gr.HTML(label="Mindmap Output")
        submit_btn.click(
            fn=create_mindmap_html,
            inputs=plantuml_input,
            outputs=mindmap_output
        )

#
# End of Mind_Map_tab.py
######################################################################################################################
