import gradio as gr
import os
import traceback

from library import run_program

DEFAULT_OUTPUT_DIR = "output/"

def run_wrapper(input_file, output_file, model_name, use_full_path, strip_punct):
    try:
        input_path = input_file.name

        if not use_full_path:
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            output_file = os.path.join(DEFAULT_OUTPUT_DIR, output_file)

        if not (output_file.endswith('.csv') or output_file.endswith('.txt')):
            return "Please provide a valid .csv or .txt file for output!"

        if os.path.isdir(output_file):
            return "Please provide a valid file name, not just a directory."

        result_df = run_program(input_path, model_name, strip_punct)

        if output_file.endswith('.csv'):
            result_df.to_csv(output_file, index=False)
        elif output_file.endswith('.txt'):
            result_df.to_csv(output_file, sep='\t', index=False)

        return f"Program executed successfully! DataFrame has been written to: {output_file}"

    except Exception:
        error_message = traceback.format_exc()
        return f"An error occurred:\n{error_message}"

def create_app():
    with gr.Blocks() as demo:
        gr.Markdown("## Run Your Program")

        input_file = gr.File(label="Input File", file_types=['file'])
        use_full_path = gr.Checkbox(label="Use full output file path", value=False)
        strip_punct = gr.Checkbox(label="Strip punctuation")
        output_file = gr.Textbox(
            label="Output File Path or Name",
            placeholder="Enter full path or just file name (e.g., output.txt)"
        )

        model_name = gr.Dropdown(
            label="Select Model",
            choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
            value="gpt2"
        )

        run_button = gr.Button("Run")
        output_message = gr.Textbox(label="Output Message")

        run_button.click(
            run_wrapper,
            inputs=[input_file, output_file, model_name, use_full_path, strip_punct],
            outputs=output_message
        )

    demo.launch()

if __name__ == "__main__":
    create_app()