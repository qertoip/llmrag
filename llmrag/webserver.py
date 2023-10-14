import gradio

from assistants import dummy_assistant
from assistants import llm_assistant


def main():
    gradio_ui = gradio.ChatInterface(
        llm_assistant.answer_question,
        chatbot=gradio.Chatbot(height=512, show_label=False, show_copy_button=True),
        textbox=gradio.Textbox(placeholder="Please ask a question", container=False, scale=7),
        title="Corp 🕮 Assistant",
        description="👨 Ask anything <strong>work</strong> related.",
        theme="soft",
        examples=[
            'What is SageMaker?',
            'What are all AWS regions where SageMaker is available?',
            'How to check if an endpoint is KMS encrypted?',
            'What are SageMaker Geospatial capabilities?',
        ],
        cache_examples=False,
        retry_btn=None,
        submit_btn='Submit 🚀',
        undo_btn="Delete Previous",
        clear_btn="Clear",
        css="footer{ display:none !important }"
    )

    gradio_ui.launch(server_name='127.0.0.1', server_port=8080)


if __name__ == '__main__':
    main()
