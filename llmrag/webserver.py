import os

import gradio


def yes_man(message, history):
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"


def main():
    gradio_ui = gradio.ChatInterface(
        yes_man,
        chatbot=gradio.Chatbot(height=300),
        textbox=gradio.Textbox(placeholder="Ask me a yes or no question!", container=False, scale=7),
        title="Yes Man",
        description="Ask Yes Man any question",
        theme="soft",
        examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
        cache_examples=True,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    )

    #port = int(os.environ.get("PORT"))
    gradio_ui.launch(server_name='0.0.0.0', server_port=8080)


if __name__ == '__main__':
    main()
