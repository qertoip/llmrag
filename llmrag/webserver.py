import gradio

from assistants.dummy_assistant import DummyAssistant
from assistants.kbonly_assistant import KBOnlyAssistant
from assistants.llmonly_assistant import LlmOnlyAssistant
from assistants.llmrag_assistant import LlmRagAssistant
from embedders.aws_universal_sentence_encoder_cmlm import AwsUniversalSentenceEncoderCMLM
from embedders.local_all_mpnet_base_v2 import LocalAllMpnetBaseV2
from vectordbs.chroma_vector_db import ChromaVectorDB


def main():
    #embedder = AwsUniversalSentenceEncoderCMLM()
    embedder = LocalAllMpnetBaseV2()

    vectordb = ChromaVectorDB()

    #assistant = DummyAssistant()
    #assistant = KBOnlyAssistant(embedder=embedder, vectordb=vectordb)
    #assistant = LlmOnlyAssistant()
    assistant = LlmRagAssistant(embedder=embedder, vectordb=vectordb)

    gradio_ui = gradio.ChatInterface(
        fn=assistant,
        chatbot=gradio.Chatbot(
            height=512,
            show_label=False,
            show_copy_button=False,
            sanitize_html=False,
            render_markdown=True
        ),
        textbox=gradio.Textbox(placeholder="Please ask a question", container=False, scale=7),
        title="Developer Docs 🕮 POC Assistant for X",
        description="👨 Hello! I have read all of our vast internal docs **so you don't have to**. Ask me a question.",
        theme="soft",
        examples=[
            'What is SageMaker?',
            'What are all AWS regions where SageMaker is available?',
            'How to check if an endpoint is KMS encrypted?',
            'What are SageMaker Geospatial capabilities?',
        ],
        cache_examples=False,
        retry_btn=None,
        submit_btn='Submit (20 sec) 🚀',
        undo_btn="Delete Previous",
        clear_btn="Clear",
        css="""
            footer { display:none !important }
            h1 { font-size: 2em !important }
            h2 { font-size: 1.6em !important }
            h4 { font-weight: bold !important; margin-top: 24px; padding-top: 24px; }
        """
    )
    gradio_ui.launch(server_name='127.0.0.1', server_port=8080)


if __name__ == '__main__':
    main()
