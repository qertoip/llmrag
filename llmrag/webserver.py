import gradio

from assistants.dummy_assistant import DummyAssistant
from assistants.kbonly_assistant import KBOnlyAssistant
from assistants.llmonly_assistant import LlmOnlyAssistant
from assistants.llmrag_assistant import LlmRagAssistant
from embedders.aws_universal_sentence_encoder_cmlm import AwsUniversalSentenceEncoderCMLM
from embedders.local_universal_sentence_encoder import LocalUniversalSentenceEncoder
from embedders.local_all_mpnet_base_v2 import LocalAllMpnetBaseV2
from vectordbs.chroma_vector_db import ChromaVectorDB


def main():
    dummy_assistant = DummyAssistant()

    #embedder = AwsUniversalSentenceEncoderCMLM()
    #embedder = LocalUniversalSentenceEncoder()
    embedder = LocalAllMpnetBaseV2()

    vectordb = ChromaVectorDB()

    #assistant = KBOnlyAssistant(embedder=embedder, vectordb=vectordb)
    #assistant = LlmOnlyAssistant()
    assistant = LlmRagAssistant(embedder, vectordb)

    gradio_ui = gradio.ChatInterface(
        fn=assistant,
        chatbot=gradio.Chatbot(
            height=1024,
            show_label=False,
            show_copy_button=False,
            sanitize_html=False,
            render_markdown=True
        ),
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
        css="""
            footer { display:none !important }
            h1 { font-size: 2em !important }
            h2 { font-size: 1.6em !important }
        """
    )
    gradio_ui.launch(server_name='127.0.0.1', server_port=8080)


if __name__ == '__main__':
    main()
