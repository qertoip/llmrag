from assistants.assistant import Assistant


class DummyAssistant(Assistant):
    """
    Use for testing UI.
    """

    def answer_question(self, question, history):
        if question.endswith("?"):
            return "Yes."
        else:
            return "Ask me anything and I will confirm."
