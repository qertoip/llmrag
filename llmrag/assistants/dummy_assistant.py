def answer_question(question, history):
    if question.endswith("?"):
        return "Yes."
    else:
        return "Ask me anything and I will confirm."
