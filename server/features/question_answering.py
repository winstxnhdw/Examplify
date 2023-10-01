from collections import deque

from server.features import LLM
from server.features.llm.types import Message


def question_answering(question: str, context: str, message_history: list[Message]) -> deque[Message]:
    """
    Summary
    -------
    ask a question and get an answer

    Parameters
    ----------
    question (str): the question to ask
    context (str): the context to ask the question in
    message_history (list[Message]): the message history

    Returns
    -------
    deque[Message]: the message history
    """
    messages = deque(message_history)

    messages.append({
        'role': 'user',
        'content': f'Given the following context:\n\n{context}\n\nPlease answer the following question:\n\n{question}'
    })

    while not (answer := LLM.query(messages)):
        messages.popleft()
        messages.popleft()

    messages.append(answer)

    return messages
