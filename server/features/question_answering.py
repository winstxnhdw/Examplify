from collections import deque
from itertools import chain

from server.features import LLM
from server.features.llm.types import Message


def question_answering(question: str, context: str, message_history: deque[Message]) -> deque[Message]:
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
    system_messages: tuple[Message, Message] = (
        {
            'content': 'You are a helpful AI assistant. You are given the following chat history. Answer the question based on the context provided as truthfully as you are able to. If you do not know the answer, you may respond with "I do not know". What is the Baloney Detection Kit?',
            'role': 'user'
        },
        {
            'content': 'The Baloney Detection Kit is a a set of cognitive tools and techniques that fortify the mind against penetration by falsehoods. It was created by Carl Sagan.',
            'role': 'assistant'
        }
    )

    message_history.append({
        'role': 'user',
        'content': f'Given the following context:\n\n{context}\n\nPlease answer the following question:\n\n{question}'
    })

    while not (answer := LLM.query(chain(system_messages, message_history))):
        message_history.popleft()
        message_history.popleft()

    message_history.append(answer)

    return message_history
