# coding: utf-8
from deepvoice3_pytorch.frontend.text.symbols import symbols

n_vocab = len(symbols)


def character_subtitutions(text):
    import re
    text = re.sub(r'[()[\]:;−­–…—]', r', ', text)
    text = re.sub(r'[«»“„”]', r'"', text)
    text = re.sub(r'[*\'\\/-]', r' ', text)
    text = re.sub(r'[`´’\']', r'', text)
    text = re.sub(r' +([.,!?])', r'\g<1>', text)
    text = re.sub(r', ?([.,?!])', r'\g<1>', text)
    text = re.sub(r'\.+', r'.', text)

    text = re.sub(r' +', r' ', text)
    text = re.sub(r'^ | $', r'', text)
    text = re.sub(r'^, ?', r'', text)

    return text


def text_to_sequence(text, p=0.0):
    from deepvoice3_pytorch.frontend.text import text_to_sequence
    import re
    from deepvoice3_pytorch.frontend.et.tts_preprocess_et.convert import convert_sentence
    text = re.sub(r'[`´’\']', r'', text)  # TODO: may not be advised but currently "See'p" -> "see pee"
    text = re.sub(r'[()]', r', ', text)  # TODO: not pronunced in corpus
    try:
        text = convert_sentence(text)
    except Exception as e:
        print(str(e), text)
    text = character_subtitutions(text)
    text = text_to_sequence(text, ["basic_cleaners"])

    return text


from deepvoice3_pytorch.frontend.text import sequence_to_text