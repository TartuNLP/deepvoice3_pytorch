# coding: utf-8

symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'()[],-.:;? ÕÄÖÜŠŽõäöüšž"“„’*«»'
n_vocab = len(symbols)


def _character_subtitutions(text):
    import re
    text = re.sub(r'[()[\]:;−­]', r', ', text)
    # text = re.sub(r'["“`´„’*\'\\/-]', r' ', text)
    text = re.sub(r'[«»“„]', r'"', text)
    text = re.sub(r'[`´’*\'\\/-]', r' ', text)
    text = re.sub(r' +([.,!?])', r'\g<1>', text)
    text = re.sub(r', ?([.,?!])', r'\g<1>', text)
    text = re.sub(r'\.+', r'.', text)

    text = re.sub(r' +', r' ', text)
    text = re.sub(r'^ | $', r'', text)
    return text


def text_to_sequence(text, p=0.0):
    from deepvoice3_pytorch.frontend.text import text_to_sequence
    from deepvoice3_pytorch.frontend.et.tts_preprocess_et.convert import convert_sentence
    try:
        text = convert_sentence(text)
    except Exception as e:
        print(str(e), text)
    text = _character_subtitutions(text)
    text = text_to_sequence(text, ["basic_cleaners"])

    return text


from deepvoice3_pytorch.frontend.text import sequence_to_text