# coding: utf-8
from deepvoice3_pytorch.frontend.text.symbols import symbols

n_vocab = len(symbols)


def text_to_sequence(text, p=0.0):
    from deepvoice3_pytorch.frontend.text import text_to_sequence
    from deepvoice3_pytorch.frontend.et import character_subtitutions
    from deepvoice3_pytorch.frontend.et.tts_preprocess_et.convert import simplify_unicode
    text = simplify_unicode(text)
    text = character_subtitutions(text)
    text = text_to_sequence(text, ["basic_cleaners"])

    return text


from deepvoice3_pytorch.frontend.text import sequence_to_text