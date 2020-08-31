"""
Utilities for pre-processing sequence data.
Special tokens that are in all dictionaries:
<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
<PAD>: To pad to the sequences to make them of equal length
"""

SPECIAL_TOKENS = {
    "<NULL>": 0,
    "<START>": 1,
    "<END>": 2,
    "<UNK>": 3,
    "<PAD>": 4
}


def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string)
    tokens by splitting on the specified delimiter. Optionally keep or
    remove certain punctuation marks and add start and end tokens.

    :param s: string to tokenize
    :param delim: delimiter
    :param add_start_token: start token to identify string
    :param add_end_token: end token to identify string
    :param punct_to_keep: punctuations to keep
    :param punct_to_remove: punctuations not to be considered

    :return: list containing all the tokens of the string
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, "%s%s" % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, "")

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, "<START>")
    if add_end_token:
        tokens.append("<END>")

    return tokens


def build_vocab(sequences, min_token_count=1, delim=" ", punct_to_keep=None,
                punct_to_remove=None):
    """

    :param sequences: list of vocabulary to be tokenized
    :param min_token_count: minimum count to be considered while tokenizing
    :param delim: delimiting character amongst tokens
    :param punct_to_keep: punctuations to keep
    :param punct_to_remove: punctuations to remove

    :return: dictionary containing words as keys and their tokens as value
    """

    token_to_count = {}
    tokenize_kwargs = {
        "delim": delim,
        "punct_to_keep": punct_to_keep,
        "punct_to_remove": punct_to_remove,
    }

    for seq in sequences:
        seq_tokens = tokenize(seq, **tokenize_kwargs, add_start_token=False,
                              add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    """

    :param seq_tokens: sequence of tokens
    :param token_to_idx: vocabulary
    :param allow_unk: if any specific token in the sequence is not contained
                      in the vocab, then set token as <UNK> (Unknown)
    :return: dictionary with tokens as keys and their ids as values
    """
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = "<UNK>"
            else:
                raise KeyError("Token '%s' not in vocab" % token)
        seq_idx.append((token_to_idx[token]))

    return seq_idx
