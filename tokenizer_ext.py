import multiprocessing as mp
from functools import partial

# ---------- CORE FUNCTIONS (single text, distributed across cores) ----------

# global tokenizer reference inside workers
_tokenizer = None
_pad_token_id = None

def _init_worker(tokenizer, pad_token_id):
    global _tokenizer, _pad_token_id
    _tokenizer = tokenizer
    _pad_token_id = pad_token_id

def _encode_chunk(chunk):
    return _tokenizer.encode(chunk)

def _decode_chunk(tokens):
    tokens = [t for t in tokens if t != _pad_token_id]
    return _tokenizer.decode(tokens)

def encode_text_parallel(tokenizer, text, num_workers=None, chunk_size=256):
    """
    Encode a single text by splitting into chunks and distributing across cores.
    """
    # split text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer, None)
    ) as pool:
        encoded_chunks = pool.map(_encode_chunk, chunks)

    # flatten result
    return [tok for chunk in encoded_chunks for tok in chunk]

def decode_text_parallel(tokenizer, tokens, pad_token_id=0, num_workers=None, chunk_size=256):
    """
    Decode a single sequence of tokens in parallel, ignoring padding.
    """
    # split into chunks
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer, None)
    ) as pool:
        decoded_chunks = pool.map(_decode_chunk, chunks)

    return "".join(decoded_chunks)


# ---------- BATCH FUNCTIONS (multiple texts, distributed again) ----------

def batch_encode_parallel(tokenizer, texts, pad_token_id=0, max_length=None,
                          num_workers=None, chunk_size=256):
    """
    Encode a batch of texts using core parallel encode_text_parallel,
    then pad results.
    """
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer, None)
    ) as pool:
        sequences = pool.map(
            partial(encode_text_parallel, tokenizer, num_workers=num_workers, chunk_size=chunk_size),
            texts
        )

    # determine padding length
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = [
        seq[:max_length] + [pad_token_id] * (max_length - len(seq))
        for seq in sequences
    ]
    return padded

def batch_decode_parallel(tokenizer, batch_ids, pad_token_id=0,
                          num_workers=None, chunk_size=256):
    """
    Decode a batch of token sequences using core parallel decode_text_parallel.
    """
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer, None)
    ) as pool:
        texts = pool.map(
            partial(decode_text_parallel, tokenizer,
                    pad_token_id=pad_token_id,
                    num_workers=num_workers,
                    chunk_size=chunk_size),
            batch_ids
        )
    return texts
    