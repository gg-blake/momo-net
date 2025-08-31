import unittest
from tokenizer import Tokenizer

from tokenizer_ext import (
    encode_text_parallel,
    decode_text_parallel,
    batch_encode_parallel,
    batch_decode_parallel,
)


class TestParallelTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()
        self.pad_token_id = 0

    def test_encode_decode_single(self):
        text = "Hello World"
        encoded = encode_text_parallel(self.tokenizer, text, chunk_size=3)
        decoded = decode_text_parallel(self.tokenizer, encoded, pad_token_id=self.pad_token_id, chunk_size=3)

        self.assertEqual(decoded, text)
        self.assertTrue(all(isinstance(x, int) for x in encoded))

    def test_batch_encode_padding(self):
        texts = ["Hi", "Hello"]
        encoded_batch = batch_encode_parallel(
            self.tokenizer, texts, pad_token_id=self.pad_token_id, chunk_size=2
        )

        # all sequences should have equal length
        lengths = {len(seq) for seq in encoded_batch}
        self.assertEqual(len(lengths), 1)

        # check padding applied correctly
        max_len = max(len(t) for t in texts)
        self.assertEqual(len(encoded_batch[0]), max_len)

    def test_batch_decode(self):
        texts = ["abc", "de"]
        encoded_batch = batch_encode_parallel(
            self.tokenizer, texts, pad_token_id=self.pad_token_id, chunk_size=2
        )
        decoded_batch = batch_decode_parallel(
            self.tokenizer, encoded_batch, pad_token_id=self.pad_token_id, chunk_size=2
        )

        self.assertEqual(decoded_batch, texts)

    def test_padding_is_ignored_on_decode(self):
        text = "padtest"
        encoded = self.tokenizer.encode(text)
        padded = encoded + [self.pad_token_id] * 5

        decoded = decode_text_parallel(
            self.tokenizer, padded, pad_token_id=self.pad_token_id, chunk_size=3
        )
        self.assertEqual(decoded, text)
        
if __name__ == "__main__":
    unittest.main()
