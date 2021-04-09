from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

class Tokenizer:
    model, vocab = get_pytorch_kobert_model()
    tok_path = get_tokenizer()
    sp = SentencepieceTokenizer(tok_path)

    def __init__(self):
        self.vocab.token_to_idx['[SOS]'] = len(self.vocab)
        self.vocab.token_to_idx['[EOS]'] = len(self.vocab)+1
        self.vocab.idx_to_token.extend(['[SOS]', '[EOS]'])

    def tokenize(self, sentence: str):
        return self.sp(sentence)

    def tok2idx(self, tokens: list):
        return [self.vocab.token_to_idx[token] for token in tokens]

    def get_word_map(self):
        return self.vocab.token_to_idx

    def get_embedding_dim(self):
        return list(self.model.embeddings.children())[0].embedding_dim  # 768

    def get_pretrained_embedding(self):
        return self.model.embeddings.word_embeddings
