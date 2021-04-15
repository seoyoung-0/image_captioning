import torch
from torch import nn
import torchvision
from tokenizer import Tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model tokenizer (KOBERT)
tokenizer = Tokenizer()

# Load pre-trained model (weights) (KOBERT)
BertModel, _ = get_pytorch_kobert_model()
BertModel.eval()

PAD_INDEX = 1

'''
    - resnet에서 뒤의 2개의 레이어 제거한 후, fine-tuning 진행하는 코드
    - fine-tuning 은 뒤의 5개 레이어에 대해서만 수행
    - Encoder 의 최종 output shape = (batch_size, encoded_image_size, encoded_image_size, encoder_dim)
    - 여기서 우리는 뒤에 몇 개의 레이어를 추가하는 식으로 개선할 수 있을 것 같다 ...
'''

from torch import nn
import torchvision
class Encoder(nn.Module):

  def __init__(self,encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

  def forward(self, images):
        """
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)        # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)    # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0,2,3,1)       # (batch_size, encoded_image_size, encoded_image_size, 2048)
        
        print('출력해보기_image_feature',out)

        return out
  def fine_tune(self, fine_tune=True):
    """
     Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

     :param fine_tune: Allow?
    """
    # parameter update되지 않도록 고정
    for p in self.resnet.parameters():
      p.requires_grad = False
    # If fine-tuning, only fine-tune convolutional blocks 2 through 4
    for c in list(self.resnet.children())[5:]:
      for p in c.parameters():
        p.requires_grad = fine_tune 



'''
    - encoder_out (batch_size, num_pixels, encoder_dim) -> FC -> att1 (batch_size, num_pixels, attention_dim)
    - decoder_hidden (batch_size, decoder_dim) -> FC -> att2 (batch_size, attention_dim)
    - att : encoder_out + decoder_hidden.unsqueeze(1) -> FC -> relu -> resize (batch_size, num_pixels)
    - alpha : softmax(att) (batch_size, num_pixels)
    - attention_weighted_encoding : encoder_out * alpha.unsqueeze(2) -> sum(axis=1)
    
'''
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # encoder_out : (batch_size, num_pixels, encoder_dim)
        # decoder_hideen: (batch_size, decoder_dim)

        att1 = self.encoder_att(encoder_out)        # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)     # (batch_size, attention_dim)

        # att2.unsqueeze(1) -> (batch_size, 1, attention_dim)
        # att1 + att2.unsqueeze(1) -> (batch_size, num_pixels, attention_dim)
        # self.full_att(att1+att2.unsqueeze(1)) -> (batch_size, num_pixels, 1)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels) <- 여기가 이해되지 X
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        # encoder_out : (batch_size, num_pixels, encoder_dim)
        # alpha.unsqueeze(2) : (batch_size, num_pixels, 1)
        # encoder_out*alpha.unsqueeze(2) -> (batch_size, num_pixels, encoder_dim)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size=7004, encoder_dim=2048, dropout=0.5, use_bert=True):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.use_bert = use_bert

        if self.use_bert:
            self.embed_dim = 768
            self.vocab_size = tokenizer.get_vocab_size()  # 7004

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)  # dropout layer
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # encoder_out -> (batch_size, encoded_image_size, encoded_image_size, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        max_dec_len = caption_lengths.reshape(-1).max().item()

        # Sort input data by decreasing lengths; why? apparent below
        # caption_lengths -> (batch_size, 1)
        # caption_lengths.squeeze(1) -> (batch_size)
        # caption length 별 내림차순 정렬
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        if not self.use_bert:
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        else:
            embeddings = []
            for cap_idx in encoded_captions:
                while len(cap_idx) < max_dec_len:
                    cap_idx.append(PAD_INDEX)  # PAD (1)

                cap = " ".join(tokenizer.convert_ids_to_tokens(cap_idx))

                tokenized_cap = tokenizer.convert_ids_to_tokens(cap_idx)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_cap)
                tokens_tensor = torch.tensor([indexed_tokens])

                bert_embedding, _ = BertModel(tokens_tensor)
                bert_embedding = bert_embedding.squeeze(0)
                embeddings.append(bert_embedding)

            embeddings = torch.stack(embeddings).to(device)

        # Initialize LSTM state
        # encoder_out : (batch_size, num_pixels, encoder_dim)
        # h : (batch_size, encoder_dim) - hidden state
        # c : (batch_size, encoder_dim) - cell state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        # caption_lengths : (batch_size)
        # caption_lengths -1 : 모든 caption의 길이에서 1 빼준다 (<end> token 포함하지 않기 위해서)
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])  # 학습할 데이터의 수

            # encoder output, current hidden state -> attention -> weighted_encoding, alphs
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            # hidden_state -> fully connected -> sigmoid -> gate
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)

            # hadamard product (gate and attention_weighted_encoding)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
            # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            # input
            #   1. embedding과 attention_weighted_encoding concatenate -> input
            #   2. h -> hidden state
            #   3. c -> cell state
            # output
            #   1. output
            #   2. hidden state
            #   3. cell state
            # 여기서 왜 embedding과 attention_weighted_encoding 을 합치는거야??
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            # h -> dropout -> fully connected -> preds
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

            # update predictions and alphas(=attention score)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
