import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from tokenizer import Tokenizer

# OK
def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder, max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    tokenizer = Tokenizer()
    word_map = tokenizer.get_word_map()

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        datas = json.load(j, encoding="utf-8")

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    word_freq = Counter()

    for data in datas:
        captions = []
        file_path = data.get("file_path", "")
        mode = file_path.split("/")[0][:-4]
        for caption in data.get("caption_ko", []):
            # Update word frequency
            tokens = tokenizer.tokenize(caption)
            word_freq.update(tokens)
            if len(tokens) <= max_len:
                captions.append(tokens)

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, file_path)

        if mode == "train":
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif mode == "val":
            val_image_paths.append(path)
            val_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions) #이미지 개수랑 캡션 개수 일치하는지 
    assert len(val_image_paths) == len(val_image_captions)


    # words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    # word_map = {k: v + 1 for v, k in enumerate(words)}
    # word_map['<unk>'] = len(word_map) + 1
    # word_map['<start>'] = len(word_map) + 1
    # word_map['<end>'] = len(word_map) + 1
    # word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []   # encoding된 caption 저장
            caplens = []        # caption의 길이 저장

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    # 임의의 (captions_per_image-len(captions[i])개 데이터 샘플링하여 추가 (데이터 수 늘리기)
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    # 임의의 captions_per_image 데이터를 샘플링
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                # 이미지의 크기 조절 (3(C), 256(W), 256(H))
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))  # image size 256*256으로 조절
                img = img.transpose(2, 0, 1)  # channel first
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                # 캡션 처리 ([START], [END], [PAD], [UNK] token 추가)
                for j, c in enumerate(captions):
                    # Encode captions
                    # caption 시작되기 전 [START] 넣어주고, 중간에는 본래 문장(이 때 대응되는 단어가 없다면 [UNK], 마지막에는 [END]
                    # 이 때, max_len보다 caption의 길이가 짧다면 [PAD] 추가
                    enc_c = [word_map['[CLS]']] + [word_map.get(word, word_map['[UNK]']) for word in c] + [
                        word_map['[SEP]']] + [word_map['[PAD]']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2  # <start>, <end> token 개수 추가해준 것 같다.

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


# # OK
def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    # torch.nn.init.uniform_(tensor, a, b)
    #   tensor: n-dimensional torch.Tensor
    #   a : lower bound of uniform distribution
    #   b : upper bound of uniform distribution

    # embeddings.size(1) = embedding size
    # embedding size 클수록 초기화 값 작아지도록 설정 (gradient exploding 방지 목적)
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


# OK
def load_embeddings(emb_file=None, word_map=None):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    tokenizer = Tokenizer()

    # Find embedding dimension
    pretrained_embedding = tokenizer.get_pretrained_embedding()
    emb_dim = tokenizer.get_embedding_dim()
    vocab = tokenizer.vocab.token_to_idx

    # Create tensor to hold embeddings, initialize
    # row -> vocab size
    # col -> embedding size
    embeddings = torch.FloatTensor(len(vocab), emb_dim)  # embedding matrix
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for i in range(pretrained_embedding.num_embeddings):
        vector = pretrained_embedding(torch.LongTensor([[i]]))
        embeddings[i] = torch.reshape(vector, (-1,))

    return embeddings, emb_dim


# OK
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    # gradient exploding 방지 목적으로 gradient clipping
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)   # min_norm, max_norm


# OK
def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    # save model checkpoint
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


# NOT OK
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # 여기서 n의 의미란??
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# OK
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """
    # learning rate 조절
    # learning_rate * shrink_factor
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


# NOT OK
def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
