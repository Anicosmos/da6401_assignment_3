import spacy
from datasets import load_dataset
from collections import Counter

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        rawdata = load_dataset('bentrevett/multi30k', split=self.split)
        self.train = rawdata['train']
        self.validation = rawdata['validation']
        self.test = rawdata['test']
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        # pass
        # Define special tokens
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        # raise NotImplementedError
        de_counter = Counter()
        en_counter = Counter()

        # Iterate through training data ONLY to build vocab 
        for example in self.train_data:
            de_counter.update(self.tokenize_de(example['de']))
            en_counter.update(self.tokenize_en(example['en']))

        # Create the mapping (Word -> Index)
        # Always put special tokens first so <pad> is index 0
        self.de_vocab = {token: i for i, token in enumerate(self.special_tokens)}
        self.en_vocab = {token: i for i, token in enumerate(self.special_tokens)}

        # Add words 
        for word, count in de_counter.items():
            if word not in self.de_vocab:
                self.de_vocab[word] = len(self.de_vocab)

        for word, count in en_counter.items():
            if word not in self.en_vocab:
                self.en_vocab[word] = len(self.en_vocab)
        
        # Create reverse mapping (Index -> Word) for decoding 
        self.de_inv_vocab = {i: token for token, i in self.de_vocab.items()}
        self.en_inv_vocab = {i: token for token, i in self.en_vocab.items()}
        
        print(f"German Vocab Size: {len(self.de_vocab)}")
        print(f"English Vocab Size: {len(self.en_vocab)}")


    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        # raise NotImplementedError
