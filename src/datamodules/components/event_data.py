import torch
from transformers import BertTokenizerFast
class EventDataSequence(torch.utils.data.Dataset):
    def __init__(
        self, 
        df, 
        tokenizer_kwargs={
            'padding':'max_length', 
            'max_length':30, 
            'truncation':True, 
            'return_tensors':"pt"
            },
        label_all_tokens=True
        ):
        # define the tokenizer
        self.label_all_tokens = label_all_tokens
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        
        # import the data
        label = df['labels'].apply(lambda x: x.split()).to_list()
        texts  = df['event_names'].values.tolist()

        # Create a column for the tokenized inputs 
        self.text = df['event_names'].apply(lambda x: self.tokenizer(str(x), **tokenizer_kwargs)).to_list()
        self.labels = [self.align_label(i,j) for i,j in zip(self.text, label)]
        
        # create a mapping from labels to ids
        self.unique_labels = set()

        for label in self.labels:
            [self.unique_labels.add(i) for i in label if i not in self.unique_labels]
        
        # Map each label into its id representation and vice versa
        self.labels_to_ids = {key: value for value, key in enumerate(sorted(self.unique_labels))}
        self.ids_to_labels = {value: key for value, key in enumerate(sorted(self.unique_labels))}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels

    def get_batch_data(self, idx):
        return self.text[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def align_label(self, text, label, max_length=30):
        """Aligns the label to the tokenized input text. 
        This is necessary because the tokenization will change the length of the label.
        
        Args:
            texts: tokenized text.
            labels: label.
        
        Returns:
            Length matched label ids.
        """
        
        word_ids = text.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # if label_all_tokens is True, then we add all tokens of the label are labeled with the same ID.
            # if label_all_tokens is False, then we add only the first token of the label is labeled with the same ID.
            # All other values are set to -100.
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(self.labels_to_ids[label[word_idx]])
                except:
                    label_ids.append(-100)
            else:
                try:
                    label_ids.append(self.labels_to_ids[label[word_idx]] if self.label_all_tokens else -100)
                except:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        return label_ids
