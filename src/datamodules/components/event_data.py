import torch
from numpy import split
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
        label_all_tokens=True,
        labels_to_ids=False,
        ids_to_labels=False
        ):
        # define the tokenizer
        self.label_all_tokens = label_all_tokens
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        self.tokenizer_kwargs = tokenizer_kwargs

        # Preserve the dataframe
        self.df = df
        
        # import the labels
        labels = df['labels'].apply(lambda x: x.split()).to_list()

        self.unique_labels = set()

        for label in labels:
            [self.unique_labels.add(i) for i in label if i not in self.unique_labels]

        # create a mapping from labels to ids if not provided
        if not labels_to_ids and not ids_to_labels:

            # Map each label into its id representation and vice versa
            self.labels_to_ids = {key: value for value, key in enumerate(sorted(self.unique_labels))}
            self.ids_to_labels = {value: key for value, key in enumerate(sorted(self.unique_labels))}

        # make sure we have not one, but both mappings
        elif not labels_to_ids or not ids_to_labels:
            raise ValueError("Both labels_to_ids and ids_to_labels must be provided.")

        # if we have both mappings, then we use them
        else:
            self.labels_to_ids = labels_to_ids
            self.ids_to_labels = ids_to_labels

        # generate the text and labels 
        self.text = df['event_names'].apply(lambda x: self.tokenizer(str(x), **self.tokenizer_kwargs)).to_list()

        for index, (text, label) in enumerate(zip(self.text, labels)):
            label_tensor = torch.tensor(self.align_label(text, label))
            # reshape the label tensor to match the shape of the text tensor
            self.text[index]['label'] = label_tensor[None, :]

        
    def __len__(self):
        return len(self.text)


    def __getitem__(self, idx):
        batch = self.get_batch_data(idx)

        return batch


    def get_batch_data(self, idx):
        return self.text[idx]


    def align_label(self, text, label):
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

    def split(self, split_ratio: float=0.8):
        """Splits the data into training and test/val sets.
        
        Args:
            split_ratio: ratio of the data to be used for training.
        
        Returns:
            training_data: training data.
            test: test/val data.
        """
        # split the data into training and test sets

        df_train, df_test = split(self.df, [int(split_ratio*len(self.df))])

        # create the training/test data split
        training_data = EventDataSequence(
            df_train, 
            labels_to_ids=self.labels_to_ids, 
            ids_to_labels=self.ids_to_labels,
            label_all_tokens=self.label_all_tokens,
            tokenizer_kwargs=self.tokenizer_kwargs
            )

        test_data = EventDataSequence(
            df_test, 
            labels_to_ids=self.labels_to_ids, 
            ids_to_labels=self.ids_to_labels,
            label_all_tokens=self.label_all_tokens,
            tokenizer_kwargs=self.tokenizer_kwargs
            )

        return training_data, test_data