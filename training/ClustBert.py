import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler, DataCollatorWithPadding
from transformers.modeling_outputs import TokenClassifierOutput
from training import DataSetUtils


class ClustBERT(nn.Module):

    def __init__(self, k: int):
        super(ClustBERT, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, k)  # load and initialize weights
        self.clustering = KMeans(k)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

    def cluster_and_generate(self, data: Dataset) -> Dataset:
        print("Start Step 1 --- Clustering")

        test_data = data["new_sentence"]
        sentence_embedding = clust_bert.get_sentence_vectors_with_token_average(test_data)
        X = [sentence.cpu().detach().numpy() for sentence in sentence_embedding]
        pseudo_labels = self.clustering.fit_predict(X)

        data = data.remove_columns(["label"])
        data = data.map(lambda example, idx: {"label": pseudo_labels[idx]}, with_indices=True)

        print("Finished Step 1 --- Clustering " + str(data))
        return data

    def get_sentence_vectors_with_token_average(self, texts: list):
        return [self.get_sentence_vector_with_token_average(text) for text in texts]

    def get_sentence_vector_with_token_average(self, text):
        tokens = torch.LongTensor(self.tokenizer.encode(text))
        tokens = tokens.to(device)
        tokens = tokens.unsqueeze(0)

        with torch.no_grad():
            out = clust_bert.model(input_ids=tokens)

        # we only want the hidden_states
        hidden_states = out[2]
        return torch.mean(hidden_states[-1], dim=1).squeeze()


snli = DataSetUtils.get_snli_dataset()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
clust_bert = ClustBERT(3).to(device)
clust_bert.model.eval()

dataset = clust_bert.cluster_and_generate(snli)

data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)
train_dataloader = DataLoader(
    dataset, shuffle=True, batch_size=8, collate_fn=data_collator
)

optimizer = AdamW(clust_bert.model.parameters(), lr=3e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

clust_bert.model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = clust_bert.model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
