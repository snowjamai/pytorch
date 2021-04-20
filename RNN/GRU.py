import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import datasets, data

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 40
DEVICE = torch.device("cpu")

TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

tr_set, test_set = datasets.IMDB.splits(TEXT, LABEL)

print(len(tr_set))

TEXT.build_vocab(tr_set, min_freq=5)
LABEL.build_vocab(tr_set)

tr_set, val_set = tr_set.split(split_ratio=0.8)

tr_iter, val_iter, test_iter = data.BucketIterator.splits((tr_set, val_set, test_set),
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=True, repeat=False)

vocab_size = len(TEXT.vocab)
n_classes = 2

print("%d, %d, %d, %d" % (len(tr_set), len(val_set), vocab_size, n_classes))

class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicGRU, self).__init__()
        print("Building Basic GRU model ...")

        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.gru(x, h_0)
        h_t = x[:, -1, :]
        self.dropout(h_t)
        logit = self.out(h_t)

        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data

        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

def train(model, optimizer, train_iter):
    model.train()

    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0

    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size

    return avg_loss, avg_accuracy

model = BasicGRU(1,256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = None

for e in range(1, EPOCHS+1):
    train(model, optimizer, tr_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("에포크 : %d val_loss : %5.2f : val_accuracy : %5.2f" % (e, val_loss, val_accuracy))

    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('test loss : %5.2f test accuracy : %5.2f' % (test_loss, test_acc))



