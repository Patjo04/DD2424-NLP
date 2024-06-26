from lex import Lexer

class DataSource:
    def __init__(self, path):
        self._path = path
        self._encoding = 'utf-8'

    def vocab(self) -> list[str]:
        return ['$'] + Lexer.VOCAB

    # Return a list of data (context, label):
    def labeled_samples_batch(self, batch_size: int, discard_trailing=False) -> any:
        batch_size = max(1, batch_size)
        features = []
        labels = []
        for feature, label in self.labeled_samples():
            features.append(feature)
            labels.append(label)
            if len(labels) == batch_size:
                yield features, labels
                features = []
                labels = []
        if len(labels) > 0 and not discard_trailing:
            yield features, labels

    def labeled_samples(self, ):
        with open(self._path, 'r', encoding=self._encoding) as f:
            for line in f.readlines():
                parts = line.split(',')
                feature = parts[0].split()
                label = parts[1].strip()
                yield feature, label


def batch(features, labels, batch_size, discard_trailing=False):
    batch_size = max(1, batch_size)
    batch_features = []
    batch_labels = []
    for i in range(len(labels)):
        feature = features[i]
        label = labels[i]
        batch_features.append(feature)
        batch_labels.append(label)
        if len(batch_labels) == batch_size:
            yield batch_features, batch_labels
            batch_features = []
            batch_labels = []
        if len(batch_labels) > 0 and not discard_trailing:
            yield batch_features, batch_labels

