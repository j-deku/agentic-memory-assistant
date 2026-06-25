import torch
from tasks import list_tasks
from learning_logger import get_events  # assuming you have it

def build_sequences(window_size=10):
    tasks = list_tasks()
    events = get_events()

    sequences = []
    labels = []

    for i in range(len(tasks)):

        window = tasks[max(0, i-window_size): i]

        seq_features = []

        for t in window:
            seq_features.append([
                t[4],  # completed
                len(t[1]),  # task length (proxy complexity)
                hash(t[2]) % 100,  # category encoded
                i  # time step proxy
            ])

        if len(seq_features) < window_size:
            continue

        sequences.append(seq_features)
        labels.append(tasks[i][4])  # completion prediction target

    X = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y