import numpy as np

class DataSource:
    def __init__(self, data_file, pad_idx):
        self.start = 0
        self.dataset = {}
        self.pad_idx = pad_idx
        with open(data_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip("\n").split() for line in lines]
            lines = [np.array([int(x) for x in line]) for line in lines]
            targets = [np.append(line[1:], pad_idx) for line in lines]

            self.dataset["input"] = lines
            self.dataset["target"] = targets

        self.size = len(self.dataset["input"])

    #def next_train_batch(self, batch_size):
    #    batch_inputs = []
    #    batch_targets = []
    #    for i in range(batch_size):
    #        line = self.f.readline()
    #        if len(line) == 0:
    #            f.seek(0, 0)
    #            line = f.readline()

    #        line = line.strip("\n").split()
    #        inputs = np.array([int(x) for x in line])
    #        target = np.append(inputs[1:], self.pad_idx)

    #        batch_inputs.append(inputs)
    #        batch_targets.append(target)

    #    return batch_inputs, batch_targets

    def next_train_batch(self, batch_size):
        batch_inputs = []
        batch_targets = []
        end = self.start + batch_size
        batch_inputs = self.dataset["input"][self.start:end]
        batch_targets = self.dataset["target"][self.start:end]

        self.start = end

        if (len(batch_inputs) < batch_size):
            rest = batch_size - len(batch_inputs)
            batch_inputs += self.dataset["input"][:rest]
            batch_targets += self.dataset["target"][:rest]
            self.start = rest

        return batch_inputs, batch_targets
