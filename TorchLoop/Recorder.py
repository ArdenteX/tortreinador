class Recorder:
    def __init__(self):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.sum = 0

    def update(self, val):
        self.val = val
        self.count += 1
        self.sum += val * 1
        self.avg = self.sum / self.count
