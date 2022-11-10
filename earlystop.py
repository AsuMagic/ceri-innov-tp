class EarlyStopping:
    def __init__(self, patience, acc_threshold):
        self.patience = patience
        self.acc_threshold = acc_threshold

        self.best_accuracy = 0
        self.stalled_for = 0

    def update(self, accuracy):
        if accuracy > self.best_accuracy + self.acc_threshold:
            self.best_accuracy = accuracy

            if self.stalled_for != 0:
                print(f"Observed an improvement in accuracy. The stall counter was reset to 0.")
                self.stalled_for = 0
        else:
            self.stalled_for += 1
            print(f"Dev accuracy has stalled for {self.stalled_for} epochs; exhausted in {self.patience - self.stalled_for}")

    def should_stop(self):
        return self.stalled_for >= self.patience
