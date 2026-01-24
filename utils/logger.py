import logging
import os
import time

class Logger:
    def __init__(self, log_dir='logs', log_file='train.log'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, log_file)
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)

    def log_epoch(self, epoch, loss, accuracy):
        message = f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}'
        self.log(message)

    def log_training_start(self):
        self.log('Training started.')

    def log_training_end(self):
        self.log('Training ended.')

    def log_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        self.log(f'Time taken: {elapsed_time:.2f} seconds')