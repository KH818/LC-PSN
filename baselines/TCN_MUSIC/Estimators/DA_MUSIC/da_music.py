"""Train and evaluate the learned MUSIC-style baseline regressors and classifiers."""

import sys
import os
import numpy as np
from utils.ask import ask
from utils.uniform_linear_array import UniformLinearArray
from tqdm import tqdm
import h5py
from utils.torch_dataset import TorchDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as tnf
from Estimators.DA_MUSIC.da_music_model import DeepAugmentedMUSICModel
from Estimators.DA_MUSIC.da_music_classifier_model import DAMUSICClassifierModel
from time import time
from time import sleep
from datetime import datetime
from utils.minimal_permutation_rmspe import MinimalPermutationRMSPE
from utils.awgn import awgn
import matplotlib.pyplot as plt


class DeepAugmentedMUSIC:
    def __init__(self,
                 angle_grids=np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False),
                 number_of_snapshots=200,
                 rx_ula=UniformLinearArray(8),
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        self.device = device
        self.angle_grids = angle_grids
        self.number_of_angles = angle_grids.shape[0]
        self.number_of_snapshots = number_of_snapshots
        self.rx_ula = rx_ula
        self.number_of_ula_elements = rx_ula.number_of_elements
        self.regressor_model = None
        self.classifier_model = None
        rmspe = MinimalPermutationRMSPE(device=device)
        self.rmspe = rmspe.loss
        self.name = self.system_name()

    def system_name(self, name=None):
        if name is None:
            self.name = "DA-MUSIC"
        else:
            self.name = name
        return self.name

    def generate_data(
            self,
            receiver,
            number_of_samples_per_snr,
            snr_db: list,
            number_of_sources: list,
            data_path=None,
    ):
        if os.path.exists(data_path):
            if ask('Data exists in path. Use/Generate'):
                file = h5py.File(data_path, 'r')
                data = np.array(file.get('data'))
                targets = np.array(file.get('targets'))
                file.close()
                return data, targets

        data = np.zeros((number_of_samples_per_snr * len(snr_db),
                         2 * self.number_of_ula_elements,
                         self.number_of_snapshots), np.float32)
        targets = np.zeros((number_of_samples_per_snr * len(snr_db),
                            self.number_of_ula_elements), np.float32)

        with tqdm(total=number_of_samples_per_snr * len(snr_db)) as progressbar:
            for snr_index in range(len(snr_db)):
                for sample_index in range(number_of_samples_per_snr):
                    number_of_sources_in_this_iteration = number_of_sources[sample_index % len(number_of_sources)]
                    angles = np.random.uniform(low=self.angle_grids[0],
                                               high=self.angle_grids[-1],
                                               size=number_of_sources_in_this_iteration)
                    snr_db_in_this_iteration = snr_db[snr_index]
                    # Noise is added inside receive() (before any 1-bit
                    # quantization) so the pipeline realizes z = Q1(A x + n).
                    noisy_received_signal = \
                        receiver.receive(angles,
                                         number_of_sources_in_this_iteration,
                                         snr_db=snr_db_in_this_iteration)
                    index = snr_index * number_of_samples_per_snr + sample_index
                    data[index, :self.number_of_ula_elements, :] = np.real(noisy_received_signal)
                    data[index, self.number_of_ula_elements:, :] = \
                        np.imag(noisy_received_signal)

                    targets[index, :] = \
                        np.pad(angles,
                               (0, self.number_of_ula_elements - number_of_sources_in_this_iteration),
                               'constant',
                               constant_values=np.pi)
                    progressbar.update()

        file = h5py.File(data_path, 'w')
        file.create_dataset('data', data=data)
        file.create_dataset('targets', data=targets)
        file.close()

        return data, targets

    def initiate_the_regressor(self):
        model = DeepAugmentedMUSICModel(self.angle_grids, self.rx_ula, self.device)
        return model

    def initiate_the_classifier(self):
        model = DAMUSICClassifierModel(self.regressor_model, self.device)
        return model

    def train_regressor(
            self,
            training_data,
            training_targets,
            validation_data,
            validation_targets,
            learning_rate=0.001,
            betas=(0.9, 0.999),
            batch_size=64,
            number_of_epochs=200,
            model_path=None,
    ):
        if os.path.exists(model_path):
            if not ask('Model already exists in the path. Do you like to overwrite?'):
                sys.exit()

        print("regressor training...\n")

        train_dataset = TorchDataset(training_data, training_targets)
        validation_dataset = TorchDataset(validation_data, validation_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        model = self.initiate_the_regressor()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)

        training_loss_history = np.zeros(number_of_epochs, np.float32)
        validation_loss_history = np.zeros(number_of_epochs, np.float32)
        start_time_of_training = time()
        print("Training started at:", datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))
        sleep(0.1)
        for epoch in range(number_of_epochs):
            training_loss = 0.0
            validation_loss = 0.0
            model.train()
            for input_batch, target_batch in tqdm(train_loader):
                optimizer.zero_grad()
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                estimations_batch, _ = model(input_batch)
                batch_loss = self.rmspe(estimations_batch, target_batch)
                batch_loss.backward()
                optimizer.step()
                training_loss += batch_loss.item()

            training_loss = training_loss / len(train_loader)
            training_loss_history[epoch] = training_loss

            model.eval()
            for input_batch, target_batch in validation_loader:
                with torch.no_grad():
                    input_batch = input_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                    estimations_batch, _ = model(input_batch)
                    batch_loss = self.rmspe(estimations_batch, target_batch)
                    validation_loss += batch_loss.item()

            validation_loss = validation_loss / len(validation_loader)
            validation_loss_history[epoch] = validation_loss

            print(f'|epoch={epoch + 1:3d}/{number_of_epochs:3d}'
                  f'|total_time={(time() - start_time_of_training) / 60:.2f}m'
                  f'|training_loss={training_loss:.4f}'
                  f'|validation_loss={validation_loss:.4f}|')
            sleep(0.1)

        print("Training ended at:", datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))
        torch.save(model.state_dict(), model_path)
        return training_loss_history, validation_loss_history

    def train_classifier(
            self,
            training_data,
            training_targets,
            validation_data,
            validation_targets,
            learning_rate=0.001,
            betas=(0.9, 0.999),
            batch_size=50,
            number_of_epochs=200,
            model_path=None,
    ):
        if os.path.exists(model_path):
            if not ask('Model already exists in the path. Do you like to overwrite?'):
                sys.exit()

        print("classifier training...\n")

        train_dataset = TorchDataset(training_data, training_targets)
        validation_dataset = TorchDataset(validation_data, validation_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        model = self.initiate_the_classifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)

        training_loss_history = np.zeros(number_of_epochs, np.float32)
        validation_loss_history = np.zeros(number_of_epochs, np.float32)
        training_accuracy_history = np.zeros(number_of_epochs, np.float32)
        validation_accuracy_history = np.zeros(number_of_epochs, np.float32)
        start_time_of_training = time()
        print("Training started at:", datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))
        sleep(0.1)
        for epoch in range(number_of_epochs):
            training_loss = 0.0
            validation_loss = 0.0
            training_accuracy = 0.0
            validation_accuracy = 0.0
            model.train()
            for input_batch, target_batch in tqdm(train_loader):
                optimizer.zero_grad()
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                estimations_batch = model(input_batch)
                batch_loss = self.loss_number_of_sources_estimation(estimations_batch, target_batch)
                batch_loss.backward()
                optimizer.step()
                training_loss += batch_loss.item()

                probabilities_batch = tnf.softmax(estimations_batch, dim=1)
                true_number_of_sources_batch = torch.argmax(target_batch, dim=1)
                estimated_number_of_sources_batch = torch.argmax(probabilities_batch, dim=1)
                estimated_number_of_sources_batch += 2
                batch_accuracy = \
                    (true_number_of_sources_batch == estimated_number_of_sources_batch).float().mean().item()

                training_accuracy += batch_accuracy

            training_loss = training_loss / len(train_loader)
            training_loss_history[epoch] = training_loss
            training_accuracy = training_accuracy / len(train_loader)
            training_accuracy_history[epoch] = training_accuracy

            model.eval()
            for input_batch, target_batch in validation_loader:
                with torch.no_grad():
                    input_batch = input_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                    estimations_batch = model(input_batch)
                    batch_loss = self.loss_number_of_sources_estimation(estimations_batch, target_batch)
                    validation_loss += batch_loss.item()

                    probabilities_batch = tnf.softmax(estimations_batch, dim=1)
                    true_number_of_sources_batch = torch.argmax(target_batch, dim=1)
                    estimated_number_of_sources_batch = torch.argmax(probabilities_batch, dim=1)
                    estimated_number_of_sources_batch += 2
                    batch_accuracy = \
                        (true_number_of_sources_batch == estimated_number_of_sources_batch).float().mean().item()

                    validation_accuracy += batch_accuracy

            validation_loss = validation_loss / len(validation_loader)
            validation_loss_history[epoch] = validation_loss
            validation_accuracy = validation_accuracy / len(validation_loader)
            validation_accuracy_history[epoch] = validation_accuracy

            print(f'|epoch={epoch + 1:3d}/{number_of_epochs:3d}'
                  f'|total_time={(time() - start_time_of_training) / 60:.2f}m'
                  f'|training_loss={training_loss:.4f}'
                  f'|validation_loss={validation_loss:.4f}\n'
                  f'|training_accuracy={training_accuracy:.4f}|validation_accuracy={validation_accuracy:.4f}|')
            sleep(0.1)

        print("Training ended at:", datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))
        torch.save(model.state_dict(), model_path)
        return training_loss_history, training_accuracy_history, validation_loss_history, validation_accuracy_history

    def plot(self, training, validation, ylabel):
        plt.figure()
        plt.plot(training, label='Training')
        plt.plot(validation, label='Validation')
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(self.name)
        plt.legend()
        plt.show()

    def loss_number_of_sources_estimation(self, estimations, true):
        batch_size = true.shape[0]
        true_number_of_sources = torch.argmax(true, dim=1)

        criterion = torch.nn.CrossEntropyLoss()

        one_hot_encoded_true_number_of_sources = torch.zeros(batch_size, 4).to(self.device)

        for batch_index in range(batch_size):
            one_hot_encoded_true_number_of_sources[batch_index][true_number_of_sources[batch_index] - 2] = 1

        return criterion(estimations, one_hot_encoded_true_number_of_sources)

    def load_regressor_model(self,
                             model_path):
        if not os.path.exists(model_path):
            print('There is no model in the path.')
            sys.exit()

        self.regressor_model = self.initiate_the_regressor().to(self.device)
        self.regressor_model.load_state_dict(torch.load(model_path, weights_only=True))
        self.regressor_model.eval()

    def load_classifier_model(self,
                              model_path):
        if not os.path.exists(model_path):
            print('There is no model in the path.')
            sys.exit()

        self.classifier_model = self.initiate_the_classifier().to(self.device)
        self.classifier_model.load_state_dict(torch.load(model_path, weights_only=True))
        self.classifier_model.eval()

    def estimate(self, received_signal, number_of_sources=None):
        if self.regressor_model is None:
            print('Model is not loaded. Load the model first.')
            sys.exit()

        with torch.no_grad():
            received_signal = torch.from_numpy(received_signal)
            received_signal = torch.cat([received_signal.real.float(),
                                         received_signal.imag.float()]).to(self.device)
            received_signal = received_signal.unsqueeze(dim=0).to(self.device)
            estimation, feature_for_number_of_sources_estimation = \
                self.regressor_model(received_signal)
            torch.cuda.synchronize()
            estimation = estimation.cpu().numpy()[0, :number_of_sources]
            if number_of_sources is None:
                number_of_sources_probabilities = tnf.softmax(self.classifier_model(received_signal), dim=1)
                number_of_sources = torch.argmax(number_of_sources_probabilities, dim=1)
                number_of_sources = number_of_sources.cpu().numpy() + 2
        return estimation, number_of_sources
