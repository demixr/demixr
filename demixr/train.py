from tqdm import tqdm
import argparse
import matplotlib as plt
import numpy as np
import metrics
import utils
from dataset import create_dataloaders
from pathlib import Path


import torch
from torchmetrics.functional import si_sdr
import torch.optim as optim


class Trainer:
    def __init__(self, model, lr, loss=torch.nn.MSELoss, stream="vocals"):
        self.model = model
        self.lr = lr
        self.stream = stream
        self.criterion = loss
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.4, patience=2, cooldown=2
        )
        self.history = {"lr": [], "loss": [], "val_loss": [], "sdr": [], "val_sdr": []}
        self.max_sdr = float("-inf")

    def fit(self, train_dataloader, val_dataloader, nb_epochs):
        for epoch in range(nb_epochs):
            train_loss = val_loss = train_sdr = val_sdr = 0.0
            print(f"\nEpoch {epoch + 1}/{nb_epochs}")

            nb_batches = len(train_dataloader)

            tqloader = tqdm(train_dataloader, total=len(train_dataloader))

            # Turn on dropout, batch norm etc..
            self.model.train()

            for i, batch in enumerate(tqloader):
                inputs, labels = batch[0], batch[1]

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                train_loss += loss.detach().item()

                train_sdr += si_sdr(outputs, labels)

                loss.backward()
                self.optimizer.step()

                tqloader.set_postfix(
                    loss=train_loss / (i + 1),
                    sdr=train_sdr[0].item() / (i + 1),
                    lr=self.scheduler.optimizer.param_groups[0]["lr"],
                )

                self.model.eval()

                with torch.no_grad():
                    for inputs, labels in val_dataloader:
                        inputs, labels = inputs.float(), labels.float()

                        outputs = self.model(inputs)

                        loss = self.criterion(outputs, labels)
                        val_loss += loss.detach().item()

                        sdr = metrics.SDR()
                        val_sdr += sdr(outputs, labels)

                train_sdr = torch.mean(train_sdr / (len(train_dataloader)))
                val_sdr = torch.mean(val_sdr / (len(val_dataloader)))
                train_loss = train_loss / len(train_dataloader)
                val_loss = val_loss / len(val_dataloader)
                lr = self.scheduler.optimizer.param_groups[0]["lr"]

                self.__save_history(lr, train_loss, val_loss, train_sdr, val_sdr)

                print(
                    f"Epoch {epoch + 1}, "
                    f"loss = {train_loss:.3f}, "
                    f"sdr = {train_sdr:.3f}, "
                    f"val_loss = {val_loss:.3f}, "
                    f"val_sdr = {val_sdr:.3f}"
                )

                if val_sdr > self.max_sdr:
                    print(
                        f"Model saved. SDR updated: {self.max_val_sdr:.3f} -> {val_sdr:.3f}\n"
                    )
                    self.max_val_sdr = val_sdr
                    torch.save(self.model.state_dict(), "demixr_sdr.pt")

                self.scheduler.step(val_sdr)

    def evaluate(self, test_dataloader):
        torch.cuda.empty_cache()
        total_sdr = []

        acc_loss = acc_sdr = 0.0

        tqloader = tqdm(test_dataloader, total=len(test_dataloader))

        with torch.no_grad():
            for i, (inputs, labels, crop_indexes) in enumerate(tqloader):
                inputs, labels = inputs.to(self.device), labels.to(self.output_device)

                pred = self.model(inputs, pad_dimensions=crop_indexes)
                loss = self.criterion(pred, labels)

                labels = labels.detach().cpu().numpy()

                pred = pred.detach().cpu().numpy()
                acc_loss += loss.detach().item()
                result = pred.astype(np.uint)

                total_sdr.insert(0, si_sdr(result, labels))
                acc_sdr += total_sdr[0]

                tqloader.set_postfix(
                    loss=acc_loss / (i + 1),
                    sdr=acc_sdr[0].item() / (i + 1),
                    lr=self.scheduler.optimizer.param_groups[0]["lr"],
                )

        total_sdr = [input[0] for input in total_sdr]
        plt.boxplot(total_sdr, showmeans=True)
        plt.show()

        mean_loss, mean_sdr = [
            val / len(test_dataloader) for val in [acc_loss, acc_sdr]
        ]
        return mean_loss, mean_sdr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="data")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    dataset_paths = utils.get_dataset_paths(args.dataset)
    train_dl, valid_dl, test_dl = create_dataloaders(dataset_paths, args.batch_size)


if __name__ == "__main__":
    main()
