import torch
import logging
import optuna
import torchvision.transforms as T
from torch.utils.data import DataLoader
from src.hw_yolo.dataset import CustomDataset
from loss import yolo_loss
from model import YOLOv1
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(device)

transform = T.Compose(
    [
        T.Resize((448, 448)),
        T.ToTensor(),
    ]
)


def objective(trial):
    # Let Optuna suggest values
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.7, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_dataset = CustomDataset(
        root_dir="src/hw_yolo/dataset/train", transform=transform
    )
    val_dataset = CustomDataset(root_dir="src/hw_yolo/dataset/val", transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = YOLOv1()
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    writer = SummaryWriter(
        log_dir=f"runs/yolov1_lr{learning_rate}_bs{batch_size}_mom{momentum}_wdecay{weight_decay}"
    )

    model.train_model(
        train_dataloader,
        yolo_loss,
        optimizer,
        device=device,
        epochs=2,
        validation_dataloader=val_dataloader,
        validate_every=1,
        writer=writer,
    )

    writer.close()

    final_val_loss = model.last_validation_loss
    logging.info("Final validation loss: %s", final_val_loss)

    return final_val_loss


# Run the optimization
if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize"
    )  # we want to minimize validation loss
    study.optimize(objective, n_trials=20)  # Try 20 different combinations

    logging.info("Best hyperparameters: %s", study.best_params)
