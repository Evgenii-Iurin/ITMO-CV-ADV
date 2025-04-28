import torch
import logging
import torchvision.transforms as T
from torch.utils.data import DataLoader
from src.hw_yolo.dataset import CustomDataset
from loss import yolo_loss
from model import YOLOv1

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

EPOCHS = 10
BATCH_SIZE = 32
MOMENTUM = 0.9
DECAY = 5e-4
LEARNING_RATE = 1e-3


device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(device)

transform = T.Compose(
    [
        T.Resize((448, 448)),
        T.ToTensor(),
    ]
)

if __name__ == "__main__":
    train_dataset = CustomDataset(
        root_dir="src/hw_yolo/dataset/train", transform=transform
    )
    val_dataset = CustomDataset(root_dir="src/hw_yolo/dataset/val", transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sample_img, sample_label = train_dataset[1]

    logging.info("Total number of images in the dataset: %d", len(train_dataset))
    logging.info("Image size after transformation: %s", sample_img.shape)
    logging.info("Label matrix shape: %s", sample_label.shape)

    model = YOLOv1()

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=DECAY
    )

    model.train_model(
        train_dataloader,
        yolo_loss,
        optimizer,
        device=device,
        epochs=EPOCHS,
        validation_dataloader=val_dataloader,
        validate_every=1,
    )

    # torch.save(model.state_dict(), "yolo_trained.pt")
