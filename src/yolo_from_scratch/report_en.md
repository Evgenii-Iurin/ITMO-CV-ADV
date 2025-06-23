### YOLO From Scratch

- **Annotations**
    - Annotations were created using [CVAT](https://cvat.org/).
    - Every 10th frame was extracted from videos.
    - Total number of images: **182**
        - Training set: **121 images**
        - Validation set: **61 images**
    - Annotations follow the standard CVAT format (bounding boxes, class labels, etc.).

- Dataset
    - The image is divided into a **7×7 grid** (49 cells).
    - Object coordinates are returned **relative to the cell**.
    - Width and height are expressed in **cell units**, scaled by 7.
    - Images are resized to **448×448** using standard preprocessing.

- Model output format
    - The model outputs a tensor of shape `[batch_size, 7, 7, B * 5 + C]`, where:
        - `B = 2` — number of predicted boxes per cell
        - `C = 2` — class flags (double one-hot encoding)
    - Each cell prediction contains:

        ```
        [x1, y1, w1, h1, x2, y2, w2, h2, conf1, conf2, class_flag1, class_flag2]
        ```

- Loss function
    - For cells containing an object, the **responsible predictor** is determined using IoU.
    - The loss is computed only for the responsible prediction:
        - Coordinates (x, y)
        - Dimensions (w, h)
        - Confidence score
    - Handling negative width/height values:
        - Negative values under square roots can result in NaN.
        - Two strategies were considered:
            - Predict `sqrt(w)` and `sqrt(h)`, then square the result.
            - Use `sign(w) * sqrt(abs(w))` to preserve sign.

- Model architecture
    - Classic YOLO-style CNN:
        - 24 convolutional layers
        - Leaky ReLU activation

- Training
    - Optimizer: **SGD**
        - Learning rate: `0.001`
        - Weight decay: `5e-4`
        - Momentum: `0.9`
    - Hyperparameters and architecture follow the original YOLO paper.
