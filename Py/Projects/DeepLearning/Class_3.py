import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# -----------------------------
# 基础工具
# -----------------------------
def set_seed(seed: int = 42) -> None:
    """设置随机种子，尽量保证结果可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# 合成检测数据集
# -----------------------------
def make_detection_dataset(
    num_samples: int = 4200,
    image_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构造单目标检测数据集：
    - 类别0：方块（square）
    - 类别1：横向矩形（h_rect）
    - 类别2：纵向矩形（v_rect）

    返回：
    - images: [N, 1, H, W]
    - labels_cls: [N]
    - labels_box: [N, 4]，归一化格式 (cx, cy, w, h)
    """

    images = torch.zeros(num_samples, 1, image_size, image_size, dtype=torch.float32)
    labels_cls = torch.zeros(num_samples, dtype=torch.long)
    labels_box = torch.zeros(num_samples, 4, dtype=torch.float32)

    for i in range(num_samples):
        cls = random.randint(0, 2)

        # 不同类别使用不同的宽高先验
        if cls == 0:
            bw = random.randint(7, 11)
            bh = bw
        elif cls == 1:
            bw = random.randint(10, 14)
            bh = random.randint(5, 8)
        else:
            bw = random.randint(5, 8)
            bh = random.randint(10, 14)

        x0 = random.randint(0, image_size - bw - 1)
        y0 = random.randint(0, image_size - bh - 1)
        x1 = x0 + bw
        y1 = y0 + bh

        # 背景噪声 + 亮目标
        img = 0.05 * torch.randn(image_size, image_size)
        img = torch.clamp(img, 0.0, 1.0)
        img[y0:y1, x0:x1] = 1.0

        # 归一化边界框标签
        cx = (x0 + x1) / 2.0 / image_size
        cy = (y0 + y1) / 2.0 / image_size
        w = (x1 - x0) / image_size
        h = (y1 - y0) / image_size

        images[i, 0] = img
        labels_cls[i] = cls
        labels_box[i] = torch.tensor([cx, cy, w, h], dtype=torch.float32)

    return images, labels_cls, labels_box


# -----------------------------
# CNN 检测器
# -----------------------------
class TinyDetectorCNN(nn.Module):
    """
    轻量 CNN 检测器（双头输出）：
    - 分类头：输出类别 logits
    - 回归头：输出边界框 (cx, cy, w, h)
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.cls_head = nn.Linear(128, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(128, 4),
            nn.Sigmoid(),  # 将边界框限制在 [0, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        feat = self.neck(feat)
        logits = self.cls_head(feat)
        boxes = self.box_head(feat)
        return logits, boxes


# -----------------------------
# 指标与训练
# -----------------------------
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """将 (cx, cy, w, h) 转成 (x1, y1, x2, y2)，坐标为归一化值。"""
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def batch_iou(pred_box: torch.Tensor, true_box: torch.Tensor) -> torch.Tensor:
    """计算一个 batch 中每个样本的 IoU。"""
    p = cxcywh_to_xyxy(pred_box)
    t = cxcywh_to_xyxy(true_box)

    inter_x1 = torch.maximum(p[:, 0], t[:, 0])
    inter_y1 = torch.maximum(p[:, 1], t[:, 1])
    inter_x2 = torch.minimum(p[:, 2], t[:, 2])
    inter_y2 = torch.minimum(p[:, 3], t[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area_p = (p[:, 2] - p[:, 0]).clamp(min=0.0) * (p[:, 3] - p[:, 1]).clamp(min=0.0)
    area_t = (t[:, 2] - t[:, 0]).clamp(min=0.0) * (t[:, 3] - t[:, 1]).clamp(min=0.0)
    union = area_p + area_t - inter_area + 1e-8
    return inter_area / union


def evaluate_detector(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_box: float = 3.0,
) -> tuple[float, float, float, float]:
    """返回：总损失、分类准确率、边框 MAE、平均 IoU。"""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_mae = 0.0
    total_iou = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, y_cls, y_box in loader:
            xb = xb.to(device)
            y_cls = y_cls.to(device)
            y_box = y_box.to(device)

            logits, pred_box = model(xb)
            loss_cls = F.cross_entropy(logits, y_cls)
            loss_box = F.smooth_l1_loss(pred_box, y_box)
            loss = loss_cls + lambda_box * loss_box

            pred_cls = logits.argmax(dim=1)
            total_correct += (pred_cls == y_cls).sum().item()
            total_mae += F.l1_loss(pred_box, y_box, reduction="sum").item()
            total_iou += batch_iou(pred_box, y_box).sum().item()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_n += bs

    avg_loss = total_loss / total_n
    cls_acc = total_correct / total_n
    box_mae = total_mae / (total_n * 4)
    mean_iou = total_iou / total_n
    return avg_loss, cls_acc, box_mae, mean_iou


def train_detector(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    lambda_box: float = 3.0,
) -> tuple[float, float, float]:
    """训练检测器，并返回最佳验证指标：Acc / MAE / IoU。"""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_mae = float("inf")
    best_iou = 0.0
    best_score = -1.0  # 用 acc + iou 作为综合分数选择最佳轮次

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, y_cls, y_box in train_loader:
            xb = xb.to(device)
            y_cls = y_cls.to(device)
            y_box = y_box.to(device)

            logits, pred_box = model(xb)
            loss_cls = F.cross_entropy(logits, y_cls)
            loss_box = F.smooth_l1_loss(pred_box, y_box)
            loss = loss_cls + lambda_box * loss_box

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 4 == 0 or epoch == epochs:
            val_loss, val_acc, val_mae, val_iou = evaluate_detector(
                model, val_loader, device, lambda_box=lambda_box
            )
            score = val_acc + val_iou
            if score > best_score:
                best_score = score
                best_acc, best_mae, best_iou = val_acc, val_mae, val_iou

            print(
                f"[Detector] 轮次 {epoch:02d} | 验证损失={val_loss:.4f} "
                f"| 分类Acc={val_acc:.4f} | 边框MAE={val_mae:.4f} | IoU={val_iou:.4f}"
            )

    return best_acc, best_mae, best_iou


# -----------------------------
# 保存与加载
# -----------------------------
def save_detector_checkpoint(
    model: nn.Module,
    path: str = "artifacts/tiny_detector.pt",
    class_names: tuple[str, ...] = ("square", "h_rect", "v_rect"),
    input_size: int = 32,
) -> str:
    ckpt = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "input_size": input_size,
    }
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, file)
    return str(file)


def load_detector_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[TinyDetectorCNN, tuple[str, ...], int]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    class_names = tuple(ckpt.get("class_names", ("square", "h_rect", "v_rect")))
    input_size = int(ckpt.get("input_size", 32))
    model = TinyDetectorCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, class_names, input_size


# -----------------------------
# 实时推理
# -----------------------------
def run_realtime_detection(
    model: nn.Module,
    device: torch.device,
    class_names: tuple[str, ...] = ("square", "h_rect", "v_rect"),
    input_size: int = 32,
    camera_id: int = 0,
    min_conf: float = 0.35,
) -> None:
    """
    实时检测（YOLO 风格显示）：
    - 读取摄像头画面
    - 绘制边框 + 类别 + 置信度
    - 按 q 退出
    """

    try:
        import cv2
    except ImportError:
        print("缺少 opencv-python，请先安装：pip install opencv-python")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("摄像头打开失败，请检查 camera_id 或摄像头权限。")
        return

    model.eval()
    print("实时检测已启动，按 q 退出。")

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
            x = x.to(device)

            logits, pred_box = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_cls = probs.max(dim=1)

            if conf.item() >= min_conf:
                cx, cy, bw, bh = pred_box[0].tolist()
                x1 = int((cx - 0.5 * bw) * w)
                y1 = int((cy - 0.5 * bh) * h)
                x2 = int((cx + 0.5 * bw) * w)
                y2 = int((cy + 0.5 * bh) * h)

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                label = f"{class_names[int(pred_cls.item())]} {conf.item():.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Tiny CNN Detector Realtime", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# 主流程
# -----------------------------
def run_cnn_realtime_pipeline(device: torch.device) -> None:
    """如有权重则直接加载，否则先训练，然后进行实时检测。"""

    ckpt_path = "artifacts/tiny_detector.pt"
    class_names = ("square", "h_rect", "v_rect")

    if Path(ckpt_path).exists():
        print(f"检测到权重文件，直接加载：{ckpt_path}")
        detector, class_names, input_size = load_detector_checkpoint(ckpt_path, device)
    else:
        print("未检测到权重文件，开始训练 TinyDetectorCNN...")
        x_det, y_det_cls, y_det_box = make_detection_dataset(num_samples=4200, image_size=32)
        dataset_det = TensorDataset(x_det, y_det_cls, y_det_box)

        train_size = int(0.8 * len(dataset_det))
        val_size = len(dataset_det) - train_size
        train_det, val_det = random_split(dataset_det, [train_size, val_size])
        train_loader_det = DataLoader(train_det, batch_size=64, shuffle=True)
        val_loader_det = DataLoader(val_det, batch_size=64)

        detector = TinyDetectorCNN(num_classes=len(class_names))
        print(f"检测模型参数量：{count_parameters(detector)}")
        best_acc, best_box_mae, best_iou = train_detector(
            detector,
            train_loader_det,
            val_loader_det,
            epochs=16,
            lr=1e-3,
            device=device,
            lambda_box=3.0,
        )
        save_detector_checkpoint(
            detector,
            path=ckpt_path,
            class_names=class_names,
            input_size=32,
        )
        print(
            f"训练完成 -> 分类Acc={best_acc:.4f}, "
            f"边框MAE={best_box_mae:.4f}, IoU={best_iou:.4f}"
        )
        detector, class_names, input_size = load_detector_checkpoint(ckpt_path, device)

    run_realtime_detection(
        detector,
        device=device,
        class_names=class_names,
        input_size=input_size,
        camera_id=0,
        min_conf=0.35,
    )


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    run_cnn_realtime_pipeline(device)


if __name__ == "__main__":
    main()
