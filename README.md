# KD-CIFAR-10
#  Knowledge Distillation: Teacher Model Training

This project focuses on training a **Teacher model** on the CIFAR-10 dataset as a first step toward model compression via **Knowledge Distillation**.

##  What’s Done

- Trained a high-capacity **Teacher model** (Resnet18)
- Achieved high accuracy on CIFAR-10(Accuracy of 96.06%
- Saved model weights and training plots
- Ready for distillation to a smaller student model

## What’s Included

- ResNet-18 modified for CIFAR-10
- Cutout regularization
- Label smoothing loss
- Mixed-precision (AMP) training
- CosineAnnealingLR scheduler
- Confusion matrix and classification report

## 📊 Results (Teacher Model)

| Metric          | Value     |
|-----------------|-----------|
| Final Accuracy  | **96.06%** |
| Params          | ~11M      |
| File Size       | ~45MB     |

![Accuracy Graph] ![Confusion Matrix] - (teacher_accuracy_confusion_matrix.png)

---

## 🚀 How to Run

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/yourusername/knowledge-distillation.git
cd knowledge-distillation
pip install -r requirements.txt
