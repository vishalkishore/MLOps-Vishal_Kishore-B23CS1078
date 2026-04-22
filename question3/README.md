# Question3

This folder contains the complete semantic segmentation solution for the assignment's Question2 deliverable:

- 80/20 train-test split with seed 42
- custom PyTorch dataset and dataloaders
- UNet-based segmentation model for 23 classes
- training loss, mIoU, and mDice plots saved under `question3/Question2/`
- Gradio app with two tabs:
  - training summary and test metrics
  - prediction viewer for 4 uploaded test images

## Commands

Train and save artifacts:

```bash
python3.11 main.py train
```

Launch the app after training:

```bash
python3.11 main.py app
```
