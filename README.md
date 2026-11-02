# Bahraini Currency Recognition with YOLO

A deep learning project that detects and recognizes Bahraini currency (BHD) in images, with a Streamlit app for testing it on camera or phone photos.

## Dataset

- **Source:** Roboflow Universe, version 2: https://universe.roboflow.com/noora-alkhalaqi/bahraini-currency-1ka60/dataset/2
- **Task:** Object detection (boxes around each note or coin, with a class label)
- **Size:** 1075 images in version 2 (after augmentation)
- **Classes (10):** 0.5 BD, 1 BD, 10 BD, 20 BD, 5 fils, 10 fils, 25 fils, 50 fils, 100 fils
- **Splits:** train / valid / test
- **Collection:** photos taken with phones in different lighting, backgrounds and angles, plus images collected online

### Preprocessing and augmentation (done in Roboflow)

- Preprocessing: Resize width: 512 · height: 512 ·
- Augmentations:  Rotation, Noise, Image, Mosaic, Camera Gain

## Model

- **Architecture:** YOLO26s (small), starting from pretrained `yolo26s.pt`
- **Training:** 5 epochs on dataset v2, then fine-tuned for 10 more epochs starting from the first run's last checkpoint
- **Hardware:** Google Colab, Tesla T4 GPU
- **Library:** Ultralytics 8.4.198
- **Trained weights:** `best.pt`

## Evaluation (test set: 13 images, 15 objects)

| Metric | Value |
|---|---|
| Precision | 0.818 |
| Recall | 0.870 |
| mAP50 | 0.889 |
| mAP50-95 | 0.775 |



## Key questions

1. **Which currency is most confusing?** 5 and 10 fils
2. **Are similar-looking notes misclassified?** yes, the model keeps confusing 25 fils with 50 fils. 
3. **Does background affect performance?** No. it does not. 

## Real-world testing

- Tested the model with the live webcam (OpenCV) and with photos in the Streamlit app.
  

## Limitations

- Still confuses the filses with each other even after adding data. 
- Some images came from the internet, so the training data does not fully match real phone photos.
