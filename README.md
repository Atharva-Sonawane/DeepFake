# Deepfake Detection System 🕵️‍♂️🎭

A deepfake detection system built using **EfficientNet-B5** for image and video classification.  
This project detects whether a given face image or video is **Real** or **Fake**.

---

## 📌 Features
- Deepfake detection using **EfficientNet-B5**
- Supports both **image** and **video** inputs
- Training, evaluation, and inference scripts included
- Face extraction & preprocessing utilities
- Example dataset and demo included
- Model weights provided separately (download link below)
---

## 📂 Project Structure
deepfake-detection/

│── README.md # Project overview, setup & usage instructions

│── LICENSE # License (MIT/Apache/etc.)

│── requirements.txt # Dependencies

│── train.py # Training script

│── finetune_b5.py # Evaluation script

│── inference.py # Run detection on new image/video

│── model.py # Model architecture

│── data.py # Data loading & preprocessing

│── utils.py # Helper functions

📊 Dataset

Due to size and privacy, the dataset is not included here.
You can download datasets such as:

FaceForensics++  :-  https://github.com/ondyari/FaceForensics

DeepFake Detection Challenge Dataset   :-  https://www.kaggle.com/c/deepfake-detection-challenge/data


🧠 Model Weights

Trained model weights are not stored in GitHub due to size.

🔗 Download pretrained EfficientNet-B5 weights here:-  https://drive.google.com/drive/folders/1soJ9TbS2bcvA98Bm8RB_GJ2pUQL9iXI9?usp=sharing


🚀 Usage
1️⃣ Training
python train.py --data_dir data/train --epochs 20 --batch_size 32 --lr 1e-4

2️⃣ Evaluation
python evaluate.py --data_dir data/test --model_path models/deepfake_model.pth

3️⃣ Inference (Image/Video)
# For image
python inference.py --input examples/sample_fake.jpg --model_path models/deepfake_model.pth

# For video
python inference.py --input examples/sample_video.mp4 --model_path models/deepfake_model.pth


🛠️ Tech Stack

1.Python 3.8+
2.PyTorch
3.EfficientNet-B5
4.OpenCV
5.RetinaFace
6.NumPy / Pandas / Matplotlib


<img width="1630" height="515" alt="3" src="https://github.com/user-attachments/assets/91924e5b-bf83-4e5e-aa51-7258ddd0ec4d" />
<img width="1887" height="822" alt="2" src="https://github.com/user-attachments/assets/da762b46-e401-423c-ac6f-e2785b29f5cd" />
<img width="1241" height="699" alt="1" src="https://github.com/user-attachments/assets/ce8b21ec-3fdf-4e2d-90b9-1a6653bb8baa" />

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

 🤝 Contributing

Contributions are welcome!
Please open an issue or submit a pull request if you’d like to improve the project.




