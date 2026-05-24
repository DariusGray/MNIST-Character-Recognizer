# MNIST Character Recognizer

A deep learning project that recognises handwritten uppercase alphabet characters from image data using Convolutional Neural Networks (CNNs).  
This project was developed for **Deep Learning CA1 Part A** and focuses on building, evaluating, improving, saving, and testing an image classification model.

The final model is a custom CNN trained on the **EMNIST Letters** dataset and is able to classify handwritten letters from **A to Z**.

---

## Project Overview

This project explores how deep learning can be used for handwritten character recognition. The dataset contains grayscale handwritten letter images, where each image is represented as a 28x28 pixel grid.

The main goal is to train a model that can look at a handwritten alphabet image and predict which uppercase letter it represents.

The project includes:

- Dataset loading and inspection
- Exploratory Data Analysis (EDA)
- Image orientation correction for EMNIST images
- Data preprocessing and normalization
- Baseline Dense Neural Network
- CNN model development
- Hyperparameter tuning
- Data augmentation experiment
- Brief transfer learning comparison
- Error analysis
- Model saving and loading
- Interactive Gradio demo for handwritten letter prediction

---

## Dataset

The project uses the **EMNIST Letters** dataset stored as:

```
emnist-letters.csv

```
Each row contains:

- 1 label column
- 784 pixel values representing a 28x28 grayscale image

The original dataset shape used in the notebook was:

`(99040, 785)`

## Important Preprocessing Steps

The EMNIST images are not displayed in the correct orientation at first, so the notebook applies an orientation fix.

The image preprocessing includes:

1. Reshape each image into 28x28
2. Rotate the image
3. Flip the image horizontally
4. Normalize pixel values from 0-255 to 0-1
5. Reshape images for CNN input as 28x28x1

The project also handles unusual labels such as -1 and -2 by filtering them out so that the final classification task focuses only on alphabet labels from A to Z.

---

## Model Development

Several models and experiments were tested in this project.

### 1. Baseline Dense Neural Network

A simple Dense model was built first as a baseline.
This helped show why CNNs are better suited for image-based tasks compared with fully connected Dense networks.

### 2. Simple CNN

A basic CNN was then trained using convolutional and pooling layers.

### 3. Deeper CNN Experiments

The CNN architecture was improved through experiments such as:

- Adding more convolutional layers
- Increasing filter sizes
- Using Batch Normalization
- Increasing Dropout
- Trying smaller batch sizes
- Applying Early Stopping

### 4. Data Augmentation

A data augmentation experiment was also tested using small transformations such as:

```
rotation
width shift
height shift
zoom
```

This was done to check whether the model could generalise better to slightly different handwritten styles.

5. Transfer Learning

A brief MobileNetV2 transfer learning experiment was tested for comparison.
However, the custom CNN performed better because it was more suitable for the 28x28 grayscale handwritten character dataset.

---

## Best Model

The best model was the custom CNN with:

```
Convolutional layers
Batch Normalization
Max Pooling
Dropout
Dense layers
Early Stopping
```

The final saved model file is:

`best_cnn_model.h5`

The model weights were also saved as:

`best_cnn_model_weights.h5`

---

## Results

The final custom CNN achieved strong performance on the test set.

```
Final Best CNN Test Accuracy: 0.9365
Loaded Model Test Accuracy:   0.9365
```

This confirms that the saved model was loaded correctly and gave the same test accuracy as the original trained model.

The classification report showed that the model performed well for most letters, with an overall accuracy of around **94%**.

Some letters were harder to classify because of similar handwriting shapes, especially characters such as:

```
I and L
G and Q
A and H
```

These errors are realistic because handwritten letters can sometimes look visually similar, especially when the image is small or unclear.

---

## Interactive Demo

The project includes a simple interactive demo using Gradio.

The demo allows the user to draw a capital letter in a sketchpad, then the trained CNN model predicts the letter.

Example output:

`Prediction: A (Confidence: 95.42%)`

The demo loads the saved model:

`model_demo = load_model("best_cnn_model.h5")`

Then it preprocesses the drawn image into the same format used during training:

```
28x28 grayscale image
normalized pixel values
CNN input shape: (1, 28, 28, 1)
```

---

## Project Structure

```
MNIST-Character-Recognizer/
│
├── ST1504_CA1_PartA_KaungNyiHein.ipynb
├── emnist-letters.csv
├── best_cnn_model.h5
├── best_cnn_model_weights.h5
├── README.md
└── requirements.txt
```

---

## Technologies Used

```
Python
NumPy
Pandas
Matplotlib
Scikit-learn
TensorFlow
Keras
Gradio
Pillow
VisualKeras
```

---

## How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/DariusGray/MNIST-Character-Recognizer.git
cd MNIST-Character-Recognizer
```

### 2. Install the required libraries

`pip install numpy pandas matplotlib scikit-learn tensorflow gradio pillow visualkeras`

Or, if a `requirements.txt` file is usable:

`pip install -r requirements.txt`

### 3. Open the notebook

`jupyter notebook ST1504_CA1_PartA_KaungNyiHein.ipynb`

### 4. Run the notebook cells

Run the notebook from top to bottom to reproduce the full workflow.

Make sure the dataset file is in the same folder:

`emnist-letters.csv`

---

## Running the Demo

After the model has been trained and saved, run the Gradio demo section in the notebook.

The demo will start locally and show a link similar to:

`http://127.0.0.1:7861`

Open the link in your browser, draw a capital letter, and submit it to see the model prediction.

---

## Key Learning Points

Through this project, I learned how to:

- Prepare image data for deep learning
- Fix image orientation issues in EMNIST
- Normalize pixel values for neural networks
- Build a Dense baseline model
- Build and tune CNN models
- Use Batch Normalization and Dropout to improve generalisation
- Apply Early Stopping to reduce overfitting
- Compare different model experiments fairly
- Perform error analysis using confusion matrices and classification reports
- Save and reload a trained Keras model
- Build a simple interactive demo using Gradio

---

## Limitations

Although the final model performs well, there are still some limitations:

- Some handwritten letters are visually similar, such as `I` and `L`
- The model expects clean 28x28 grayscale-style input
- The Gradio sketch input may not always perfectly match the training data style
- The model only recognises uppercase alphabet characters from A to Z
- It does not currently recognise lowercase letters, digits, or symbols

---

## Future Improvements

Possible improvements include:

- Training with more varied handwriting samples
- Improving the live demo preprocessing
- Adding support for lowercase letters and digits
- Testing more advanced CNN architectures
- Improving difficult class predictions such as `I`, `L`, `G`, and `Q`
- Deploying the demo online using Hugging Face Spaces or another web platform

---

## Author
```
Kaung Nyi Hein (Darius)
Class: DAAA/FT/2B/24
Project: Deep Learning CA1 Part A
```

---

## Repository
`https://github.com/DariusGray/MNIST-Character-Recognizer`
