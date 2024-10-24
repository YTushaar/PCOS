# PCOS Prediction using Multi-Layer Perceptron

## Introduction

PCOS Prediction using Multi-Layer Perceptron (MLP)
Polycystic Ovary Syndrome (PCOS) is a common endocrine disorder affecting women of reproductive age. Characterized by irregular menstrual cycles, excess androgen levels, and polycystic ovaries, PCOS is a leading cause of infertility, metabolic disorders, and cardiovascular risks. Early detection and management of PCOS are crucial for improving patient outcomes and quality of life. However, diagnosing PCOS can be challenging due to its complex and multifactorial nature.

In recent years, machine learning has emerged as a powerful tool for aiding medical diagnosis, offering the ability to analyze large datasets and uncover hidden patterns that may not be evident to human practitioners. Among various machine learning techniques, Multi-Layer Perceptron (MLP), a type of feedforward neural network, is particularly well-suited for classification tasks such as disease prediction.

## About MLP (Multi-Layer Perceptron)

MLP is an artificial neural network composed of multiple layers of nodes: an input layer, one or more hidden layers, and an output layer. Each node, or neuron, in one layer is connected to every neuron in the next layer through weighted connections. MLPs are capable of learning complex relationships between inputs and outputs through backpropagation and gradient descent optimization.

In the context of PCOS prediction, an MLP can be trained using a dataset containing medical features (e.g., BMI, insulin resistance, hormonal levels, menstrual irregularities) to predict whether a woman has PCOS. The use of hidden layers enables MLPs to model non-linear relationships between these features, making them a robust choice for this classification problem.

## About Model

- **Data Collection and Preprocessing**: The first step involves collecting a dataset with medical features that are relevant for predicting PCOS, such as age, weight, BMI, hormonal profiles, insulin levels, and reproductive history. The data must be cleaned, normalized, and split into training and testing sets to prepare it for model development.

- **Feature Selection**: Relevant features that contribute most to the prediction of PCOS are identified, which improves the model's accuracy and reduces overfitting.

- **MLP Model Architecture**: An MLP model is designed with an input layer corresponding to the number of selected features, one or more hidden layers with a suitable number of neurons, and an output layer representing the classification result (PCOS positive or negative).

- **Model Training**: The MLP is trained using a labeled dataset. During training, the model adjusts the weights of the connections between neurons using backpropagation to minimize the error in predictions.

- **Model Evaluation**: After training, the model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score to assess how well it predicts PCOS on unseen data.

- **Hyperparameter Tuning**: The model’s performance can be improved by tuning hyperparameters such as the learning rate, number of hidden layers, activation functions, and regularization techniques.

## About Project

For this project, we used parameters out of 40 parameters.

```text
follicle_r
follicle_l
hair_growth
skin_darkening
weight_gain
cycle
fast_food
amh
pimples
weight_kg
```

## Installation

- Create a folder name in git bash

```bash
mkdir PCOS-Research
```

- Clone the project.

```bash
git clone https://github.com/debasishray16/PCOS-Research.git
```

- Run the command inside current directory.

```bash
streamlit run app.py
```

Note: The model is not imported. Everytime, you run streamlit server, it will run the model and will train at the same time.
At last, it will show the result.

## Output

![PCOS_Interface](/assets/images/PCOS_Interface.png)

## 

### Case 1: "Where there is no possibility for PCOS."

```text

Range:
------
follicle_r: (2-5)
follicle_l: (3-6)
weight_kg: (45-60)
hair_growth: (No)
skin_darkening: (No)
weight_gain: (No)
cycle: (4[regular_Cycle])
fast_food: (No)
amh: (1-3)
pimples: (No)

```

![PCOS No Result](/assets/images/PCOS_No_Result_Interface.png)

##


### Case 2: "Where there is possibility for PCOS."

```text

Range:
------
follicle_r: (14-20)
follicle_l: (12-18)
weight_kg: (70-90)
hair_growth: (Yes)
skin_darkening: (Yes)
weight_gain: (Yes)
cycle: (2[Irregular_Cycle])
fast_food: (No)
amh: (5-10)
pimples: (No)

```

![PCOS Yes Result](/assets/images/PCOS_Yes_Result_Interface.png)
