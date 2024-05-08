# NLP Sentiment Analysis with PyTorch

This project is aimed at sentiment analysis of tweets using Natural Language Processing (NLP) techniques implemented in PyTorch. The model classifies tweets into positive and negative sentiment categories. It employs tokenization and stemming techniques using the NLTK library to preprocess the tweets.

## Model Architecture

The current model architecture consists of the following layers:

1. **Embedding Bag Layer**: Operates in a 120-dimensional space.
2. **ReLU Activation Function**: Non-linear activation function.
3. **Fully Connected (FC) Layer 1**: Converts embedding output into a 30-dimensional vector.
4. **ReLU Activation Function**: Non-linear activation function.
5. **Fully Connected (FC) Layer 2**: Relates its input to two classes (positive and negative sentiment).
6. **Sigmoid Activation Function**: Outputs probability scores for each sentiment class.

## Initial Accuracy

The initial accuracy achieved with the current model architecture is 76% for the dataset [1].

## Future Improvements

We are actively engaged in enhancing the accuracy of the model by integrating more advanced architectures such as LSTM and Attention mechanisms.

## Requirements

- Python 3.x
- PyTorch
- torchtext
- NLTK
- NumPy
- Pandas
- Seaborn
- scikit-learn

## Usage

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Preprocess your dataset using NLTK for tokenization and stemming.
4. Train the model using the Train section of the notebook.
5. Evaluate the model using the Evaluate section of the notebook.
6. Experiment with different architectures and hyperparameters to improve accuracy.

## References

[1] Sentiment140 dataset with 1.6 million tweets (https://www.kaggle.com/datasets/kazanova/sentiment140)