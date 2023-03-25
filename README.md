# Sentiment-Analysis

## Dataset:
The IMDB dataset consists of 50,000 movie reviews, evenly split between positive and negative labels. The task is to develop a model that can accurately classify each review as positive or negative based on its content. This is a challenging problem because movie reviews can be subjective, and the same text can be interpreted differently by different individuals. Additionally, movie reviews can contain various linguistic features, such as sarcasm, irony, and metaphor, which can make sentiment analysis more challenging. Therefore, the goal of this project is to develop a sentiment analysis model that can accurately classify movie reviews on the IMDB dataset while addressing these challenges.

IMDB Dataset: https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf


Also see the [Sentiment Ananlysis](http://nlpprogress.com/english/sentiment_analysis.html) page at NLP Progress and the [Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)

## Model: Bidirectional GRU


This model is designed for sentiment analysis of IMDb movie reviews using bidirectional GRU and attention mechanism. The model begins with an embedding layer, which converts the input sequence of words into a dense vector representation. The embedding layer has n_unique_words as its input dimension, and each word is mapped to a 128-dimensional vector. The input_length parameter specifies the maximum length of input sequences.

The next layer is a Bidirectional GRU layer, which processes the input sequence in both forward and backward directions. The GRU layer has 128 units and returns the sequences of hidden states at each time step.


The attention layer is then added, which receives the sequences of hidden states from the Bidirectional GRU and outputs a single vector representation of the input sequence. This is accomplished by calculating a weight vector for each hidden state, which indicates its importance in representing the input sequence. The attention mechanism can capture important information from the input sequence that might be overlooked by the Bidirectional GRU.

A dropout layer with a dropout rate of 0.5 is added to prevent overfitting. A dense layer with 16 units and sigmoid activation is then added, followed by another dense layer with a single unit and sigmoid activation. The final dense layer outputs the predicted sentiment of the input sequence as either positive or negative.

The model is compiled with binary cross-entropy loss function, Adam optimizer, and accuracy as the evaluation metric. The architecture can be visualized in figure .


![model](https://user-images.githubusercontent.com/50993551/227674242-af3cdd2b-aad0-4dea-a252-3bc68137001c.png)

## Hyperparameter Tuning:

Hyperparameter tuning is an essential step in optimizing the performance of any deep-learning model. In this particular model, two hyperparameters, the dropout rate, and the GRU hidden units were tuned. The dropout rate is a regularization technique that randomly drops out some neurons during training to reduce overfitting. The GRU hidden units determine the complexity of the model and the amount of information it can capture. The values of these hyperparameters were chosen based on their effect on the validation accuracy and loss as shown in figure  where "mod1" and "mod2" are models with 64 hidden units and 0.5 and 0.3 dropout rates respectively, while "mod3" and "mod4" are models with 128 hidden units and 0.5 and 0.3 dropout rates respectively.

![plot](https://user-images.githubusercontent.com/50993551/227674600-0be22a99-b839-4367-9033-2b3e77499359.png)

![plot2](https://user-images.githubusercontent.com/50993551/227674648-05e58c89-8e89-4edc-bbab-70744ef8df8d.png)


## Results:

### ACCURACY: 86.8%

### LOSS: 0.29

After experimenting with various values of dropout rate and GRU hidden units, we chose to set the GRU hidden units to 128 and the dropout rate to 0.5 based on the validation accuracy and loss of "mod3". These values produced the best validation accuracy and loss, indicating that they provided the optimal balance between model complexity and generalization. However, it's important to note that hyperparameter tuning is an iterative process and should be performed carefully with a thorough understanding of the model and the data.

After training and evaluating the final model for sentiment analysis of IMDb movie reviews using bidirectional GRU and attention mechanism with a dropout rate of 0.5 and GRU hidden units of 128, we obtained a test accuracy of 86.8% and a test loss of 0.293. These results indicate that the model is able to effectively classify the sentiment of the movie reviews.

To further evaluate the performance of the model, we generated a confusion matrix shown in figure on the test set. The confusion matrix shows that the model correctly predicted 10947 positive reviews and 10759 negative reviews. However, the model incorrectly predicted 1741 negative reviews as positive and 1553 positive reviews as negative.

These results indicate that the model has a good balance between precision and recall, which is crucial for sentiment analysis tasks. The high accuracy indicates that the model is capable of effectively identifying the sentiment of movie reviews, making it a useful tool for analyzing large volumes of movie reviews.

![plot3](https://user-images.githubusercontent.com/50993551/227673893-bc00cb3a-8e2b-403b-90b8-5783f69fdb53.png)
