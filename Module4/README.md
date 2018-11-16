# Module4. Implementing Time Series Prediction Models (More Comprehensive)

*Will be released after Module3*

## Goal
- Learn how to implement your own neural network models with cryptocurrency data which includes bid/ask volumes, minute level prices and trading volumes 

## Modules
- M4.1 Orderbook Data
    - [Reading] [Read Section 3 Dataset Characteristics](https://drive.google.com/open?id=1U86rW0rL7ZMld4txXi40SEfACVJ0r3vZ)
    - [Hands-on-Labs] Orderbook Data Exploration [[Code](https://colab.research.google.com/drive/1eurWi1Mmw2ZuPh2KlHFG58emDoX-GHdg)] [[Data](https://drive.google.com/open?id=1_GXzTuyIopvkkOeCxHanVZKa0tKclD6F)]
    - [Assignment] Compare FNN, CNN, RNN Model with Orderbook Data [[Problem](https://colab.research.google.com/drive/1oPvZAIsS_NVd80o-iaxRMyJJa5MYrhEH)] [[Solution](https://colab.research.google.com/drive/1iRuwyBpW_Ce4QWDHw0GirMyMN5Dusn3X)]
    
- M4.2 CNN Encoder + RNN Time Series Prediction Model with Limit Orderbook Data
    - [Reading] [Read Section 5.4.1. Categorical Model Architecture](https://drive.google.com/open?id=1U86rW0rL7ZMld4txXi40SEfACVJ0r3vZ)
    - [Hands-on-Labs] CNN Encoder + RNN Model [[Code](https://colab.research.google.com/drive/10Hlek8FBzNtL1QwpfdAL3TQROjHSvv-K)]
    - [Assignment] CNN Encoder + RNN Time Series Prediction Model with Limit Orderbook Data [[Problem](https://colab.research.google.com/drive/1AH56wjfso6eEcv6hWKJ86TMI_8N-t3pH)] [[Solution](https://colab.research.google.com/drive/1XJ18u6j_3i8MlXtnkszujwwx5d4RL9eQ)]

- M4.3 Stacked Autoencoders to Extract Features from Data
    - [Reading] [A Gentle Introduction to the Keras Functional API](https://machinelearningmastery.com/keras-functional-api-deep-learning/)
    - [Hands-on-Labs] [Autoencoders for Feature Extraction](https://blog.keras.io/building-autoencoders-in-keras.html) [[Code](https://colab.research.google.com/drive/1OwcK9s5aRqBIw9vFWD6HmubJVwLG3GZz)]
    - [Assignment] Stacked Autoencoders to Extract Features from Data [[Problem](https://colab.research.google.com/drive/1W-r49D0F69WgAdyb3Vnif0Q4u-j8XANr)] [[Solution](https://colab.research.google.com/drive/1UO1GU3O9mSz9UexjLiNOOENQ_0s-b3SC)]

- M4.4 Final Neural Network Model
    - [Reading] [A Deep Learning Framework for Financial Time Series using Stacked Autoencoders and Long-Short Term Memory](https://drive.google.com/open?id=19zYBAUV5at8RY43TvOUJj_4PdSRGZcQT)

- M4.5 Final Neural Network Model
    - [Hands-on-Labs] Stacked Autoencoder LSTM [[Code](https://colab.research.google.com/drive/1OsnY5RCDSX4-JuusloTR0bkDbKb2nra8)]
    - [Assignment] Write Your Own Neural Network Model for Financial Time Series Prediction [[Problem](https://colab.research.google.com/drive/12WhwLOMxnLH94qIa8B3SaneJpePJkZMF)]

- M4.6 Assignment
- M4.7 Slack Discussion

## Q&A
- Module4. [Implementing Time Series Prediction Models (More Comprehensive)](../Q&A/Module4.md)

## What's next?
The beauty of this model is the once the construction is understood, the individual models can be swapped out for the best model there is. So over time the actual models used here will be different but the core framework will still be the same.

## References
- [A deep learning framework for financial time series using stacked autoencoders and long-short term memory](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180944), July 2017
- [AlphaAI: Multilayer neural network architecture for stock return prediction](https://github.com/VivekPa/AlphaAI?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more#neural-network-model), Oct 2018