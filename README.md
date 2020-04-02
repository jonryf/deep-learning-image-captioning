# Image captioning using vanilla RNN and Long-Term-Short-Memory (LSTM)

Exploring the power of Recurrent Neural Networks to deal with data that has temporal structure. Specifically, to generate captions for images. In order to achieve this, we will need an encoder- decoder architecture. Simply put, the encoder will take the image as input and encode it into a vector of feature values. The decoder will take this output from encoder as hidden state and starts to predict next words at each step. The following figure illustrates this:

![Architecture](/architecture.png)
