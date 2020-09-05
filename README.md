# bible-verse-generator

A simple NLP model to generate new bible verses. 

Our dataset is from kaggle, which is linked [here!](https://www.kaggle.com/oswinrh/bible)

## Usage

It's very simple, in fact you just need to run 

```
py model.py

``` 

## Training part
If you need to change training depth, width etc. you just need to change some of the lines such as:

```
def create_model():
    return (Sequential([

        LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True),
        LSTM(32),
        Dropout(0.2),
        Dense(y.shape[1], activation='softmax')]))
model = create_model()


model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
model.fit(x_train, y, epochs=7, batch_size=16)
```
You might want to save the model after training, so you just need to remove the comment tag from the 55th line of the code.

This model has been trained for 4-5 hours with Nvidia GTX 1050 card. 

## verses producted from low-trained model:

As dark on the face of the deep: and the Spirit of God was moving on the face of the waters

And will be a waste and a cause of wonder; and these nations will be the servants of the king


