# trace_classifier
## Motivation
I was inspired by Detective Conan to create this.\
名探偵コナン[から紅の恋歌](https://ja.wikipedia.org/wiki/%E5%90%8D%E6%8E%A2%E5%81%B5%E3%82%B3%E3%83%8A%E3%83%B3_%E3%81%8B%E3%82%89%E7%B4%85%E3%81%AE%E6%81%8B%E6%AD%8C)

## System Details
This system identifies the type of bloodstain.\
It uses a CNN model to perform image classification.\
Since there are no images of bloodstains, we created a pseudo-dataset using images of paint and Halloween makeup to train the system.\
Like in the movie, we classify images with blood dripping (rain) and images with blood smudged by fake makeup (draw).
Current test accuracy is 70~80%.

## Method used
Please execute the following code and install the necessary libraries.
```
pip install -r requirements.txt
```
Run train.py to train CNN model and evaluate.py to launch a trained model.\
(If you run evaluate.py, you will also see an example of the image used for training.)

## Future plans
We aim to improve the accuracy of the model by adding more images and changing the architecture.\
We also plan to develop a GUI for this bloodstain discriminator. (Ideally, the UI will be that of a movie.)
