# DeepLearning.AI Tensorflow Developer Professional Certificate
[Course 2: Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/lecture/nw4f6/introduction-a-conversation-with-andrew-ng)

## Week 1 - Dealing with Large Datasets

* Cats vs Dogs Dataset on Kaggle
    * [Notebook](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb)

## Week 2 - Image Creation and Augmentation

* Create more data and POVs to increase the size of your dataset
    * [Keras Image Preproccessing API Docs](https://keras.io/preprocessing/image/)
    * This week was more conceptual. It's basically the same challenges as last week, but with Image Augmentation


## Week 3 - Transfer Learning

Rather than needing to train a Neural Network from scratch, which needs a lot of data and time to train, you can instead download an open source model that someone else has already trained on a huge dataset maybe for weeks and use those parameters as a starting point to then train your model just a little bit more on perhaps a smaller dataset that you have for a given task.

* Use the Inception Model to bootstrap our own Neural Network
    * [Google Colab Notebook](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb)
    * [TensorFlow Docs on Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)

* Inspecting Layers (ie, get layer by name, etc)
* Dropouts - The idea behind Dropouts is that they remove a random number of neurons in your neural network. This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit!
    * Andrew Ng's YouTube video on [Dropouts](https://www.youtube.com/watch?v=ARq74QuavAo)


## Week 4 - Multiclass Classification

Classification problems that have 3 or more classes (like Rock/Paper/Scissors)

* Sign Language MNIST challenge to classify the alphabet using hand signs
