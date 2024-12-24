# My classifier for MNIST
Credit is given to [GeeksForGeeks](https://www.geeksforgeeks.org/handwritten-digit-recognition-using-neural-network/)
and Stanford CS231n.
Also, big thanks for `hsjeong5`,
who made a python script for loading the MNIST dataset.
`tf.keras` isn't supported on Python 3.13, and the script (`load_mnist.py`)
was a big help in loading the dataset.
## Backstory
This comes from a while back, specifically last summer,
when I was first getting my feet wet in the field of neural networks.
I was trying to create a classifier for MNIST.
I spent weeks trying, but ended up with nothing.
Because of how vindictive I am, I'm counting this as my revenge.

The backend for the GUI is a 6-layer deep neural network that achieves
a validation accuracy of 98.47% on the MNIST test set. (It was not
introduced to these examples during training.) It was trained using code
that was previously used to do a homework assignment from Stanford CS231n.

To run this, go to your terminal and run
```
ipython classifier.py
```
## To-do
I'm finishing up the batchnorm section of the homework for CS231n,
expect to see batchnorm and layernorm here in the near feature!
