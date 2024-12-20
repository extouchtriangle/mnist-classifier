from tensorflow.keras.datasets import mnist
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
import pickle
(mnist_X_train, mnist_y_train), (mnist_X_test, mnist_y_test) = mnist.load_data()
mnist_data={}
mnist_data['X_train'] = mnist_X_train
mnist_data['y_train'] = mnist_y_train
mnist_data['X_val'] = mnist_X_test
mnist_data['y_val'] = mnist_y_test
mnist_data['X_test'] = mnist_X_test
mnist_data['y_test'] = mnist_y_test
with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)
print(len(model.params))
y_test_pred = np.argmax(model.loss(mnist_data['X_test']), axis=1)
print('Test set accuracy: ', (y_test_pred == mnist_data['y_test']).mean())
from tkinter import *
from PIL import ImageGrab

window = Tk()
window.title("Handwritten digit recognition")
l1 = Label()


def MyProject():
    global l1

    widget = cv
    # Setting co-ordinates of canvas
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Image is captured from canvas and is resized to (28 X 28) px
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))

    # Converting rgb to grayscale image
    img = img.convert('L')

    # Extracting pixel matrix of image and converting it to a vector of (1, 784)
    x = np.asarray(img)
    x = x.reshape((1, 28, 28))


    # Calling function for prediction
    pred = np.argmax(model.loss(x), axis=1)

    # Displaying the result
    l1 = Label(window, text="Digit = " + str(pred[0]), font=('Algerian', 20))
    l1.place(x=230, y=420)


lastx, lasty = None, None


# Clears the canvas
def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()


# Activate canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


# To draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


# Label
L1 = Label(window, text="Handwritten Digit Recoginition", font=('Algerian', 25), fg="blue")
L1.place(x=35, y=10)

# Button to clear canvas
b1 = Button(window, text="1. Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_widget)
b1.place(x=120, y=370)

# Button to predict digit drawn on canvas
b2 = Button(window, text="2. Prediction", font=('Algerian', 15), bg="white", fg="red", command=MyProject)
b2.place(x=320, y=370)

# Setting properties of canvas
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)

cv.bind('<Button-1>', event_activation)
window.geometry("600x500")
window.mainloop()
