# lab

## nn_preparing
just a simple dnn model made only by numpy.<br>
mnist number classification<br>
you can see my past endeavor in pdf file.<br>
build this model to do experiment

### main.py
tensorflow library has only used by downloading mnist data<br>
if you run you'll see<br>
![loss/acc](gif1.gif)

### test.py
compare hand made model's accruracy vs. tf.keras model accruracy

## loss_function_experiment
### idea 1
not using a gradient descent to find a global minimum<br>
just set loss as 0, and back propagating using inversed matrix

#### idea1 - result
make a single layer nn, (dnn)<br>
![acc_without gradient_descent](img1.png)

#### blocker
it's difficult to split 1 matrix to 2.<br>
wanna make a two layer model (first output size 128, second output size 10) without doing gradient descent<br>
![acc_without gradient_descent](img2.png)
I have yellow, I can make green, but not a freakin tensor (double line)

#### things to do
make a 2 layer model first. (random number)<br>
do forward propagation.<br>
and think about what i can do with inversed matrix i got.

