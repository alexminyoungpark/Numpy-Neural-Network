# lab

## nn_preparing
just a simple dnn model made only by numpy.<br>
mnist number classification<br>
you can see my past endeavor how i struggled with math in pdf file.<br>
build this model to do whatever i want

### main.py
tensorflow library has only used by downloading mnist data<br>
if you run you'll see<br>
![loss/acc](gif1.gif)

### test.py
compare hand made model's accruracy vs. tf.keras model accruracy

## loss_function_experiment
### idea
not using a gradient descent to find a global minimum<br>
just set loss as 0, and back propagating using inversed matrix

#### result
85.34% accuracy w/ only one layer, no gradient descent, one propagation yay!<br>
![acc_without gradient_descent](img1.png)

#### goal
want model to make matrix if it's needed (if accr is not enough).

#### blocker
wanna make a two layer model (first output size 128, second output size 10) without doing gradient descent<br>
it's difficult to split 1 matrix to 2.<br>

![acc_without gradient_descent](img2.png)<br>
I have yellow, I can make green, but not a freakin tensor (double line)

#### plan
make a 2 layer model first. (random number)<br>
do forward propagation.<br>
and think about what i can do with inversed matrix i got.

#### things to do
-study matrix<br>
-deep devising about activation functon

## fine tuner (not yet)
### idea
tweezering the weight.
