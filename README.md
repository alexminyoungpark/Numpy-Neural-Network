# Numpy로 DNN 만들기
# DNN과 numpy pseudo inverse로 만든 regression 성증 비교
# 만든 DNN을 뒤집어서 index로 28*28 size 숫자 이미지 생성

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

입력 index로 0과 8을 줬을 때 결과
!(nn_preraring/gan/80.png)
