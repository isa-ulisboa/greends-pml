# Computing gradients with PyTorch

Video suggestion: [Backpropagation by Patrick Loeber](https://www.youtube.com/watch?v=3Kb0QS6z7WA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=4). The author explains what is a computation graph and how PyTorch uses it to compute gradients. The example uses a very simple model: $\hat{y}=w \cdot x$ and the MSE loss which is just $(\hat{y}-y)\^2$.

- Pipeline for the regression problem:
  - Prepare data
  - Design model (input, output size, model: perceptron)
  - Construct loss and optimizer
  - Training loop
    - Forward pass: prediction and loss
    - Backward pass: gradients
    - Update weights

Below is discussed a step by step code in Python for the linear regression model  that is based on our knowledge o  the gradient expression for the MSE loss function. Then, the script is converted into a Pytorch code that can be easily generalized to other models and other loss functions.

Video suggestion: [Gradient Descent with Autograd and Backpropagation by Patrick Loeber](https://www.youtube.com/watch?v=E-I2DNVzQLg). The author first uses `numpy` to create a gradient descent script for a  linear regression model and then replaces the manual gradient calculation by `PyTorch` automatic gradient calculation. The following scripts illustrate the possibilities to move from a low level Python code to a higher level PyTorch code for the simple Linear Regression problem. The higher-level code can be easily adapted to more complex models and other optimizers.

- [numpy version](https://github.com/patrickloeber/pytorchTutorial/blob/master/05_1_gradientdescent_manually.py)
- [torch version](https://github.com/patrickloeber/pytorchTutorial/blob/master/05_2_gradientdescent_auto.py)
- [torch version with torch loss criterion and optimizer](https://github.com/patrickloeber/pytorchTutorial/blob/master/07_linear_regression.py)

The following table illustrates the changes from a basic Python script which is dependent on the model, loss, etc,  to a PyTorch higher-level script that can easily generalized to other models, loss functions and optimizer strategies.

| Basic Python | PyTorch 
|---|---
| Define model explicitly | Use a pre-defined model
|`def predict(x):`|`torch.nn.Linear(in_size,out_size)`
| Define loss explicitly | Use a pre-defined loss function
|`def loss(y,y_pred):`|`loss=torch.nn.MSEloss(y,y_pred)`
| Loss optimization strategy | Use a pre-defined optimizer
| | `optimizer=torch.optim.SGD(params, learn_rate)`
| Compute *ad hoc* gradient | **Use built-in backpropagation mechanism**
|`def gradient(x,y,y_pred):`|`loss.backward()`
|Update weights explicitly| `optimizer.step()`

