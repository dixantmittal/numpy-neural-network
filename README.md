# Numpy Neural Network
This project implements the neural network using numpy. 
After reading CS231n, it was initially made to test the understanding of Backpropagation.
Later, I started adding more stuff into it such as adam optimizer. 
Code structure is simple and written in an understandable manner.

### Installation

* Install Python 3
* Install pip
* run $> pip install -r requirements.txt

### Running the Code
Core code is written in class NeuralNetwork. 
It represents a model which can be saved and retrieved. 
Main.py is a helper class to test the code.

First instantiate the NeuralNetwork model with architecture (dimensions of hidden layers), number of features and number of output classes.
Then call model.train() with input data and their labels passed as parameters. Details about parameters in next section.
After the model is trained, call model.predict() to get the predicted labels.

### Model Parameters

Following are the parameters to define the model.
<table>
<tr>
    <td><b> n_features </b></td>
    <td> Number of features in the training data X. </td>
</tr>
<tr>
    <td><b> n_classes </b></td>
    <td> Number of output classes(labels) for the data. </td>
</tr>
<tr>
    <td><b> architecture </b></td>
    <td> Number of dimensions for each hidden layer. It is passed as a list containing dimension for each hidden layer in order.</td>
</tr>
</table>

### Training Parameters

Following are the training parameters to control the training process.
<table>
<tr>
    <td><b> epochs </b></td>
    <td> Number of times the full training data will be iterated. Default is 10 </td>
</tr>
<tr>
    <td><b> alpha </b></td>
    <td> Learning rate for the model. Default is 0.0003 </td>
</tr>
<tr>
    <td><b> batch_size </b></td>
    <td> Batch size to be processed on for single iteration </td>
</tr>
<tr>
    <td><b> delta </b></td>
    <td> L2 regularization rate. Default value is 0.0001 </td>
</tr>
<tr>
    <td><b> optimizer  </b></td>
    <td> Optimizer to use for the training. You can choose from 'adam'(ADAM) or 'sgd'(Stochastic Gradient). Default is 'adam' </td>
</tr>
<tr>
    <td><b> optimizer_params </b></td>
    <td> Based on the optimizer you choose, you need to pass the corresponding optimizer paramters in a dict. For e.g. for 'adam', you need to pass beta1 and beta2. Default parameters are for 'adam' </td>
</tr>
<tr>
    <td><b> verbose </b></td>
    <td> Whether to print the logs while training the model. Default is True. </td>
</tr>
</table>

### Authors

* **Dixant Mittal** - [dixantmittal](https://github.com/dixantmittal)

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
