import data_loader
import network

training_data, validation_data, test_data = data_loader.load_data_wrapper()

net = network.Network([784, 30, 30, 10])

training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)


net.SGD(training_data, 30, 10, 1.0, test_data=test_data)
