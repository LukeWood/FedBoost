num_filters = 8
filter_size = 3
pool_size = 2
input_image_shape = (32, 32, 3)
output_class_size = 10
client_size = 3
client_epochs = 100
server_epochs = 2
v = 1 / server_epochs

RANDOM = "random"
AVERAGE = "average"
AVERAGE_OUTPUT = "average_output"
GDBOOST = "gdboost"
