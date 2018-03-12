# dataset
dataset can be downloaded from http://yann.lecun.com/exdb/mnist/ the program will read from a folder called `dataset`

# additional packages
tensorflow `pip3 install tensorflow`

python-mnist `pip3 install python-mnist`

# result
2 layer ANN with L2 soft max loss on Adamax optimizer

28 x 28 fc -> 10 x 10 c -> 5 x 5 fc -> 10

test accuracy: 0.9778

2 layer CNN with ce loss on Adamax optimizer

5 x 5 x 32 conv -> max pool 2 x 2 -> 5 x 5 x 64 conv2 -> 1024 fc -> drop out 50% -> 10

test accuracy: 0.9813
