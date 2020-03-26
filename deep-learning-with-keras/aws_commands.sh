# Connecting to the instance
ssh -i "deep-learning-ubuntu.pem" ubuntu@ec2-18-217-32-215.us-east-2.compute.amazonaws.com

# Connecting to jupyter notebook
https://localhost:8888/token

# Copy data from local computer to amazon instance
scp -i ~/deep-learning-ubuntu.pem ~/Downloads/train/*  ubuntu@ec2-18-217-32-215.us-east-2.compute.amazonaws.com:~/data/

# Copy data from amazon instance to local computer
scp -i ~/deep-learning-open.pem ubuntu@ec2-54-186-112-245.us-west-2.compute.amazonaws.com:cats_and_dogs_small_data_aug.h5 ~/Download/
