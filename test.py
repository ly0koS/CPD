import tensorflow as tf
from tensorflow import keras
from train import train

Model_Path="/home/ly0kos/Car/model/1/"

try:
    model=tf.saved_model.load(Model_Path)
except :
    print("load Model Error!\nTrying to train first!\n")
    train()



