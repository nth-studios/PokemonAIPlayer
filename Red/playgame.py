import random
import time
from collections import Counter

from pytesseract import pytesseract
from pyboy import PyBoy
import tensorflow as tf
import numpy as np
import math



pyboy = PyBoy('red.gb')
pyboy.set_emulation_speed(0)

with open("state_file.state", "rb") as f:
    pyboy.load_state(f)

while pyboy.tick():
    pass

print("Saving")
with open("state_file.state", "wb") as f:
    pyboy.save_state(f)
print("Saved!")


