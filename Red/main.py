import random
import time
from pytesseract import pytesseract
from pyboy import PyBoy
# import tensorflow as tf

pyboy = PyBoy('red.gb')
pyboy.set_emulation_speed(0)

with open("state_file.state", "rb") as f:
    pyboy.load_state(f)

time_train = 60*60*60

outputs = ["a", "b", "start", "select", "left", "right", "up", "down"]

red = pyboy.game_wrapper
red.start_game()

count = 0

random.seed(time.process_time())

while count < time_train:
    pyboy.tick(24)
    image = pyboy.screen.image

    resize = image.resize((40, 36))
    # image.save("Training Frames/" + "frame" + str(count) + ".png")
    # resize.save("Training Frames/" + "frame" + str(count) + ".png")
    input = random.randint(0, 7)
    print(input)
    pyboy.button(outputs[input], 12)
    count = count + 1


# while pyboy.tick():
#     pass

# print("Saving")
# with open("state_file.state", "wb") as f:
#     pyboy.save_state(f)
# print("Saved!")

pyboy.stop()
