from pyboy import PyBoy

pyboy = PyBoy('red.gb')

with open("state_file.state", "rb") as f:
    pyboy.load_state(f)


while pyboy.tick():
    print("playing")
    pass

# print("Saving")
# with open("state_file.state", "wb") as f:
#     pyboy.save_state(f)
# print("Saved!")

pyboy.stop()