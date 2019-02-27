import time
from pynput.keyboard import Key, Controller


keyboard = Controller()

class States():
	obstacles = []
	nearest_obstacle = (600,150) #distance
	game_over =  False

class Actions():

	def jump():
		keyboard.press(Key.space)
		time.sleep(0.05)
		keyboard.release(Key.space)

	def crouching():
		keyboard.press(Key.down)
		time.sleep(0.03)
		keyboard.release(Key.down)

class Reawards():
	pass


def worker(input_q):
	while True:
		if input_q.empty():
			pass
		else:
			frame = input_q.get()
			cv2.imshow('Captured', frame)
			cv2.waitKey(1)
