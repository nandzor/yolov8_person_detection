# config.py
import os

FACES_DIR = "faces"
HAAR_CASCADE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'haarcascade_frontalface_default.xml')
API_ENDPOINT = "https://httpbin.org/post"
MAX_DISAPPEARED_FRAMES = 50
MAX_DISTANCE = 60
CONFIRMATION_FRAMES_THRESHOLD = 10
