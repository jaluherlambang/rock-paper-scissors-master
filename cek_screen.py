import cv2
from screeninfo import get_monitors  # Install using: pip install screeninfo

# Ambil ukuran layar pertama (biasanya utama) jika tersedia
monitors = get_monitors()
if monitors:
    print('ada monitor')
    screen_width = monitors[0].width
    screen_height = monitors[0].height
else:
    ('tdk dapat diambil kembali ke default')
    # Jika ukuran layar tidak dapat diambil, gunakan ukuran default
    screen_width = 1920  # Ganti dengan lebar layar Anda
    screen_height = 1080  # Ganti dengan tinggi layar Anda

print(screen_height)
print(screen_width)
