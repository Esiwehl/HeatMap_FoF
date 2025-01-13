import board
import busio
import numpy as np
import cv2
import adafruit_mlx90640
import time

# Initialize I2C bus at 1 MHz
i2c = busio.I2C(board.SCL, board.SDA)
mlx = adafruit_mlx90640.MLX90640(i2c)

# Set refresh rate to 8 Hz
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

# Initialize storage for thermal data
frame = np.zeros((24 * 32,))

try:
    while True:
        start_time = time.time()
        try:
            mlx.getFrame(frame)
        except ValueError:
            continue  # Retry on error

        # Reshape and normalize
        thermal_data_2d = np.reshape(frame, (24, 32))
        normalized = ((thermal_data_2d - np.min(thermal_data_2d)) / 
                      (np.max(thermal_data_2d) - np.min(thermal_data_2d)) * 255).astype(np.uint8)

        # Display using OpenCV
        upscale_factor = 20
        upscaled = cv2.resize(normalized, (32 * upscale_factor, 24 * upscale_factor), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap(upscaled, cv2.COLORMAP_INFERNO)

        flipped = cv2.flip(colored, 1)
        cv2.imshow("Thermal Image", flipped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Calculate and display frame rate
        elapsed_time = time.time() - start_time
        print(f"FPS: {1 / elapsed_time:.2f}")
        time.sleep(max(0, 1 / 8 - elapsed_time))

finally:
    cv2.destroyAllWindows()

