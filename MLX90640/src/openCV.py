import board
import busio
import numpy as np
import cv2
import adafruit_mlx90640
import time

# Set up the I2C bus
i2c = busio.I2C(board.SCL, board.SDA)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ  # Do not exceed this, your RPI will thank me.

# Initialize storage for the 1D thermal data frame
frame = np.zeros((24 * 32,))  # Flattened array for mlx.getFrame()

colormap = {
    "COLORMAP_AUTUMN": cv2.COLORMAP_AUTUMN,
    "COLORMAP_BONE": cv2.COLORMAP_BONE,
    "COLORMAP_JET": cv2.COLORMAP_JET,
    "COLORMAP_WINTER": cv2.COLORMAP_WINTER,
    "COLORMAP_RAINBOW": cv2.COLORMAP_RAINBOW,
    "COLORMAP_OCEAN": cv2.COLORMAP_OCEAN,
    "COLORMAP_SUMMER": cv2.COLORMAP_SUMMER,
    "COLORMAP_SPRING": cv2.COLORMAP_SPRING,
    "COLORMAP_COOL": cv2.COLORMAP_COOL,
    "COLORMAP_HSV": cv2.COLORMAP_HSV,
    "COLORMAP_PINK": cv2.COLORMAP_PINK,
    "COLORMAP_HOT": cv2.COLORMAP_HOT,
    "COLORMAP_PARULA": cv2.COLORMAP_PARULA,
    "COLORMAP_MAGMA": cv2.COLORMAP_MAGMA,
    "COLORMAP_INFERNO": cv2.COLORMAP_INFERNO,
    "COLORMAP_PLASMA": cv2.COLORMAP_PLASMA,
    "COLORMAP_VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "COLORMAP_CIVIDIS": cv2.COLORMAP_CIVIDIS,
    "COLORMAP_TWILIGHT": cv2.COLORMAP_TWILIGHT,
    "COLORMAP_TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
    "COLORMAP_TURBO": cv2.COLORMAP_TURBO,
    "COLORMAP_DEEPGREEN": cv2.COLORMAP_DEEPGREEN,
}

colormap_keys = list(colormap.keys())
colormap_values = list(colormap.values())
colormap_idx = 0

try:
    while True:
        try:
            # Grab data from the sensor
            #print("blah")
            #print(f"Frame: {frame[:10]}")
            #print(f"Frame lenght: {len(frame)}")
            mlx.getFrame(frame)
            #print(f"Frame: {frame[:10]}")

            # Reshape into 2D (24x32)
            #print(f"Hierdan? {frame}")
            thermal_data_2d = np.reshape(frame, (24, 32))
            #print(f"Hierdan? {thermal_data_2d}")

            # Normalize temperature data to 8-bit range (0–255)
            #min_temp = 15
            #max_temp = 35
            min_temp = np.min(thermal_data_2d)
            max_temp = np.max(thermal_data_2d)
            normalized = ((thermal_data_2d - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)
            #print(f"Normalized: {normalized}")

            #Maybe only clahe instead of the whole shabam?
            #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
            #normalized = clahe.apply(normalized)


            # Upscale and apply colormap
            upscale_factor = 60
            upscaled = cv2.resize(normalized, (32 * upscale_factor, 24 * upscale_factor), interpolation=cv2.INTER_CUBIC) #Think INTER_CUBIC was better tbh
            #thermal_data_2d = cv2.fastNlMeansDenoising(np.uint8(thermal_data_2d), h=10)
            colored = cv2.applyColorMap(upscaled, colormap_values[colormap_idx])
            #thermal_data_2d = cv2.fastNlMeansDenoising(np.uint8(thermal_data_2d), h=5)
            thermal_data_2d = cv2.fastNlMeansDenoising(np.uint8(colored), h=5)

            # We can also mess around with this one I suppose?
            #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            #colored = cv2.filter2D(colored, -1, kernel)

            #Resizig window
            colormap_name = colormap_keys[colormap_idx]
            cv2.namedWindow("Thermal Image", cv2.WINDOW_NORMAL)
            cv2.setWindowTitle("Thermal Image", f"Colormap: {colormap_name}")
            cv2.resizeWindow("Thermal Image", 1060, 720)
            # Display the image
            rotated = cv2.rotate(colored, cv2.ROTATE_90_CLOCKWISE)
            flipped = cv2.flip(rotated, 0)
            cv2.imshow("Thermal Image", flipped)

            # Break on 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                colormap_idx = (colormap_idx -1) % len(colormap)
            elif key == ord('x'):
                colormap_idx = (colormap_idx +1) % len(colormap)

        except ValueError as ve:
            print(f"ValueError: retrying... {ve}")
        except RuntimeError as e:
            print(f"RuntimeError: {e}, retrying...")
            time.sleep(0.1)  # Short delay before retrying

finally:
    cv2.destroyAllWindows()

#Emissivity Setting: Set the correct emissivity for your target surface (e.g., 0.95 for human skin).
