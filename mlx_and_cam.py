import cv2
import numpy as np
from picamera2 import Picamera2

import board
import busio
import adafruit_mlx90640


def get_mlx90640_data():
    # Initialiseer de MLX90640
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

    # Maak opslag voor framegegevens
    frame = np.zeros((24 * 32,), dtype=np.float32)

    try:
        mlx.getFrame(frame)  # Haal een frame van de sensor
        thermal_data = np.reshape(frame, (24, 32))  # 2D array
        return thermal_data
    except Exception as e:
        print(f"Fout bij ophalen MLX90640 data: {e}")
        return None


def process_thermal_data(thermal_data, target_resolution, stretch_factor=1.0):
    # Rescale thermische data naar de gewenste resolutie
    stretched_width = int(target_resolution[0] * stretch_factor)
    thermal_rescaled = cv2.resize(thermal_data, (stretched_width, target_resolution[1]), interpolation=cv2.INTER_CUBIC)

    # Voeg een colormap toe om grijswaarden naar RGB om te zetten
    thermal_colored = cv2.applyColorMap(
        cv2.convertScaleAbs(thermal_rescaled, alpha=255 / thermal_rescaled.max()),
        cv2.COLORMAP_JET
    )

    # Corrigeer de oriëntatie door het beeld 90 graden tegen de klok in te draaien
    thermal_colored = cv2.rotate(thermal_colored, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return thermal_colored


def combine_frames(camera_frame, thermal_frame, alpha=0.6, offset_x=0, offset_y=0, scale_factor=1.0):
    # Debugging: Controleer de vormen van de frames
    print(f"Camera frame shape: {camera_frame.shape}")
    print(f"Thermal frame shape: {thermal_frame.shape}")

    # Converteer camera_frame naar RGB als het RGBA is
    if camera_frame.shape[2] == 4:  # RGBA naar RGB
        camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGRA2BGR)
        print("Camera frame geconverteerd van RGBA naar RGB.")

    # Schaal het thermische beeld
    scaled_width = int(thermal_frame.shape[1] * scale_factor)
    scaled_height = int(thermal_frame.shape[0] * scale_factor)
    thermal_frame = cv2.resize(thermal_frame, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)

    # Bereken de ROI (Region of Interest) voor het overlay
    y_start = max(0, offset_y)
    y_end = min(camera_frame.shape[0], offset_y + scaled_height)
    x_start = max(0, offset_x)
    x_end = min(camera_frame.shape[1], offset_x + scaled_width)

    # Selecteer de relevante delen van het thermische beeld en camera
    thermal_cropped = thermal_frame[max(0, -offset_y):max(0, -offset_y) + (y_end - y_start),
                                    max(0, -offset_x):max(0, -offset_x) + (x_end - x_start)]

    # Maak een masker voor niet-groene gebieden
    lower_green = np.array([0, 100, 0])  # Onderste drempel voor groen
    upper_green = np.array([100, 255, 100])  # Bovenste drempel voor groen
    mask_green = cv2.inRange(thermal_cropped, lower_green, upper_green)

    # Inverteer het masker zodat niet-groene gebieden worden behouden
    mask_not_green = cv2.bitwise_not(mask_green)

    # Maak het thermische beeld alleen met niet-groene delen
    thermal_no_green = cv2.bitwise_and(thermal_cropped, thermal_cropped, mask=mask_not_green)

    # Combineer alleen de niet-groene delen met het camerabeeld
    combined_frame = camera_frame.copy()
    combined_frame[y_start:y_end, x_start:x_end] = cv2.addWeighted(
        thermal_no_green, alpha,
        camera_frame[y_start:y_end, x_start:x_end], 1 - alpha,
        0
    )

    return combined_frame


def main():
    # Initieer de camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()

    # Stel de offset en schaal in
    offset_x, offset_y = 120, -340  # Pas aan op basis van eerdere waarden
    scale_factor = 0.6
    #stretch_factor = 1.2

    # Doelresolutie (camera-outputresolutie)
    target_resolution = (1536, 864)

    try:
        while True:
            # Haal cameraframe op
            camera_frame = picam2.capture_array()
            camera_frame = cv2.flip(camera_frame, 1)

            # Haal thermische data op
            thermal_data = get_mlx90640_data()
            if thermal_data is None:
                print("Fout bij ophalen thermische data, overslaan...")
                continue

            # Verwerk thermische data
            thermal_colored = process_thermal_data(thermal_data, target_resolution)

            # Combineer camerabeeld en thermisch beeld
            combined_frame = combine_frames(
                camera_frame, thermal_colored,
                alpha=0.6, offset_x=offset_x, offset_y=offset_y, scale_factor=scale_factor
            )

            # Toon het gecombineerde beeld
            cv2.imshow("Camera met Thermische Overlay", combined_frame)

            # Sluit af bij 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Zorg ervoor dat de camera netjes stopt
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
