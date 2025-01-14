import cv2
import numpy as np
import board
import busio
import adafruit_mlx90640

from picamera2 import Picamera2

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


def process_thermal_data(thermal_data, target_resolution):
    # Rescale thermische data naar de gewenste resolutie
    thermal_rescaled = cv2.resize(thermal_data, target_resolution, interpolation=cv2.INTER_CUBIC)

    # Voeg een colormap toe om grijswaarden naar RGB om te zetten
    thermal_colored = cv2.applyColorMap(
        cv2.convertScaleAbs(thermal_rescaled, alpha=255 / thermal_rescaled.max()),
        cv2.COLORMAP_JET,
    )
	
    # Corrigeer de oriëntatie door het beeld 90 graden tegen de klok in te draaien
    thermal_colored = cv2.rotate(thermal_colored, cv2.ROTATE_90_COUNTERCLOCKWISE)
	
    return thermal_colored


def main():
    # Initieer de camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()

    # Doelresolutie (camera-outputresolutie)
    camera_resolution = (1536, 864)

    # Initialiseer variabelen voor positie en schaal
    scale_factor = 1.0
    offset_x, offset_y = 0, 0

    try:
        while True:
            # Haal cameraframe op
            camera_frame = picam2.capture_array()

            # Flip het camerabeeld horizontaal
            camera_frame = cv2.flip(camera_frame, 1)

            # Converteer camera_frame van 4 kanalen (RGBA) naar 3 kanalen (RGB)
            if camera_frame.shape[2] == 4:  # Check of het 4 kanalen heeft
                camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGRA2BGR)

            # Haal thermische data op
            thermal_data = get_mlx90640_data()
            if thermal_data is None:
                print("Fout bij ophalen thermische data, overslaan...")
                continue

            # Verwerk thermische data
            thermal_colored = process_thermal_data(
                thermal_data, 
                (int(camera_resolution[0] * scale_factor), int(camera_resolution[1] * scale_factor))
            )

            # Zorg ervoor dat de overlay correct wordt geplaatst met offset
            overlay = np.zeros_like(camera_frame)
            h, w, _ = thermal_colored.shape
            y_start = max(0, offset_y)
            y_end = min(camera_frame.shape[0], offset_y + h)
            x_start = max(0, offset_x)
            x_end = min(camera_frame.shape[1], offset_x + w)

            # Selecteer het relevante deel van de overlay
            overlay[y_start:y_end, x_start:x_end] = thermal_colored[
                max(0, -offset_y) : max(0, -offset_y) + (y_end - y_start),
                max(0, -offset_x) : max(0, -offset_x) + (x_end - x_start),
            ]

            # Combineer de frames
            combined_frame = cv2.addWeighted(overlay, 0.6, camera_frame, 0.4, 0)

            # Toon het gecombineerde beeld
            cv2.imshow("Camera met Thermische Overlay", combined_frame)

            # Lees toetsenbordinvoer voor aanpassingen
            key = cv2.waitKey(1) & 0xFF
            if key == ord("w"):  # Verplaats omhoog
                offset_y -= 10
            elif key == ord("s"):  # Verplaats omlaag
                offset_y += 10
            elif key == ord("a"):  # Verplaats naar links
                offset_x -= 10
            elif key == ord("d"):  # Verplaats naar rechts
                offset_x += 10
            elif key == ord("z"):  # Zoom uit
                scale_factor = max(0.5, scale_factor - 0.1)
            elif key == ord("x"):  # Zoom in
                scale_factor = min(2.0, scale_factor + 0.1)
            elif key == ord("q"):  # Stop het script
                break

            # Druk huidige positie en schaal af
            print(f"Offset: ({offset_x}, {offset_y}), Scale: {scale_factor:.2f}")
    finally:
        # Zorg ervoor dat de camera netjes stopt
        picam2.stop()
        cv2.destroyAllWindows()

#Offset: (120, -340), Scale: 0.60

if __name__ == "__main__":
    main()
