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


def process_thermal_data(thermal_data, target_resolution):
    # Rescale thermische data naar de gewenste resolutie
    thermal_rescaled = cv2.resize(thermal_data, target_resolution, interpolation=cv2.INTER_CUBIC)

    # Voeg een colormap toe om grijswaarden naar RGB om te zetten
    thermal_colored = cv2.applyColorMap(
        cv2.convertScaleAbs(thermal_rescaled, alpha=255 / thermal_rescaled.max()),
        cv2.COLORMAP_JET
    )

    # Corrigeer de oriëntatie door het beeld 90 graden tegen de klok in te draaien
    thermal_colored = cv2.rotate(thermal_colored, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return thermal_colored


def combine_frames(camera_frame, thermal_frame, alpha=0.6, offset_x=0, offset_y=0):
    # Debugging: Controleer de vormen van de frames
    print(f"Camera frame shape: {camera_frame.shape}")
    print(f"Thermal frame shape: {thermal_frame.shape}")

    # Converteer camera_frame naar RGB als het RGBA is
    if camera_frame.shape[2] == 4:  # RGBA naar RGB
        camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGRA2BGR)
        print("Camera frame geconverteerd van RGBA naar RGB.")

    # Bereken de ROI (Region of Interest) voor het overlay
    y_start = max(0, offset_y)
    y_end = min(camera_frame.shape[0], offset_y + thermal_frame.shape[0])
    x_start = max(0, offset_x)
    x_end = min(camera_frame.shape[1], offset_x + thermal_frame.shape[1])

    # Controleer of de ROI binnen de cameraframe-grenzen valt
    if y_start >= y_end or x_start >= x_end:
        print("ROI valt buiten de camera-afmetingen. Overlay wordt overgeslagen.")
        return camera_frame

    # Selecteer de relevante delen van het thermische beeld en camera
    thermal_cropped = thermal_frame[max(0, -offset_y):max(0, -offset_y) + (y_end - y_start),
                                    max(0, -offset_x):max(0, -offset_x) + (x_end - x_start)]

    # Controleer of de dimensies van de cropped thermal overeenkomen
    if thermal_cropped.shape[:2] != (y_end - y_start, x_end - x_start):
        print("Thermal cropped dimensies komen niet overeen met camera ROI.")
        return camera_frame

    # Combineer alleen de relevante delen met het camerabeeld
    combined_frame = camera_frame.copy()
    combined_frame[y_start:y_end, x_start:x_end] = cv2.addWeighted(
        thermal_cropped, alpha,
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
    offset_x, offset_y = 0, -370  # Pas aan om de positie van de overlay te bepalen
    target_resolution = (960, 720)  # Maak het thermische frame geschikt voor het camera frame

    try:
        while True:
            # Haal cameraframe op
            camera_frame = picam2.capture_array()
            camera_frame = cv2.flip(camera_frame, 1)

            # Debugging: Controleer het cameraframe
            print(f"Camera frame shape: {camera_frame.shape}")

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
                alpha=0.6, offset_x=offset_x, offset_y=offset_y
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
