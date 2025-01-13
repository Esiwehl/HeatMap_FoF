import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
import scipy.ndimage

# Setup I2C and MLX90640
i2c = busio.I2C(board.SCL, board.SDA)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ  # Anything Higher than this RPi5 does not like.

# Set up plot for live thermal image
plt.ion()
fig, ax = plt.subplots(figsize=(12, 7))
therm1 = ax.imshow(np.zeros((24, 32)), vmin=0, vmax=370, interpolation='bilinear', cmap='plasma')
cbar = fig.colorbar(therm1)
cbar.set_label('Temperature [$^{\circ}$C]', fontsize=14)

frame = np.zeros((24*32,))
t_array = []
max_retries = 5

# Function to downsample image (optional)
def downsample_image(data_array, block_size=2):
    shape = (data_array.shape[0] // block_size, data_array.shape[1] // block_size)
    downsampled = data_array[:shape[0] * block_size, :shape[1] * block_size].reshape(
        shape[0], block_size, shape[1], block_size
    ).mean(axis=(1, 3))
    return downsampled

while True:
    t1 = time.monotonic()
    retry_count = 0
    while retry_count < max_retries:
        try:
            mlx.getFrame(frame)
            data_array = np.reshape(frame, (24, 32))
            
            # Optionally downsample and/or interpolate
            upsampled_data = scipy.ndimage.zoom(data_array, (4, 4), order=3)  # Upsample using cubic interpolation
            # downsampled_data = downsample_image(data_array)  # Uncomment this line for downsampling instead
            
            # Update the plot with the smoother data
            therm1.set_data(np.fliplr(upsampled_data))  # Or use downsampled_data if using that option
            therm1.set_clim(vmin=np.min(upsampled_data), vmax=np.max(upsampled_data))
            fig.canvas.draw()  # Redraw the figure to update the plot and colorbar
            fig.canvas.flush_events()
            plt.pause(0.001)

            t_array.append(time.monotonic() - t1)
            print('Sample Rate: {0:2.1f}fps'.format(len(t_array) / np.sum(t_array)))
            break
        except ValueError:
            retry_count += 1
        except RuntimeError as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Failed after {max_retries} retries with error: {e}")
                exit()

