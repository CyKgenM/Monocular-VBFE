import pyrealsense2 as rs
import cv2
import numpy as np
import serial
import csv

pipe = rs.pipeline()
conf = rs.config()

conf.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
conf.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

points = rs.points()

profile = pipe.start(conf)

depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(
    rs.option.visual_preset, 3
)  # A mélység szenzor beállítása magas pontosságra
depth_scale = depth_sensor.get_depth_scale()
colorizer = rs.colorizer()

im_count = 0
key = cv2.waitKey(1)

# Soros kommunikáció beállítása
ser = serial.Serial('COM3', 9600) 

csv_file_path = "data\\labels.csv"

try:
    
    frames = pipe.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    colorized = colorizer.process(frames)

    # Színes kép konvertálása Numpy array adattípusra
    color_image = np.asanyarray(color.get_data())

    # Kép elmentáse OpenCV használatával
    cv2.imwrite(f"data\\images\\frame_{im_count}.png", color_image)

    print(f"Image saved as 'frame_{im_count}.png' in 'images' folder.")

    # CSV fájl megnyítása az adat hozzáadására 
    with open(csv_file_path, mode='a', newline='') as csvfile:
        # CSV objektum definiálása
        csv_writer = csv.writer(csvfile)

        # Adat beolvasás a soros portról
        serial_data = ser.readline().decode('utf-8').strip()
                
        if serial_data:
            data = serial_data
            image_name = f"frame_{im_count}"
            contact = 0 if data < 0.01 else 1
                    
            csv_writer.writerow([image_name, data, contact])

        # Pontfelhő elmentése .ply kiterjesztésben
        ply = rs.save_to_ply(f"data\\pointclouds\\frame_{im_count}.ply")

        ply.set_option(rs.save_to_ply.option_ply_binary, False)
        ply.set_option(rs.save_to_ply.option_ply_normals, True)

        ply.process(colorized)

        print(f"Depth image saved as 'frame_{im_count}.ply' in 'pointclouds' folder.")

        im_count += 1

except KeyboardInterrupt:
    exit()

finally:
    # Folyamatlánc leállítása
    pipe.stop()