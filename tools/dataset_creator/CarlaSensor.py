import sys

from carla_settings import carla_egg_path
try:
    sys.path.append(carla_egg_path)
except IndexError:
    pass
import carla
import numpy as np
import cv2


class Sensor: 
    def __init__(self, sensor_attr):
        self.sensor_name = sensor_attr["name"].split(".")[-1]
        self.sensor_attr = sensor_attr

        self.spawn_point = self.sensor_attr["attach_transform"]
        self.width = int(self.sensor_attr["carla_attr"]["image_size_x"])
        self.height = int(self.sensor_attr["carla_attr"]["image_size_y"])

        self.sensor = None

    def create(self, world, vehicle): 
        bp = world.get_blueprint_library().find(self.sensor_attr["name"])
        for attr in self.sensor_attr["carla_attr"]: 
            bp.set_attribute(attr, self.sensor_attr["carla_attr"][attr])
        self.sensor = world.spawn_actor(bp, self.spawn_point, attach_to=vehicle)

    def listen(self): 
        return self.sensor.listen

    def process_data(self, data): 
        frame = data.frame
        data = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        data = np.reshape(data, (self.height, self.width, 4))
        data = data[:, :, :3] #taking out opacity channel
        data = data[:, :, ::-1]
        if self.sensor_name == 'rgb': 
            pass
        elif self.sensor_name == 'depth':
            data = data.astype(np.float32)
            # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
            normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
            normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
            depth_meters = normalized_depth * 1000
            data = depth_meters
        elif self.sensor_name == 'semantic_segmentation': 
            data = data[:, :, 0] #only r channel -> semantic values are in the r channel
        save_path = "/home/transfuser/autonomous_car/transfuser-erkam/semantic-segmentation-dataset"
        if self.sensor_name == 'rgb': 
            cv2.imwrite(f"{save_path}/rgb/{frame}.jpg", data)
        elif self.sensor_name == 'semantic_segmentation': 
            cv2.imwrite(f"{save_path}/semantic/{frame}.jpg", data)
        
        return data

    def destroy(self): 
        self.sensor.destroy()