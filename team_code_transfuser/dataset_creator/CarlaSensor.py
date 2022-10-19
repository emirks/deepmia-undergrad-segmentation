import sys

from team_code_transfuser import dataset_creator
try:
    sys.path.append(dataset_creator.carla_egg_path)
except IndexError:
    pass
import carla
import numpy as np


class Sensor: 
    def __init__(self, sensor_attr):
        self.sensor_name = sensor_attr["name"].split(".")[-1]
        self.sensor_attr = sensor_attr

        self.bp = self.blueprint_library.find(self.sensor_attr["name"])
        for attr in self.sensor_attr["carla_attr"]: 
            self.bp.set_attribute(attr, self.sensor_attr["carla_attr"][attr])
        self.spawn_point = carla.Transform(self.sensor_attr["attach_transform"])
        self.width = self.sensor_attr["carla_attr"]["image_size_x"]
        self.height = self.sensor_attr["carla_attr"]["image_size_y"]

        self.sensor = None
        self.idx = 0

    def create(self, world, vehicle): 
        self.sensor = world.spawn_actor(self.bp, self.spawn_point, attach_to=vehicle)

    def listen(self): 
        return self.sensor.listen

    def process_data(self, data): 
        data = np.frombuffer(data.rav_data, dtype=np.type("uint8"))
        data = np.reshape(data, (self.height, self.width, 4))
        if self.sensor_name == 'rgb': 
            data = data[:, :, :3] #taking out opacity channel
        elif self.sensor_name == 'depth':
            data = data.astype(np.float32)
            # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
            normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
            normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
            depth_meters = normalized_depth * 1000
            data = depth_meters
        elif self.sensor_name == 'semantic_segmentation': 
            data = data[:, :, 0] #taking out r channel -> semantic values are in the r channel
        return data

    def destroy(self): 
        self.sensor.destroy()