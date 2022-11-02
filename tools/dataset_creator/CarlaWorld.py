import sys

from carla_settings import carla_egg_path
try:
    sys.path.append(carla_egg_path)
except IndexError:
    pass
import carla
import random
import time

from WeatherSelector import WeatherSelector
from CarlaNPC import CarlaNPC
from CarlaSensor import Sensor
from CarlaSyncMode import CarlaSyncMode
from HDF5Saver import HDF5Saver

class CarlaWorld: 
    def __init__(self, town, hdf5_file):
        self.hdf5_file = hdf5_file
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world(town)
        self.blueprint_library = self.world.get_blueprint_library()
        print('Successfully connected to CARLA')

        self.weather_options = WeatherSelector().get_weather_options()
        self.total_recorded_frames = 0

        self.sensor_list = []

        self.simulation_new_started = True
        self.frame_interval = 10    

    def set_weather(self, weather_option): 
        weather = carla.WeatherParameters(*weather_option)
        self.world.set_weather(weather)
        print("Weather changed succesfully.")
    
    def spawn_npcs(self, number_of_vehicles, number_of_walkers): 
        self.NPC = CarlaNPC(self.client)
        self.vehicles, _ = self.NPC.create_npcs(number_of_vehicles, number_of_walkers)

    def remove_npcs(self): 
        print("Destroying actors...")
        self.NPC.remove_npcs()
        print("Done destroying actors")

    def add_sensor_to_vehicle(self, vehicle, sensor_attr):
        sensor = Sensor(sensor_attr)
        sensor.create(self.world, vehicle)
        self.sensor_list.append(sensor) 
        return sensor
        
    def remove_sensors(self): 
        for sensor in self.sensor_list: 
            sensor.destroy()
        self.sensor_list = []

    def begin_data_acquisition(self, sensor_attrs, frames_to_record_one_ego=1, timestamps=[], egos_to_run=10): 
        current_ego_recorded_frames = 0
        ego_vehicle = random.choice([x for x in self.world.get_actors().filter("vehicle.*") if x.type_id not in
                    ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.volkswagen.t2']])
        for sensor_attr in sensor_attrs: 
            self.add_sensor_to_vehicle(ego_vehicle, sensor_attr)
        
        with CarlaSyncMode(self.world, self.sensor_list, fps=30) as sync_mode: 
            # Skip initial frames where the car is being put on the ambient
            if self.simulation_new_started:
                for _ in range(30):
                    sync_mode.tick_no_data()
            
            while True: 
                if current_ego_recorded_frames == frames_to_record_one_ego: 
                    print('\n')
                    self.remove_sensors()
                    return
                
                # Skip every nth frame for data recording, so that one frame is not that similar to another
                wait_frame_ticks = 0
                while wait_frame_ticks < self.frame_interval:
                    sync_mode.tick_no_data()
                    wait_frame_ticks += 1
                
                # first data of tick is world data
                sensor_datas = sync_mode.tick(timeout=2.0)[1:]

                # Process raw data and save them into the HDF5 dataset file
                timestamp = round(time.time() * 1000.0)
                for i in range(len(self.sensor_list)):
                    sensor = self.sensor_list[i]
                    processed_sensor_data = sensor.process_data(sensor_datas[i]) 
                    self.hdf5_file.record_data(sensor.sensor_name, processed_sensor_data, timestamp)
                
                current_ego_recorded_frames += 1
                self.total_recorded_frames += 1
                timestamps.append(timestamp)

                sys.stdout.write("\r")
                sys.stdout.write('Frame {0}/{1}'.format(
                    self.total_recorded_frames, frames_to_record_one_ego*egos_to_run*len(self.weather_options)))
                sys.stdout.flush()



            
