import argparse
import os
import sys

from carla_settings import carla_egg_path, team_code_path
try:
    sys.path.append(carla_egg_path)
except IndexError:
    pass
import carla
import h5py
import cv2
import numpy as np

sys.path.append(team_code_path)
import team_code_transfuser.segmentation.config as config
from CarlaWorld import CarlaWorld
from HDF5Saver import HDF5Saver

def visualize_semantic(sem, labels=[4,5,6,7,10,18]):
    canvas = np.zeros(sem.shape+(3,), dtype=np.uint8)
    for label in config.SEM_COLORS.keys():
        canvas[sem==label] = config.SEM_COLORS[label]

    return canvas

def create_video_sample(hdf5_file, frame_width, frame_height):
    with h5py.File(hdf5_file, 'r') as file:
        frame_width = frame_width * 2
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (frame_width, frame_height))

        for time_idx, time in enumerate(file['timestamps']['timestamps']):
            rgb_data = np.array(file['rgb'][str(time)])
            semantic_data = np.array(file['semantic_segmentation'][str(time)])

            sys.stdout.write("\r")
            sys.stdout.write('Recording video. Frame {0}/{1}'.format(time_idx, len(file['timestamps']['timestamps'])))
            sys.stdout.flush()

            composed_frame = np.hstack((rgb_data, visualize_semantic(semantic_data)))
            cv2.putText(composed_frame, 'timestamp', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(composed_frame, str(time), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(composed_frame)

    print('\nDone.')



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Settings for the data capture", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('hdf5_file', default=None, type=str, help='name of hdf5 file to save the data')
    parser.add_argument('--town-index', default=0, type=int, help='index of the town to create dataset on')
    # total number of frames that will be generated from the script: 
    #   number_of_egos * frames_per_ego * 5
    parser.add_argument('-egos', '--number-of-egos', default=13, type=int, help="number of egos to run in the simulation")
    parser.add_argument('-fpe', '--frames-per-ego', default=15, type=int, help="number of egos to run in the simulation")
    parser.add_argument('-ve', '--vehicles', default=100, type=int, help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=150, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('-v', '--video', action="store_true", help="record a mp4 video on top of the recorded hdf5 file")
    parser.add_argument('-d', '--depth', action='store_true', help="show the depth video side by side with the rgb")
    args = parser.parse_args()
    assert(args.hdf5_file is not None)

    # sensor_setup
    sensor_attrs = []
    for i in range(len(config.camera_rots)): 
        sensor_transform = carla.Transform(
            carla.Location(x=config.camera_pos[0], y = config.camera_pos[1], z=config.camera_pos[2]),
            carla.Rotation(roll=config.camera_rots[i][0], pitch=config.camera_rots[i][1], yaw=config.camera_rots[i][2])
        )
        rgb_camera = {
            "id" : f"rgb_{i}",
            "name" : 'sensor.camera.rgb',
            "carla_attr" : {
                'image_size_x': str(config.camera_width),
                'image_size_y': str(config.camera_height),
                'fov': str(config.camera_fov)
            },
            "attach_transform" : sensor_transform
        }
        semantic_camera = {
            "id" : f"semantic_{i}",
            "name" : 'sensor.camera.semantic_segmentation',
            "carla_attr" : {
                'image_size_x': str(config.camera_width),
                'image_size_y': str(config.camera_height),
                'fov': str(config.camera_fov)
            },
            "attach_transform" : sensor_transform
        }
        sensor_attrs.append(rgb_camera)
        sensor_attrs.append(semantic_camera)
    lidar_transform = carla.Transform(
        carla.Location(x=config.lidar_pos[0], y = config.lidar_pos[1], z=config.lidar_pos[2]),
        carla.Rotation(roll=config.lidar_rot[0], pitch=config.lidar_rot[1], yaw=config.lidar_rot[2])
    )
    lidar = {
        'id': 'lidar',
        'name': 'sensor.lidar.ray_cast',
        "carla_attr" : {
            'channels': str(config.lidar_channels), 
            'range': str(config.lidar_range), 
            'points_per_second': str(config.lidar_points_per_second),
            'lower_fov': str(config.lidar_lower_fov), 
            'upper_fov': str(config.lidar_upper_fov)
        },
        'attach_transform' : lidar_transform
    }
    sensor_attrs.append(lidar)


    # Beginning data capture proccedure
    hdf5_file_path = f"{config.SAVE_DIR}/datasets/{args.hdf5_file}.hdf5"
    HDF5_file = HDF5Saver(hdf5_file_path, groups=sensor_attrs)
    print("HDF5 File opened")

    timestamps = []
    egos_to_run = args.number_of_egos
    print('Starting to record data...')
    town_options = carla.Client('localhost', 2000).get_available_maps()
    carla_world = CarlaWorld(town_options[args.town_index], HDF5_file)
    carla_world.spawn_npcs(number_of_vehicles=args.vehicles, number_of_walkers=args.walkers)
    for weather_option in carla_world.weather_options:
        carla_world.set_weather(weather_option)
        ego_vehicle_iteration = 0
        while ego_vehicle_iteration < egos_to_run:
            carla_world.begin_data_acquisition(sensor_attrs=sensor_attrs, frames_to_record_one_ego=args.frames_per_ego, 
                                                timestamps=timestamps, egos_to_run=egos_to_run)
            print('Setting another vehicle as EGO.')
            ego_vehicle_iteration += 1
    carla_world.remove_npcs()
    print('Saving timestamps...')
    carla_world.hdf5_file.record_all_timestamps(timestamps)
    print('Finished simulation.')
    HDF5_file.close_HDF5()

    # For later visualization
    if args.video:
        create_video_sample(hdf5_file_path, args.width, args.height)