import argparse
import os
import sys

from carla_settings import carla_egg_path
try:
    sys.path.append(carla_egg_path)
except IndexError:
    pass
import carla
import h5py
import cv2
import numpy as np

from CarlaWorld import CarlaWorld
from HDF5Saver import HDF5Saver

SEM_COLORS = {
    0 : (0, 0, 0),
    1 : (70, 70, 70),
    2 : (100, 40, 40),
    3 : (55, 90, 80),
    4 : (220, 20, 60), #pedestrian
    5 : (153, 153, 153), #pole
    6 : (157, 234, 50), #road line
    7 : (128, 64, 128), #road
    8 : (244, 35, 232),  #side walk
    9 : (107, 142, 35),
    10: (0, 0, 142), #vehicles
    11 : (102, 102, 156),
    12 : (220, 220, 0),
    13 : (70, 130, 180),
    14 : (81, 0, 81),
    15 : (150, 100, 100),
    16 : (230, 150, 140),
    17 : (180, 165, 180),
    18 : (250, 170, 30), #traffic-light
    19 : (110, 190, 160),
    20 : (170, 120, 50),
    21 : (45, 60, 150),
    22 : (145, 170, 100)
}

def visualize_semantic(sem, labels=[4,5,6,7,10,18]):
    canvas = np.zeros(sem.shape+(3,), dtype=np.uint8)
    for label in SEM_COLORS.keys():
        canvas[sem==label] = SEM_COLORS[label]

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
    parser.add_argument('-wi', '--width', default=1024, type=int, help="sensor widths in pixels")
    parser.add_argument('-he', '--height', default=768, type=int, help="sensor heights in pixels")
    parser.add_argument('-ve', '--vehicles', default=100, type=int, help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=150, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('--no-rendering',action='store_true', help='disable rendering')
    parser.add_argument('-v', '--video', action="store_true", help="record a mp4 video on top of the recorded hdf5 file")
    parser.add_argument('-d', '--depth', action='store_true', help="show the depth video side by side with the rgb")
    args = parser.parse_args()
    assert(args.hdf5_file is not None)
    assert(args.width > 0 and args.height > 0)

    # sensor_setup
    rgb_camera = {
        "name" : 'sensor.camera.rgb',
        "carla_attr" : {
            'image_size_x': str(args.width),
            'image_size_y': str(args.height),
            'fov': str(90)
        },
        "attach_transform" : carla.Transform(carla.Location(x=1.0, z=2.0))
    }
    semantic_camera = {
        "name" : 'sensor.camera.semantic_segmentation',
        "carla_attr" : {
            'image_size_x': str(args.width),
            'image_size_y': str(args.height),
            'fov': str(90)
        },
        "attach_transform" : carla.Transform(carla.Location(x=1.0, z=2.0))
    }
    sensor_attrs = [rgb_camera, semantic_camera]

    # Beginning data capture proccedure
    save_path = "/home/transfuser/autonomous_car/transfuser-erkam/semantic-segmentation-dataset"
    HDF5_file = HDF5Saver(os.path.join(save_path, args.hdf5_file + ".hdf5"))
    print("HDF5 File opened")
    carla_world = CarlaWorld(HDF5_file, args.no_rendering)

    timestamps = []
    egos_to_run = 6
    print('Starting to record data...')
    carla_world.spawn_npcs(number_of_vehicles=args.vehicles, number_of_walkers=args.walkers)
    for weather_option in carla_world.weather_options:
        carla_world.set_weather(weather_option)
        ego_vehicle_iteration = 0
        while ego_vehicle_iteration < egos_to_run:
            carla_world.begin_data_acquisition(sensor_attrs=sensor_attrs, frames_to_record_one_ego=2, 
                                            timestamps=timestamps, egos_to_run=egos_to_run)
            print('Setting another vehicle as EGO.')
            ego_vehicle_iteration += 1

    carla_world.remove_npcs()
    print('Finished simulation.')
    print('Saving timestamps...')
    carla_world.hdf5_file.record_all_timestamps(timestamps)
    HDF5_file.close_HDF5()

    # For later visualization
    if args.video:
        create_video_sample(os.path.join(save_path, args.hdf5_file + ".hdf5"), args.width, args.height)