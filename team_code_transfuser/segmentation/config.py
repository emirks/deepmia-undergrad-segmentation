# labels=[1,4,6,7,8,10,18]
labels=[*range(20)]

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

scale = 1 # image pre-processing
img_resolution = (160, 704) # image pre-processing in H, W
img_width = 320 # important this should be consistent with scale, e.g. scale = 1, img_width 320, scale=2, image_width 640
img_height = 160

camera_pos = [1.3, 0.0, 2.3] #x, y, z mounting position of the camera
camera_width = 960 # Camera width in pixel
camera_height = 480 # Camera height in pixel
camera_fov = 120 #Camera FOV in degree
camera_rot_0 = [0.0, 0.0, 0.0] # Roll Pitch Yaw of camera 0 in degree
camera_rot_1 = [0.0, 0.0, -60.0] # Roll Pitch Yaw of camera 1 in degree
camera_rot_2 = [0.0, 0.0, 60.0] # Roll Pitch Yaw of camera 2 in degree
camera_rots = [camera_rot_0, camera_rot_1, camera_rot_2]

towns = ["town-1", "town-2", "town-3", "town-4", "town-5", "town-6", "town-7"]