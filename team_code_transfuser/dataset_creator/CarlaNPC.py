import glob
import os
import sys

from team_code_transfuser import dataset_creator
try:
    sys.path.append(dataset_creator.carla_egg_path)
except IndexError:
    pass
import carla
import logging
import random

class CarlaNPC: 
    def __init__(self, world) -> None:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.world = world
    
    def create_npcs(self, number_of_vehicles=150, number_of_walkers=70): 
        vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        walker_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # if number of vehicles is higher than spawn points, we should decrease number of vehicles
        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points
        
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles
        batch = []
        for n, transform in enumerate(spawn_points): 
            if n >= number_of_vehicles: 
                break
            blueprint = random.choice(vehicle_blueprints)
            # Taking out bicycles and motorcycles, since the semantic/bb labeling for that is mixed with pedestrian
            if int(blueprint.get_attribute('number_of_wheels')) > 2:
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
        
        for response in self.client.apply_batch_sync(batch): 
            if response.error: 
                logging.error(response.error)
            else: 
                self.vehicles_list.append(response.actor_id)
        
        # Spawn Walkers
        spawn_points = []
        for i in range(number_of_walkers): 
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc != None: 
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        
        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_blueprints)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
        
        batch = []
        for i in range(self.walkers_list): 
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)): 
            if results[i].error: 
                logging.error(results[i].error)
            else: 
                self.walkers_list[i]["con"] = results[i].actor_id
        
        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.tick()
        self.world.wait_for_tick()

        for i in range(len(self.walkers_list)): 
            walker_controller = self.world.get_actor(self.walkers_list[i]["con"])
            self.all_id += [self.walkers_list[i]["id"], self.walkers_list[i]["con"]]

            walker_controller.start()
            walker_controller.go_to_location(self.world.get_random_location_from_navigation())
            walker_controller.set_max_speed(1 + random.random()/2)

            
        print(f"Spawned {len(self.vehicles_list)} vehicles and {len(self.walkers_list)} walkers")
        return self.vehicles_list, self.walkers_list
    
    def remove_npcs(self): 
        print('Destroying %d NPC vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        for i in range(0, len(self.walkers_list)):
            walker_controller = self.world.get_actor(self.walkers_list[i]["con"])
            walker_controller.stop()

        print('Destroying %d NPC walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
