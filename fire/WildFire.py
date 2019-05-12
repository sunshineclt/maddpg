"""
# *******************<><><><><>**************************
# * Script for Wildfire Environment Python  Translation *
# *******************<><><><><>**************************
#
# This script and all its dependencies are implemented by: Esmaeil Seraj
#   - Esmaeil Seraj, CORE Robotics Lab, Robotics & Intelligent Machines,
#   Georgia Tech, Atlanta, GA, USA
#   - email <eseraj3@gatech.edu>
#   - website <https://github.gatech.edu/MCG-Lab/DistributedControl>
#
# Published under GNU GENERAL PUBLIC LICENSE ver. 3 (or any later version)
#
"""

import numpy as np


# wildfire simulation
class WildFire(object):

    def __init__(self, terrain_sizes, hotspot_areas, num_ign_points, duration):
        self.terrain_sizes = [int(terrain_sizes[0]), int(terrain_sizes[1])]  # sizes of the terrain
        self.initial_terrain_map = np.zeros(shape=self.terrain_sizes)  # initializing the terrain
        self.hotspot_areas = hotspot_areas  # format:: [[x_min, x_max, y_min, y_max]]
        self.num_ign_points = num_ign_points  # number of fire-spots in each area
        self.duration = duration  # total runtime steps

    # initializing hotspots
    def hotspot_init(self):
        """
        This function generates the initial hotspot areas

        :return: ignition points across the entire map
        """

        ign_points_all = np.zeros(shape=[0, 2])
        for hotspot in self.hotspot_areas:
            x_min, x_max = hotspot[0], hotspot[1]
            y_min, y_max = hotspot[2], hotspot[3]
            ign_points_x = np.random.randint(low=x_min, high=x_max, size=(self.num_ign_points, 1))
            ign_points_y = np.random.randint(low=y_min, high=y_max, size=(self.num_ign_points, 1))
            ign_points_this_area = np.concatenate([ign_points_x, ign_points_y], axis=1)
            ign_points_all = np.concatenate([ign_points_all, ign_points_this_area], axis=0)

        return ign_points_all

    # initialize the geo-physical information
    def geo_phys_info_init(self, max_fuel_coeff, avg_wind_speed, avg_wind_direction):
        """
        This function generates a set of Geo-Physical information based on user defined ranges for each parameter

        :param max_fuel_coeff: maximum fuel coefficient based on vegetation type of the terrain
        :param avg_wind_speed: average effective mid-flame wind speed
        :param avg_wind_direction: wind azimuth
        :return: a dictionary containing geo-physical information
        """

        min_fuel_coeff = 1e-15
        fuel_rng = max_fuel_coeff - min_fuel_coeff
        spread_rate = fuel_rng*np.random.rand(self.terrain_sizes[0], self.terrain_sizes[1])+min_fuel_coeff
        wind_speed = np.random.normal(avg_wind_speed, 2, size=(self.terrain_sizes[0], 1))
        wind_direction = np.random.normal(avg_wind_direction, 2, size=(self.terrain_sizes[0], 1))

        geo_phys_info = {'spread_rate': spread_rate,
                         'wind_speed': wind_speed,
                         'wind_direction': wind_direction}

        return geo_phys_info

    # wildfire propagation
    def fire_propagation(self, ign_points_all, time_point, time_step, geo_phys_info, previous_terrain_map):
        """
        This function implements the simplified FARSITE wildfire propagation mathematical model

        :param ign_points_all: array including all fire-spots across entire terrain [output of hotspot_init()]
        :param time_point: current time step
        :param time_step: the time-step
        :param geo_phys_info: a dictionary including geo-physical information [output of geo_phys_info_inti()]
        :param previous_terrain_map: the terrain in previous time point as an array
        :return: new terrain map, new fire front points and their corresponding geo-physical information
        """

        if time_point == 1:
            final_terrain_map = self.initial_terrain_map
        else:
            final_terrain_map = previous_terrain_map

        current_geo_phys_info = np.zeros(shape=[ign_points_all.shape[0], 3])
        new_fire_front = np.zeros(shape=[ign_points_all.shape[0], 2])
        counter = 0
        for point in ign_points_all:
            # extracting the data
            x, y = int(point[0]), int(point[1])
            spread_rate = geo_phys_info['spread_rate']
            wind_speed = geo_phys_info['wind_speed']
            wind_direction = geo_phys_info['wind_direction']

            # extracting the required information
            final_terrain_map[x, y] = 1
            R = spread_rate[x, y]
            U = wind_speed[np.random.randint(low=0, high=self.terrain_sizes[0])][0]
            Theta = wind_direction[np.random.randint(low=0, high=self.terrain_sizes[0])][0]
            current_geo_phys_info[counter] = np.array([R, U, Theta])  # storing GP information

            # Simplified FARSITE
            LB = 0.936 * np.exp(0.2566 * U) + 0.461 * np.exp(-0.1548 * U) - 0.397
            HB = (LB + np.sqrt(np.absolute(np.power(LB, 2) - 1))) / (LB - np.sqrt(np.absolute(np.power(LB, 2) - 1)))
            C = 0.5 * (R - (R / HB))

            x_diff = C * np.sin(Theta)
            y_diff = C * np.cos(Theta)

            x_new = x + x_diff * time_step
            y_new = y + y_diff * time_step
            new_fire_front[counter] = np.array([x_new, y_new])  # storing new fire-front locations

            final_terrain_map[int(round(x_new)), int(round(y_new))] = 1  # TODO: 2 is for visualization [change to 1 for normal]
            counter += 1

        return final_terrain_map, new_fire_front, current_geo_phys_info
