"""
# ******************************<><><><><>************************************
# * Script for Uncertainty Based (AEKF) Wildfire Coverage Python Environment *
# ******************************<><><><><>************************************
#
# Dependencies:
#       - WildFire.py
#       - AEKF.py
#       - EKF.py
#       - KF.py
#       - UnicycleDynamics.py
#       - Models_Dynamics_for_AEKF.py
#       - numpy
#       - matplotlib
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
import matplotlib.pyplot as plt
from WildFire import WildFire
# from AEKF import AEKF
from EKF import EKF
# from KF import KF
import time
from UnicycleDynamics import UnicycleDynamics
from Models_Dynamics_for_AEKF import ModelDynamicsAEKF


# wildfire environment parameter initialization
hotspot_areas = [[300, 320, 400, 450], [400, 450, 300, 320]]  # [[x_min, x_max, y_min, y_max]]
terrain_sizes, num_ign_points = [int(5e2), int(5e2)], 25
duration, time_point, time_step = 500, 0, 1
max_fuel_coeff, avg_wind_speed, avg_wind_direction = 7, 5, np.pi / 8  # higher fuel_coeff:: more circular shape fire

# initializing UAV parameters
safe_altitude = 15
num_drones = 2
orientation = np.cos(1)  # we only consider planar movement here so orientation is enough and is defined instead of altitude
drone_loci = np.array([[200.34654, 400.3214, orientation], [250.31416354, 450.355354, orientation]])
drone_loci = drone_loci.T
initial_drone_loci = drone_loci[:2, :].T.copy()

# Vehicles Dynamics
position_error = 5
linear_velocity_limit = 15
angular_velocity_limit = 4 * np.pi
dubins = UnicycleDynamics(time_step=time_step, pos_error=position_error, rot_error=100, prop_ctrl_gain=1,
                          prop_controller_mag_limit=0.1, linear_velocity_limit=linear_velocity_limit,
                          angular_velocity_limit=angular_velocity_limit)

# EKF environment initialization
num_states_ekf = 8  # number of observation states
num_obs_states_ekf = 8  # number of observation states
mapping_flg = False  # is there gonna be any dynamics (e.g., mapping to new variables) for observation model?
fire_env = WildFire(terrain_sizes, hotspot_areas, num_ign_points, duration)  # initialize the fire environment
mdlynamics = ModelDynamicsAEKF(F_dim=num_states_ekf, H_dim=num_obs_states_ekf)  # initialize the wildfire dynamics for EKF
ign_points_all = fire_env.hotspot_init()  # initializing hotspots
geo_phys_info = fire_env.geo_phys_info_init(max_fuel_coeff, avg_wind_speed, avg_wind_direction)  # initializing terrain

# simulation initialization
final_visualization_flg = True
measurement_quality = 1  # variance of the noise (higher means worse quality)
fire_map = ign_points_all  # initializing fire-map
previous_terrain_map = np.zeros(shape=terrain_sizes)
total_uncertainty = np.zeros(shape=[duration, 1])
max_errors = []
dxu = np.zeros(shape=[2, num_drones])
path_trajectory = []
velocity_trajectory = []
start_time = time.time()
while time_point < duration:
    final_terrain_map, new_fire_front, current_geo_phys_info = \
        fire_env.fire_propagation(ign_points_all, time_point, time_step, geo_phys_info, previous_terrain_map)

    # next step data
    fire_map = np.concatenate([fire_map, new_fire_front], axis=0)
    ign_points_all = new_fire_front
    previous_terrain_map = final_terrain_map

    # extracting/receiving geo-physical information
    R = current_geo_phys_info[:, 0].reshape(ign_points_all.shape[0], 1)
    U = current_geo_phys_info[:, 1].reshape(ign_points_all.shape[0], 1)
    Theta = current_geo_phys_info[:, 2].reshape(ign_points_all.shape[0], 1)

    # uncertainty residual propagation with EKF
    drones_uncertainty_all = []
    error_map = np.zeros(shape=terrain_sizes)
    for d in drone_loci.T:
        px, py, pz = d[0], d[1], d[2]
        counter = 0
        this_drones_uncertainty_about_q = np.zeros(shape=[new_fire_front.shape[0], 1])
        for q in new_fire_front:
            qx, qy = q[0], q[1]

            # initializing AEKF parameters
            x0, P0, Q0, R0 = mdlynamics.kf_initializer(qx, qy, px, py, pz, R[counter][0], U[counter][0], Theta[counter][0])
            state_grads = mdlynamics.state_gradients(R[counter][0], U[counter][0], Theta[counter][0])
            F = mdlynamics.state_jacobian(state_grads)
            H = mdlynamics.observation_jacobian(state_grads, mapping_flg)  # direct state estimation with no mapping

            # initializing the EKF
            ekf = EKF(F=F, H=H, xhat0=x0, P0=P0, Q=Q0, R=R0, num_iter=500)

            # EKF predictor
            state_estimates, uncertainties = ekf.ekf(z=None, measurement_quality=1)

            # propagating uncertainty
            uncertainty_about_q = np.sum(uncertainties)
            kf_fire_spot_estimate = np.array([state_estimates[0], state_estimates[1]])  # TODO: this is only planar
            kf_drone_pose_estimate = np.array([state_estimates[2], state_estimates[3]])
            distance_error = np.linalg.norm(kf_fire_spot_estimate - kf_drone_pose_estimate)
            this_drones_uncertainty_about_q[counter] = uncertainty_about_q * distance_error

            # updating the error-map
            apprx_qx = int(round(state_estimates[0]))
            apprx_qy = int(round(state_estimates[1]))
            error_map[apprx_qx, apprx_qy] += this_drones_uncertainty_about_q[counter][0]
            counter += 1

        drones_uncertainty_all.append(this_drones_uncertainty_about_q)

    # calculating the total uncertainty
    total_uncertainty[time_point] = error_map.sum()

    # keeping track of all fire-spot uncertainties
    errors = np.zeros(shape=[drones_uncertainty_all[0].shape[0], len(drones_uncertainty_all)])
    for i in range(len(drones_uncertainty_all)):
        for j in range(drones_uncertainty_all[0].shape[0]):
            errors[j, i] = drones_uncertainty_all[i][j]
    max_errors.append(np.amax(errors, axis=1).reshape(new_fire_front.shape[0], 1))

    # maneuvering drones
    arrival_loci = drone_loci[:2, :].copy()  # storing the starting location
    area_centroids = np.vstack(  # TODO: now, giving explicit goal points to test the environment
        (np.mean(new_fire_front[:num_ign_points], axis=0), np.mean(new_fire_front[num_ign_points:], axis=0)))
    goals = np.concatenate((area_centroids, np.zeros(shape=[2, 1])), axis=1)
    goals = goals.T
    while np.size(dubins.at_pose(drone_loci, goals)) != num_drones:

        # create single-integrator control inputs
        dxi = dubins.single_integrator_position_controller(drone_loci[:2, :], goals[:2, :])

        # TODO: if collision avoidance is required here would be the place to add the function

        # decide what to do based on how close each agent is to its desired goal
        norms = np.linalg.norm(goals[:2, :] - drone_loci[:2, :], axis=0)
        not_there = np.where(norms > position_error)[0]
        there = np.where(norms <= position_error)[0]

        # calculate angle proportional controller
        wrapped_angles = goals[2, there] - drone_loci[2, there]
        wrapped_angles = np.arctan2(np.sin(wrapped_angles), np.cos(wrapped_angles))

        # set the velocities by mapping the single-integrator dynamics to unicycle dynamics
        # dxu = dubins.single_integrator_to_unicycle(dxi, drone_loci)  # in case don't want to change the angle proportional
        dxu[:, not_there] = dubins.single_integrator_to_unicycle(dxi[:, not_there], drone_loci[:, not_there])
        dxu[:, there] = np.vstack([np.zeros(np.size(there)), wrapped_angles])
        velocities = dxu.copy()

        # update agents unicycle dynamics
        drone_loci[0, :] = drone_loci[0, :] + time_step * np.cos(drone_loci[2, :]) * velocities[0, :]
        drone_loci[1, :] = drone_loci[1, :] + time_step * np.sin(drone_loci[2, :]) * velocities[0, :]
        drone_loci[2, :] = drone_loci[2, :] + time_step * velocities[1, :]

        # store data
        path_trajectory.append(drone_loci[:2, :].T.copy())
        velocity_trajectory.append(drone_loci[2, :].copy())

        # check for time
        distance_traversed = np.linalg.norm(arrival_loci - drone_loci[:2, :], axis=0)/linear_velocity_limit
        if np.size(np.where(distance_traversed >= 1)):
            break

    time_point += time_step
    print('time = ' + str(time_point))

end_time = time.time()
print('time to execute: ' + str(end_time - start_time))

# visualization
if final_visualization_flg:
    plt.rcParams['figure.figsize'] = (10, 8)

    plt.figure()
    plt.imshow(final_terrain_map, origin={'lower'}, cmap='jet')
    plt.title('The Truth fire map', fontweight='bold')

    plt.figure()
    plt.imshow(error_map, origin={'lower'}, cmap='jet')
    plt.title("Drone's Uncertainty Map of the Firefront", fontweight='bold')

    plt.figure()
    plt.plot(total_uncertainty, label='total uncertainty residual')
    plt.xlabel("time")
    plt.ylabel("total uncertainty residual over time", fontweight='bold')
    plt.legend()

    point_wise_errors = np.zeros(shape=[max_errors[0].shape[0], len(max_errors)])
    for i in range(len(max_errors)):
        for j in range(max_errors[0].shape[0]):
            point_wise_errors[j, i] = max_errors[i][j]
    plt.figure()
    plt.plot(point_wise_errors.T)
    plt.xlabel("time")
    plt.ylabel("uncertainty residuals for each fire-spot", fontweight='bold')

    counter = 1
    plt.figure()
    for start_pose in initial_drone_loci:
        plt.plot(start_pose[0], start_pose[1], 'k+', label='start#' + str(counter))
        plt.axis([0, terrain_sizes[0], 0, terrain_sizes[1]])
        counter += 1

    for data in path_trajectory:
        for drone_pose in data:
            plt.plot(drone_pose[0], drone_pose[1], 'b.', label='trajectory points')
    plt.ylabel("Planar Trajectories of the Drones", fontweight='bold')

    plt.figure()
    for vel_data in velocity_trajectory:
        plt.plot(vel_data[0], vel_data[1], 'r*', label='velocity points')
    plt.ylabel("Drones' Velocities During Mission ", fontweight='bold')

plt.show()
