from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import wandb


class EnergyManagementEnv(py_environment.PyEnvironment):

    """
    Initialize the environment. Default values simulate tesla's powerwall (13.5 kWh, with 4.6 kW power, 2.3 charge, -2.3 discharge, and grid 25 kW)

    :param data: load, pv and electricity price data
    :param init_charge: initial state of charge of the battery in kWh
    :param timeslots_per_day: count of timeslots in 24h interval
    :param max_days: count of days to train on
    :param capacity: capacity of the battery in kWh
    :param power_battery: power of the battery in kW
    :param power_grid: power of the electricity grid in kW
    :param logging: activates test logs
    """
    def __init__(
            self, 
            data, #load, pv and electricity price data
            init_charge=0.0, #initial state of charge of the battery in kWh
            timeslots_per_day=48, #count of timeslots in 24h interval
            days=365, #count of days to train on
            capacity=13.5, #capacity of the battery in kWh
            power_battery=4.6, #power of the battery in kW
            power_grid=25.0, #power of the electricity grid in kW
            ecoPriority=0, #0 -> only consider cost for reward / 1 only consider emissions / 0.5 consider both equally
            logging=False
        ):
 
        self._current_timestep = -1 #Tracks the current timeslot in the simulation.
        self._timeslots_per_day = timeslots_per_day #Maximum number of timeslots in a 24-hour interval.
        self._max_timesteps = timeslots_per_day * days #Maximum number of days to train on.
        self._capacity = capacity #Capacity of the battery.
        self._power_battery = power_battery # Power of the battery.
        self._init_charge = init_charge #Initial state of charge of the battery.
        self._soe = init_charge #State of charge of the battery (initialized based on init_charge).
        self._power_grid = power_grid #Power of the electricity grid.
        self._episode_ended = False #Boolean flag indicating whether the current episode has ended.
        self._electricity_cost = 0.0 #Cumulative electricity cost incurred during the simulation.
        self._total_emissions = 0.0 #Cumulative emissions incurred during the simulation
        self._logging = logging #Boolean flag indicating whether the environment is in test mode.
        self._feed_in_price = 0.076
        self._ecoPriority = ecoPriority

        # Continuous action space battery=[-2.3,2.3]
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-self._power_battery/2, maximum=self._power_battery/2, name='action')

        # Observation space [day,timeslot,soe,load,pv,price]
        self._observation_spec = array_spec.BoundedArraySpec(shape=(40,), dtype=np.float32, name='observation')
        self._data = data #Data: load, PV, price, fuel mix

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_timestep = 0
        
        p_load = self._data.iloc[self._current_timestep,0]
        p_pv = self._data.iloc[self._current_timestep,1]
        p_net_load = p_load - p_pv
        electricity_price = self._data.iloc[self._current_timestep,2]
        grid_emissions = self._data.iloc[self._current_timestep,3]

        pv_forecast = self._data.iloc[self._current_timestep+1 : self._current_timestep+19, 1]
        electricity_price_forecast = self._data.iloc[self._current_timestep+1 : self._current_timestep+19,2]

        self._soe = self._init_charge
        self._episode_ended = False
        self._electricity_cost = 0.0
        self._total_emissions = 0.0

        observation = np.concatenate(([self._soe, p_net_load, grid_emissions, electricity_price], electricity_price_forecast, pv_forecast), dtype=np.float32)
        # observation = np.array([self._soe, p_net_load, electricity_price], dtype=np.float32)
        return ts.restart(observation)

    def _step(self, action):

        #0. Set time
        self._current_timestep += 1
        #Check for Episode termination to reset
        if self._episode_ended:
            return self.reset()

        # 1. Balance Battery
        penalty_factor = 3
        soe_old = self._soe
        self._soe = np.clip(soe_old + action[0], 0.0, self._capacity, dtype=np.float32)
        #1.1 Physikal limitations: Guides the agent to explore charging and discharging
        penalty_soe  = np.abs(action[0] - (self._soe - soe_old))*penalty_factor

        #1.2 Battery aging
        lower_threshold, upper_threshold = 0.10 * self._capacity, 0.90 * self._capacity
        if self._soe < lower_threshold:
            penalty_aging = (lower_threshold - self._soe) * penalty_factor
        elif self._soe > upper_threshold:
            penalty_aging = (self._soe - upper_threshold) * penalty_factor
        else:
            penalty_aging = 0 

        p_battery = soe_old - self._soe #Clipped to actual charging. + -> discharging/ providing energy
        
        #2. Get data
        p_load = self._data.iloc[self._current_timestep, 0] 
        p_pv = self._data.iloc[self._current_timestep, 1] 
        p_net_load = p_load - p_pv
        electricity_price = self._data.iloc[self._current_timestep, 2]
        grid_emissions = self._data.iloc[self._current_timestep, 3]

        #2.1 Get forecasts
        pv_forecast = self._data.iloc[self._current_timestep+1 : self._current_timestep+19, 1]
        price_forecast = self._data.iloc[self._current_timestep+1 : self._current_timestep+19, 2]
        
        #3. Balance Grid
        grid = p_load - p_pv - p_battery
        grid_buy = grid if grid > 0 else 0
        grid_sell = abs(grid) if grid < 0 else 0

        #4. Calculate profit
        cost = grid_buy*electricity_price
        profit = grid_sell*self._feed_in_price
        self._electricity_cost += profit - cost

        #4.1 Calculate emissions
        emissions = grid_buy*grid_emissions
        self._total_emissions += emissions

        emissions_penalty_factor = 0.05  # This value could be adjusted based on how severely you want to penalize emissions
        emissions_impact = emissions * emissions_penalty_factor

        reward_scaling_factor = 5
        reward = ((profit - cost)*reward_scaling_factor)*(1-self._ecoPriority) - (self._ecoPriority * emissions_impact) - penalty_soe - penalty_aging

        #6. Create observation
        observation = np.concatenate(([self._soe, p_net_load, grid_emissions, electricity_price], price_forecast, pv_forecast), dtype=np.float32)
  
        # Logging
        if self._logging:
            wandb.log({
            'Action [2.3, -2.3]': action[0], 
            'SoE [0, 13.5]': self._soe, 
            'Battery wear cost': penalty_aging,
            'Profit (+ profit, - cost)': profit - cost,
            'Reward' : reward,
            'PV': p_pv, 
            'Load' : p_load, 
            'Price' : electricity_price,
            'Net load': p_net_load,
            'Emissions [kg]': emissions,
            })

        # Check for episode end
        if self._current_timestep >= self._max_timesteps - 19:
            self._episode_ended = True
            if self._logging:
                wandb.log({'Final Profit': self._electricity_cost})
                wandb.log({'Final Emissions': self._total_emissions})               
            return ts.termination(observation=observation,reward=reward)
        else:
            return ts.transition(observation=observation,reward=reward)