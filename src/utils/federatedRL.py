#Imports
import os
import numpy as np
import pandas as pd
from typing import Dict, List
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.agents import ddpg
from tf_agents.agents.ddpg import ddpg_agent  

import wandb

from utils.EnergyManagementEnv import *
from utils.federatedAggregation import FederatedAggregation

def setup_energymanagement_environments(
        num_buildings=30, 
        path_energy_data="../../data/Final_Energy_dataset.csv",
        return_dataset=False,
        ecoPriority=0):
    
    energy_data = pd.read_csv(path_energy_data).fillna(0).set_index('Date')
    #emission_data = pd.read_csv(path_emission_data, index_col=0, parse_dates=True).fillna(0)
    #energy_data['emissions'] = emission_data['emissions'] 
    
    dataset = {"train": {}, "eval": {}, "test": {}}
    environments = {"train": {}, "eval": {}, "test": {}}
   
    for idx in range(num_buildings):
        user_data = energy_data[[f'load_{idx+1}', f'pv_{idx+1}', 'price', 'emissions']]
        
        dataset["train"][f"building_{idx+1}"] = user_data[0:17520].set_index(pd.RangeIndex(0,17520))
        dataset["eval"][f"building_{idx+1}"] = user_data[17520:35088].set_index(pd.RangeIndex(0,17568))
        dataset["test"][f"building_{idx+1}"] = user_data[35088:52608].set_index(pd.RangeIndex(0,17520))

        environments["train"][f"building_{idx+1}"] = tf_py_environment.TFPyEnvironment(EnergyManagementEnv(init_charge=0.0, data=dataset["train"][f"building_{idx+1}"], ecoPriority=ecoPriority))
        environments["eval"][f"building_{idx+1}"] = tf_py_environment.TFPyEnvironment(EnergyManagementEnv(init_charge=0.0, data=dataset["eval"][f"building_{idx+1}"], ecoPriority=ecoPriority))
        environments["test"][f"building_{idx+1}"] = tf_py_environment.TFPyEnvironment(EnergyManagementEnv(init_charge=0.0, data=dataset["test"][f"building_{idx+1}"], ecoPriority=ecoPriority, logging=True))

    observation_spec = environments["train"][f"building_1"].observation_spec()
    action_spec = environments["train"][f"building_1"].action_spec()

    if return_dataset:
        return environments, observation_spec, action_spec, dataset
    else:
        return environments, observation_spec, action_spec

def initialize_ddpg_agent(observation_spec, action_spec, global_step, environments, first_building=1): 
    
    actor_net = ddpg.actor_network.ActorNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec, 
        fc_layer_params=(400, 300),
        activation_fn=tf.keras.activations.relu)
     
    critic_net = ddpg.critic_network.CriticNetwork(
        input_tensor_spec=(observation_spec, action_spec),
        observation_fc_layer_params=(400,),
        joint_fc_layer_params=(300,),
        activation_fn=tf.keras.activations.relu)
    
    target_actor_network = ddpg.actor_network.ActorNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec, 
        fc_layer_params=(400, 300),
        activation_fn=tf.keras.activations.relu)

    target_critic_network = ddpg.critic_network.CriticNetwork(
        input_tensor_spec=(observation_spec, action_spec),
        observation_fc_layer_params=(400,),
        joint_fc_layer_params=(300,),
        activation_fn=tf.keras.activations.relu)
    

    agent_params = {
        "time_step_spec": environments["train"][f"building_{first_building}"].time_step_spec(),
        "action_spec": environments["train"][f"building_{first_building}"].action_spec(),
        "actor_network": actor_net,
        "critic_network": critic_net,
        "actor_optimizer": tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4), 
        "critic_optimizer": tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3), 
        "ou_stddev": 0.2,
        "ou_damping": 0.15,
        "target_actor_network": target_actor_network,
        "target_critic_network": target_critic_network,
        "target_update_tau": 0.05,
        "target_update_period": 5,
        "dqda_clipping": None,
        "td_errors_loss_fn": tf.compat.v1.losses.huber_loss,
        "gamma": 0.99, 
        "reward_scale_factor": 1,
        "train_step_counter": global_step,
    }

    # Create the DdpgAgent with unpacked parameters
    ddpg_tf_agent = ddpg_agent.DdpgAgent(**agent_params)

    ddpg_tf_agent.initialize()
    eval_policy = ddpg_tf_agent.policy
    collect_policy = ddpg_tf_agent.collect_policy

    return ddpg_tf_agent, eval_policy, collect_policy

def save_ddpg_weights(global_tf_agent, model_dir):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *global_tf_agent._actor_network.get_weights())
    np.savez(os.path.join(model_dir, "critic_weights.npz"), *global_tf_agent._critic_network.get_weights())
    np.savez(os.path.join(model_dir, "target_actor_weights.npz"), *global_tf_agent._target_actor_network.get_weights())
    np.savez(os.path.join(model_dir, "target_critic_weights.npz"), *global_tf_agent._target_critic_network.get_weights())

def initialize_fl_round_0(num_clusters, clustered_buildings, observation_spec, action_spec, environments):

    # Reset TensorFlow graph and get global step
    tf.compat.v1.reset_default_graph()
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Initialize a global model for each cluster of similar buildings
    for cluster in range(num_clusters):
        # 1. Build global agent per cluster
        first_building_in_cluster = clustered_buildings[cluster][0]
        global_ddpg_agent, global_eval_policy, global_collect_policy = initialize_ddpg_agent(
            observation_spec=observation_spec,
            action_spec=action_spec,
            global_step=global_step,
            environments=environments,
        )

        # 2. Initially store weights
        model_dir = os.path.join(os.getcwd(), f"models/Kmeans/NumCluster_{cluster}/FLround0_c{num_clusters}")
        os.makedirs(model_dir, exist_ok=True)

        # Save the DDPG agent's weights to the specified directory
        save_ddpg_weights(global_ddpg_agent, model_dir)

def set_weights_to_ddpg_agent(local_ddpg_agent, model_dir):
    # Extract the arrays using the keys corresponding to their order
    with np.load(os.path.join(model_dir, "actor_weights.npz"), allow_pickle=True) as data:
        actor_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._actor_network.set_weights(actor_weights)
    
    with np.load(os.path.join(model_dir, "critic_weights.npz"), allow_pickle=True) as data:
        critic_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._critic_network.set_weights(critic_weights)
    
    with np.load(os.path.join(model_dir, "target_actor_weights.npz"), allow_pickle=True) as data:
        target_actor_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._target_actor_network.set_weights(target_actor_weights)
    
    with np.load(os.path.join(model_dir, "target_critic_weights.npz"), allow_pickle=True) as data:
        target_critic_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._target_critic_network.set_weights(target_critic_weights)

    return local_ddpg_agent


def setup_rl_training_pipeline(tf_agent, env_train, replay_buffer_capacity,collect_policy, initial_collect_steps, collect_steps_per_iteration, batch_size):
    
    #Setup replay buffer -> TFUniform to give each sample an equal selection chance
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size= env_train.batch_size,
            max_length=replay_buffer_capacity,
        )

    # Populate replay buffer with inital experience before actual training (for num_steps times)
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        env=env_train,
        policy=collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps,
    )

    # After the initial collection phase, the collect driver takes over for the continuous collection of data during the training process
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        env=env_train,
        policy=collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration,
    )

    # For better performance
    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data
    initial_collect_driver.run()
    #initial_collect_driver.run()
    time_step = env_train.reset()
    policy_state = collect_policy.get_initial_state(env_train.batch_size)

    # The dataset is created from the replay buffer in a more structured and efficient way to provide mini-batches
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.experimental.AUTOTUNE, 
        sample_batch_size=batch_size, num_steps=2).prefetch(tf.data.experimental.AUTOTUNE)
    
    #Feed batches of experience to the agent for training
    iterator = iter(dataset)

    return iterator, collect_driver, time_step, policy_state

def local_agent_training_and_evaluation(
        iterator, collect_driver, time_step, policy_state, global_step, tf_agent, 
        eval_policy, local_storage, building_index, num_iterations, environments, agent_type): 
                    
    eval_metrics = [tf_metrics.AverageReturnMetric()]
    test_metrics = [tf_metrics.AverageReturnMetric()]

    while global_step.numpy() < num_iterations:
        time_step, policy_state = collect_driver.run(time_step=time_step, policy_state=policy_state)
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)
    
    #4.Evaluate training
    eval_metric = metric_utils.eager_compute(test_metrics, environments["eval"][f"building_{building_index}"], eval_policy)
    local_storage["performance_metrics"].append(eval_metric["AverageReturn"].numpy())
    #print("Return: ", eval_metric["AverageReturn"].numpy())
    
    if agent_type == "ddpg":
        local_storage = append_ddpg_weights_to_local_storage(tf_agent, local_storage)
     
    return  tf_agent, local_storage

def append_ddpg_weights_to_local_storage(tf_agent, local_storage): 
    local_storage["actor_weights"].append(tf_agent._actor_network.get_weights())
    local_storage["critic_weights"].append(tf_agent._critic_network.get_weights())
    local_storage["target_actor_weights"].append(tf_agent._target_actor_network.get_weights())
    local_storage["target_critic_weights"].append(tf_agent._target_critic_network.get_weights())
    return local_storage

def save_federated_ddpg_weights(model_dir, average_actor_weights, average_critic_weights, average_target_actor_weights, average_target_critic_weights):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *average_actor_weights)
    np.savez(os.path.join(model_dir, "critic_weights.npz"), *average_critic_weights)
    np.savez(os.path.join(model_dir, "target_actor_weights.npz"), *average_target_actor_weights)
    np.savez(os.path.join(model_dir, "target_critic_weights.npz"), *average_target_critic_weights)

def federated_learning_training(federated_rounds: int, num_clusters: int, 
    clustered_buildings: Dict[int, List[int]], observation_spec, action_spec, environments,
    replay_buffer_capacity: int, initial_collect_steps: int, collect_steps_per_iteration: int,
    batch_size: int, num_iterations: int):

    for federated_round in range(federated_rounds):
        for cluster_number, buildings_in_cluster in clustered_buildings.items():
            # Display information about the current federated round and cluster
            #print(f"Cluster {cluster_number}: Buildings {buildings_in_cluster} Federated round ---", federated_round + 1, f"/ {federated_rounds}")

            # Initialize local storage to collect weights and metrics
            local_storage = {
                "actor_weights": [],
                "critic_weights": [],
                "target_actor_weights": [],
                "target_critic_weights": [],
                "performance_metrics": []
            }

            for building_index in buildings_in_cluster:
                # 0. Reset global step
                tf.compat.v1.reset_default_graph()
                global_step = tf.compat.v1.train.get_or_create_global_step()

                # 1. Initialize local agent
                local_ddpg_agent, local_eval_policy, local_collect_policy = initialize_ddpg_agent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    global_step=global_step,
                    environments=environments,
                )

                # 2. Set global weights for this training round to agent
                model_dir = os.path.join(os.getcwd(), f"models/Kmeans/NumCluster_{cluster_number}/FLround{federated_round}_c{num_clusters}")
                local_ddpg_agent = set_weights_to_ddpg_agent(local_ddpg_agent, model_dir)

                # 3. Prepare training pipeline
                local_iterator, local_collect_driver, local_time_step, local_policy_state = setup_rl_training_pipeline(
                    local_ddpg_agent,
                    environments["train"][f"building_{building_index}"],
                    replay_buffer_capacity,
                    local_collect_policy,
                    initial_collect_steps,
                    collect_steps_per_iteration,
                    batch_size,
                )

                # 4. Train, evaluate agent, and store weights
                local_ddpg_agent, local_storage = local_agent_training_and_evaluation(
                    local_iterator,
                    local_collect_driver,
                    local_time_step,
                    local_policy_state,
                    global_step,
                    local_ddpg_agent,
                    local_eval_policy,
                    local_storage,
                    building_index,
                    num_iterations,
                    environments,
                    agent_type="ddpg",
                )

            # Perform Federated Aggregation
            average_actor_weights = FederatedAggregation.federated_weighted_aggregation(
                local_storage["actor_weights"], local_storage["performance_metrics"]
            )
            average_critic_weights = FederatedAggregation.federated_weighted_aggregation(
                local_storage["critic_weights"], local_storage["performance_metrics"]
            )
            average_target_actor_weights = FederatedAggregation.federated_weighted_aggregation(
                local_storage["target_actor_weights"], local_storage["performance_metrics"]
            )
            average_target_critic_weights = FederatedAggregation.federated_weighted_aggregation(
                local_storage["target_critic_weights"], local_storage["performance_metrics"]
            )

            # Save federated weights for the next round (Round + 1)
            model_dir = os.path.join(os.getcwd(), f"models/Kmeans/NumCluster_{cluster_number}/FLround{federated_round + 1}_c{num_clusters}")
            os.makedirs(model_dir, exist_ok=True)
            save_federated_ddpg_weights(
                model_dir,
                average_actor_weights,
                average_critic_weights,
                average_target_actor_weights,
                average_target_critic_weights,
            )



def set_weights_to_ddpg_agent(local_ddpg_agent, model_dir):
    # Extract the arrays using the keys corresponding to their order
    with np.load(os.path.join(model_dir, "actor_weights.npz"), allow_pickle=True) as data:
        actor_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._actor_network.set_weights(actor_weights)
    
    with np.load(os.path.join(model_dir, "critic_weights.npz"), allow_pickle=True) as data:
        critic_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._critic_network.set_weights(critic_weights)
    
    with np.load(os.path.join(model_dir, "target_actor_weights.npz"), allow_pickle=True) as data:
        target_actor_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._target_actor_network.set_weights(target_actor_weights)
    
    with np.load(os.path.join(model_dir, "target_critic_weights.npz"), allow_pickle=True) as data:
        target_critic_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._target_critic_network.set_weights(target_critic_weights)

    return local_ddpg_agent

def initialize_wandb_logging(project="DDPG_battery_testing", name="Exp", num_iterations=1500, batch_size=1, a_lr="1e-4", c_lr="1e-3"):
    wandb.login()
    wandb.init(
        project="DDPG_battery_testing",
        job_type="train_eval_test",
        name=name,
        config={
            "train_steps": num_iterations,
            "batch_size": batch_size,
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-2}
    )
    artifact = wandb.Artifact(name='save', type="checkpoint")

    """train_checkpointer = common.Checkpointer(
            ckpt_dir='checkpoints/ddpg/',
            max_to_keep=1,
            agent=tf_agent,
            policy=tf_agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step
        )
        train_checkpointer.initialize_or_restore()"""

    return artifact

def end_and_log_wandb(metrics, artifact):
    wandb.log(metrics)
    #artifact.add_dir(local_path='checkpoints/ddpg/')
    wandb.log_artifact(artifact)
    wandb.finish()

def local_agent_training_and_evaluation(
        iterator, collect_driver, time_step, policy_state, global_step, tf_agent, 
        eval_policy, local_storage, building_index, num_iterations, environments, agent_type): 
                    
    eval_metrics = [tf_metrics.AverageReturnMetric()]
    test_metrics = [tf_metrics.AverageReturnMetric()]

    while global_step.numpy() < num_iterations:
        time_step, policy_state = collect_driver.run(time_step=time_step, policy_state=policy_state)
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)
    
    #4.Evaluate training
    eval_metric = metric_utils.eager_compute(test_metrics, environments["eval"][f"building_{building_index}"], eval_policy)
    local_storage["performance_metrics"].append(eval_metric["AverageReturn"].numpy())
    #print("Return: ", eval_metric["AverageReturn"].numpy())
    
    if agent_type == "ddpg":
        local_storage = append_ddpg_weights_to_local_storage(tf_agent, local_storage)
     
    return  tf_agent, local_storage

def agent_training_and_evaluation(global_step, num_test_iterations, collect_driver, time_step, policy_state, iterator, 
    tf_agent, eval_policy, building_index, result_df, eval_interval, environments): 
                    
    eval_metrics = [tf_metrics.AverageReturnMetric()]
    test_metrics = [tf_metrics.AverageReturnMetric()]

    while global_step.numpy() < num_test_iterations:
        
        #Training
        time_step, policy_state = collect_driver.run(time_step=time_step, policy_state=policy_state)
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)

        #Evaluation
        metrics = {}
        if global_step.numpy() % eval_interval == 0:
            eval_metric = metric_utils.eager_compute(eval_metrics,environments["eval"][f"building_{building_index}"], eval_policy, train_step=global_step)
        if global_step.numpy() % 2 == 0:
            metrics["loss"] = train_loss.loss
            wandb.log(metrics)
    
    #Testing
    test_metrics = metric_utils.eager_compute(test_metrics,environments["test"][f"building_{building_index}"], eval_policy, train_step=global_step)
    result_df = pd.concat([result_df, pd.DataFrame({'Building': [building_index], 'Total Profit': [wandb.summary["Final Profit"]], 'Total Emissions': [wandb.summary["Final Emissions"]]})], ignore_index=True)
    #print('Building: ', building_index, ' - Total Profit: ', wandb.summary["Final Profit"], ' - Total Emissions: ', wandb.summary["Final Emissions"])

    return result_df, metrics

def local_refitting_and_evaluation(best_federated_round: int, num_rounds: int, num_test_iterations: int, clustered_buildings: Dict[int, List[int]],
    observation_spec, action_spec, environments, replay_buffer_capacity: int, initial_collect_steps: int, collect_steps_per_iteration: int,
    batch_size: int, num_iterations: int, eval_interval: int, csv_name: str,):

    result_df = pd.DataFrame(columns=['Building', 'Total Profit'])

    for cluster_number, buildings_in_cluster in clustered_buildings.items():
        for building_index in buildings_in_cluster:
            for round in range(num_rounds):
                #print(f"Cluster: {cluster_number} - Building: {building_index} - Round: {round}")

                # 0. Reset global step and TensorFlow graph
                tf.compat.v1.reset_default_graph()
                global_step = tf.compat.v1.train.get_or_create_global_step()

                # 1. Initialize local agent
                tf_ddpg_agent, eval_policy, collect_policy = initialize_ddpg_agent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    global_step=global_step,
                    environments=environments,
                )

                # 2. Set global weights of this training round to the agent
                model_dir = os.path.join(os.getcwd(), f"models/Kmeans/NumCluster_{cluster_number}/FLround{best_federated_round}_c{len(clustered_buildings)}")
                tf_agent = set_weights_to_ddpg_agent(tf_ddpg_agent, model_dir)

                # 3. Prepare training pipeline: Setup iterator, replay buffer, and driver
                iterator, collect_driver, time_step, policy_state = setup_rl_training_pipeline(
                    tf_ddpg_agent,
                    environments["train"][f"building_{building_index}"],
                    replay_buffer_capacity,
                    collect_policy,
                    initial_collect_steps,
                    collect_steps_per_iteration,
                    batch_size,
                )

                # 4. Setup WandB logging
                artifact = initialize_wandb_logging(name=f"{csv_name}_Home{building_index}_rd{round}", num_iterations=num_iterations)

                # 5. Train and evaluate the agent
                result_df, metrics = agent_training_and_evaluation(
                    global_step,
                    num_test_iterations,
                    collect_driver,
                    time_step,
                    policy_state,
                    iterator,
                    tf_ddpg_agent,
                    eval_policy,
                    building_index,
                    result_df,
                    eval_interval,
                    environments,
                )

                # 6. End and log WandB
                end_and_log_wandb(metrics, artifact)

    return result_df