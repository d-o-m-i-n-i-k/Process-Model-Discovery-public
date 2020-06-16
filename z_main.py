import inspect
import logging

from datetime import datetime
import timeit
import numpy as np
from hyperopt import fmin, Trials, STATUS_OK
import os
import pathlib

# import settings file
from u_utils import u_helper as helper, u_DistanceMatrixCreation as create_dm
import z_setting_parameters as settings
from a_EventCaseCorrelation import a_EventCaseCorrelation as ecc
from b_ActivityDiscovery import b_ActivityDiscovery as ad
from c_EventActivityAbstraction import c_EventActivityAbstraction as eaa
from d_ProcessDiscovery import d_ProcessDiscovery as prd

# start timer
t0_main = timeit.default_timer()

# check if runtime folder exists, if not create it
path = pathlib.Path(settings.path_data_sources + settings.dir_runtime_files)
path.mkdir(parents=True, exist_ok=True)

# Logger configuration
helper.configure_logger()

# checks settings for correctness
helper.check_settings(zero_distance_value_min=settings.zero_distance_value_min,
                      zero_distance_value_max=settings.zero_distance_value_max,
                      # trace_length_limit_min=settings.trace_length_limit_min,
                      # trace_length_limit_max=settings.trace_length_limit_max,
                      miner_type=settings.miner_type,
                      miner_type_list=settings.miner_type_list,
                      metric_to_be_maximised=settings.metric_to_be_maximised,
                      metric_to_be_maximised_list=settings.metric_to_be_maximised_list)

# get distance Matrix from imported adjacency-matrix
dict_distance_adjacency_sensor = create_dm.get_distance_matrix()

# draw a node-representation of the Smart Home
create_dm.draw_adjacency_graph(dict_room_information=dict_distance_adjacency_sensor,
                               data_sources_path=settings.path_data_sources,
                               filename_adjacency_plot=settings.filename_adjacency_plot)

# load data
# read in the sensor data as a pandas data frame
raw_sensor_data = helper.import_raw_sensor_data(filedir=settings.path_data_sources,
                                                filename=settings.filename_sensor_data,
                                                separator=settings.csv_delimiter_sensor_data,
                                                header=settings.csv_header_sensor_data,
                                                parse_dates=settings.csv_parse_dates_sensor_data,
                                                dtype=settings.csv_dtype_sensor_data)


def perform_process_model_discovery(params):
    """
    This method performs a process model discovery containing the steps event case correlation, activity discovery,
    event activity abstraction and process discovery.
    :param params: the parameter combination the steps are executed with
    :return: the result of the execution describing how precise a petri net in comparison to the event log is
    """
    # count number of iterations
    perform_process_model_discovery.iteration_counter += 1

    # apply current time to the format of the folder name containing files read and written during runtime
    dir_runtime_files = datetime.now().strftime(settings.dir_runtime_files + settings.dir_runtime_files_iteration)

    # create new folder for current run
    if not os.path.exists(settings.path_data_sources + dir_runtime_files):
        os.makedirs(settings.path_data_sources + dir_runtime_files)

    # write the parameters to the drive
    helper.create_parameters_log_file(dir_runtime_files=dir_runtime_files, params=params)

    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)
    logger.info("################# Start iteration %s of %s #################",
                perform_process_model_discovery.iteration_counter, settings.number_of_runs)

    # shows used parameters in log
    param_log_str = "Using the following parameters: "
    for param_name, param in params.items():
        param_log_str += str(param_name) + "=" + str(param) + ", "
    # cuts the ", " at the end of the string away
    param_log_str = param_log_str[0:len(param_log_str) - 2]
    logger.info(param_log_str)

    dict_distance_adjacency_sensor['distance_matrix'] = \
        create_dm.set_zero_distance_value(distance_matrix=dict_distance_adjacency_sensor['distance_matrix'],
                                          zero_distance_value=params['zero_distance_value'])

    # ################### EventCaseCorrelation ####################
    # transform raw-data to traces
    traces_vectorised, output_case_traces_cluster = \
        ecc.choose_and_perform_event_case_correlation(raw_sensor_data=raw_sensor_data,
                                                      dict_distance_adjacency_sensor=dict_distance_adjacency_sensor,
                                                      dir_runtime_files=dir_runtime_files,
                                                      hyp_vectorization_method=params['vectorization_type'],
                                                      hyp_trace_partition_method=params['trace_partition_method'],
                                                      hyp_number_of_activations_per_trace=params[
                                                          'number_of_activations_per_trace'],
                                                      hyp_trace_duration=params['trace_duration'])

    # ################### ActivityDiscovery ####################
    cluster = ad.choose_and_perform_clustering_method(clustering_method=params['clustering_method'],
                                                      hyp_number_of_clusters=params['hyp_number_of_clusters'],
                                                      trace_data_without_case_number=traces_vectorised)

    # ################### EventActivityAbstraction ####################
    output_case_traces_cluster = eaa.create_event_log_files(dir_runtime_files=dir_runtime_files,
                                                            cluster=cluster,
                                                            traces_vectorised=traces_vectorised,
                                                            output_case_traces_cluster=output_case_traces_cluster,
                                                            hyp_week_separator=params['hyp_week_separator'],
                                                            hyp_number_of_day_partitions=params[
                                                                'hyp_number_of_day_partitions'])

    # ################### ProcessDiscovery ####################
    # discover the process models of the respective clusters
    prd.create_activity_models(output_case_traces_cluster=output_case_traces_cluster,
                               path_data_sources=settings.path_data_sources, dir_runtime_files=dir_runtime_files,
                               dir_dfg_cluster_files=settings.dir_dfg_files,
                               filename_dfg_cluster=settings.filename_dfg_cluster,
                               rel_proportion_dfg_threshold=settings.rel_proportion_dfg_threshold)

    # discover the process model for the overall daily routine
    metrics = prd.create_process_model(output_case_traces_cluster=output_case_traces_cluster,
                                       dir_runtime_files=dir_runtime_files)

    # show function value history in a graph
    perform_process_model_discovery.function_values.append(metrics[settings.metric_to_be_maximised])
    helper.show_function_value_history(function_values=perform_process_model_discovery.function_values,
                                       iterations=range(1, perform_process_model_discovery.iteration_counter + 1))

    # stop timer
    t1_main = timeit.default_timer()

    # calculate runtime
    runtime_main = np.round(t1_main - t0_main, 1)

    helper.append_to_performance_documentation_file(
        path_data_sources=settings.path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_benchmark=settings.filename_benchmark,
        csv_delimiter_benchmark=settings.csv_delimiter_benchmark,
        list_of_properties={'Maximising': settings.metric_to_be_maximised,
                            'Precision': metrics['Precision'],
                            'Fitness': metrics['Fitness'],
                            'F1': metrics['F1'],
                            'FunctionValueType': settings.metric_to_be_maximised,
                            'runtime_main': runtime_main,
                            'iteration': perform_process_model_discovery.iteration_counter,
                            'zero_distance_value': params['zero_distance_value'],
                            'trace_partition_method': params['trace_partition_method'],
                            'number_of_activations_per_trace': params['number_of_activations_per_trace'],
                            'trace_duration': params['trace_duration'],
                            'vectorization_type': params['vectorization_type'],
                            'MinerType': settings.miner_type,
                            'clustering_method': params['clustering_method'],
                            'hyp_number_of_day_partitions': params['hyp_number_of_day_partitions'],
                            'hyp_week_separator': params['hyp_week_separator'],
                            'hyp_number_of_clusters': params['hyp_number_of_clusters']})

    helper.append_to_log_file(
        new_entry_to_log_variable='runtime_main',
        new_entry_to_log_value=runtime_main,
        path_data_sources=settings.path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=settings.filename_parameters_file,
        new_entry_to_log_description='Total runtime in seconds.')

    helper.append_to_log_file(
        new_entry_to_log_variable='execution_completed',
        new_entry_to_log_value=True,
        path_data_sources=settings.path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=settings.filename_parameters_file,
        new_entry_to_log_description='Successfully executed code.')

    logger.info("Total runtime: %s", runtime_main)
    logger.info("################# End iteration %s of %s #################",
                perform_process_model_discovery.iteration_counter, settings.number_of_runs)

    return {
        'loss': -metrics[settings.metric_to_be_maximised],
        'status': STATUS_OK,
        'iteration': perform_process_model_discovery.iteration_counter,
        'dir_runtime_files': dir_runtime_files,
        'opt_params': params
    }


# capture the iterations of perform_process_model_discovery method
perform_process_model_discovery.iteration_counter = 0
perform_process_model_discovery.function_values = []

# execute process model discovery with fixed parameters
if settings.execution_type == 'fixed_params':
    function_values = []
    while perform_process_model_discovery.iteration_counter < settings.number_of_runs:
        perform_process_model_discovery(settings.fixed_params)

# execute process model discovery with parameter search space (hyperopt parameter tuning)
elif settings.execution_type == 'param_optimization':
    # parameter's search space
    space = helper.create_param_opt_space()
    trials = Trials()
    # perform process model discovery for different parameter combinations and find the best outcome
    fmin(fn=perform_process_model_discovery,
         space=space,
         algo=settings.opt_algorithm,
         max_evals=settings.number_of_runs,
         verbose=False,
         trials=trials)

    # additional information of the best iteration
    best = trials.best_trial
    information_string = '\nbest iteration:\n\toptimised function value = ' + str(-best['result']['loss']) + \
                         '\n\tfunction value type = ' + settings.metric_to_be_maximised + '\n\toptimised parameters:'
    for key, value in best['result']['opt_params'].items():
        information_string += '\n\t\t' + str(key) + ' = ' + str(value)
    information_string += '\n\tfiles directory = ' + str(best['result']['dir_runtime_files']) + '\n\titeration = ' \
                          + str(best['result']['iteration'])

    # logger to print and save the information
    logger = logging.getLogger('main')
    logger.setLevel(settings.logging_level)
    logger.info(information_string)

pass
