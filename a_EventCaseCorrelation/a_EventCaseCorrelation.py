# Personentrennung und Tracelängenbestimmung
import inspect
import logging
import pathlib
import timeit

import numpy as np
import pandas as pd

import z_setting_parameters as settings
from u_utils import u_utils as utils, u_helper as helper


def choose_and_perform_event_case_correlation(dict_distance_adjacency_sensor,
                                              dir_runtime_files,
                                              raw_sensor_data,
                                              hyp_vectorization_method,
                                              hyp_trace_partition_method,
                                              hyp_number_of_activations_per_trace,
                                              hyp_trace_duration):
    filename_trace_data_time = settings.filename_trace_data_time
    filename_output_case_traces_cluster = settings.filename_output_case_traces_cluster
    path_data_sources = settings.path_data_sources
    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)

    # add LC_activity and duration to event log and room information (and the "major_room" where the sensor activity
    # takes place predominantly)
    # method only suitable for one person at the moment
    enhanced_event_log = convert_raw_data_to_event_log(dir_runtime_files=dir_runtime_files,
                                                       raw_sensor_data=raw_sensor_data,
                                                       dict_distance_adjacency_sensor=dict_distance_adjacency_sensor
                                                       )

    # returns the event log with case/cluster numbers
    log_with_case_id = partition_log_into_traces(traces_raw_pd=enhanced_event_log,
                                                 dir_runtime_files=dir_runtime_files,
                                                 hyp_trace_partition_method=hyp_trace_partition_method,
                                                 hyp_number_of_activations=hyp_number_of_activations_per_trace,
                                                 hyp_trace_duration=hyp_trace_duration)

    # transform log into vectors that can be clustered later
    vectorised_log = transform_log_to_vectors(log_with_case_id=log_with_case_id,
                                              dir_runtime_files=dir_runtime_files,
                                              vectorization_method=hyp_vectorization_method)

    log_with_case_id.rename(columns={'SensorID': 'LC_Activity'}, inplace=True)

    return vectorised_log, log_with_case_id


def convert_raw_data_to_event_log(raw_sensor_data,
                                  dict_distance_adjacency_sensor,
                                  dir_runtime_files):
    """
    Creates traces out of the raw sensor data.

    :param raw_sensor_data: pandas data frame of sensor points
    :param dict_distance_adjacency_sensor: dictionary which contains a matrix which determine which sensors are near to
    each other
    :param dir_runtime_files: directory of current run
    :return: A pandas data frame containing the created traces.
    """
    # # # new method 06-2021

    # # # import from settings file
    # data_sources_path: path of sources
    data_sources_path = settings.path_data_sources

    # filename_traces_raw: filename of trace file which will get created
    filename_traces_raw = settings.filename_traces_raw

    # csv_delimiter_traces: csv delimiter of trace file which will get created
    csv_delimiter_traces = settings.csv_delimiter_traces

    # get distance matrix from dictionary
    distance_matrix = dict_distance_adjacency_sensor['distance_matrix']

    # # #

    # start timer
    t0_convert_raw2trace = timeit.default_timer()

    raw_sensor_data_local = raw_sensor_data.copy()

    raw_sensor_data_local.rename(columns={'DateTime': 'Timestamp'}, inplace=True)

    # move index so it starts with 1 instead of zero
    # extract letter part from sensor ID. Usually contains information about the type (M=Motion, T=Temperature)
    raw_sensor_data_local.insert(loc=1, column='SensorType',
                                 value=raw_sensor_data_local['SensorID'].str.extractall('([a-zA-Z]+)').unstack().loc[:,
                                       0])

    # remove non-numeric part form sensorID to perform easier lookups with column
    raw_sensor_data_local['SensorID'] = (
        raw_sensor_data_local['SensorID'].str.extractall('(\d+)').unstack().loc[:, 0]).astype(int)

    # only keep motion sensors (for now. Take other sensor types into consideration in future versions)
    raw_sensor_data_motion = raw_sensor_data_local[
        (raw_sensor_data_local['SensorType'] == settings.prefix_motion_sensor_id)]

    # sort array by timestamp
    raw_sensor_data_motion = raw_sensor_data_motion.sort_values(by=['Timestamp']).reset_index(drop=True)
    # in case there is a motion sensorID that is not in the adjacency matrix, remove it
    raw_sensor_data_motion = raw_sensor_data_motion[(raw_sensor_data_motion['SensorID'] < len(distance_matrix))]

    # Which sensor has been added at the timestamp. If sensor was deactivated no sensor has been added
    raw_sensor_data_motion['Sensor_Added'] = \
        (raw_sensor_data_motion['Active'] * raw_sensor_data_motion['SensorID']).astype(int)

    raw_sensor_data_motion = raw_sensor_data_motion.sort_values(by=['SensorID', 'Timestamp'])
    # last occurrence of a sensor ID always has to be Active=0 and first Active=1
    raw_sensor_data_motion = raw_sensor_data_motion[~((raw_sensor_data_motion['SensorID'] !=
                                                       raw_sensor_data_motion['SensorID'].shift(-1)) &
                                                      (raw_sensor_data_motion['Active'] == 1))]
    # delete rows where sensor has activated again, but not deactivated
    raw_sensor_data_motion = \
        raw_sensor_data_motion[raw_sensor_data_motion['Active'] != raw_sensor_data_motion['Active'].shift(1)]
    # calculate the activation duration of every active sensor
    raw_sensor_data_motion['SensorActivationTime'] = (
        (raw_sensor_data_motion.sort_values(by=['SensorID', 'Timestamp'])['Timestamp'].diff(periods=1)).shift(
            -1)).dt.total_seconds()

    # restore old order and reset index
    raw_sensor_data_motion = raw_sensor_data_motion.sort_values(by=['Timestamp'])
    raw_sensor_data_motion = raw_sensor_data_motion.reset_index(drop=True)
    raw_sensor_data_motion.index = raw_sensor_data_motion.index + 1

    # remove all activation times for Active == 0, because SensorActivationTime only makes sense for activated sensors
    raw_sensor_data_motion['SensorActivationTime'] = np.where(raw_sensor_data_motion['Active'] == 0, np.nan,
                                                              raw_sensor_data_motion['SensorActivationTime'])

    raw_sensor_data_motion = raw_sensor_data_motion[raw_sensor_data_motion['SensorActivationTime'] < 1000]
    # at the end of a log, where sensors are not deactivated anymore, a large negative number would be displayed.
    # replace negative numbers with np.nan
    raw_sensor_data_motion['SensorActivationTime'] = np.where(raw_sensor_data_motion['SensorActivationTime'] < 0,
                                                              np.nan, raw_sensor_data_motion['SensorActivationTime'])

    # calculate the duration of each log entry
    # (pandas: shift column by one and then subtract the cell values column wise)
    raw_sensor_data_motion['Duration'] = ((raw_sensor_data_motion['Timestamp'].diff(periods=1)).
                                          shift(-1)).dt.total_seconds()

    # add LC_activity for process discovery later. Just use active column and replace 1 with s and 0 with c
    raw_sensor_data_motion['LC'] = np.where(raw_sensor_data_motion['Active'] == 1, 's', 'c')

    # add room/area name to log
    raw_sensor_data_motion['room'] = \
        raw_sensor_data_motion.replace({'SensorID': dict_distance_adjacency_sensor['sensor_to_room_dict']})['SensorID']

    # find out fringe areas where the sensor just gets activated by accident
    # compare rows below and above and if they are both different from the row itself then assume that the
    # new-room activation is not deliberate
    raw_sensor_data_motion['noise'] = ((raw_sensor_data_motion['room'].shift(periods=1)) !=
                                       raw_sensor_data_motion['room']) & \
                                      ((raw_sensor_data_motion['room'].shift(periods=-1)) !=
                                       raw_sensor_data_motion['room']) & \
                                      ((raw_sensor_data_motion['room'].shift(periods=-1)) == (
                                          raw_sensor_data_motion['room'].shift(periods=1)))

    # all noisy parts, take the room that was active around the outlying row and use it as the "major_room"
    raw_sensor_data_motion['room_major'] = np.where(raw_sensor_data_motion['noise'],
                                                    raw_sensor_data_motion['room'].shift(periods=-1),
                                                    raw_sensor_data_motion['room'])

    # do it one more time to get rid of the alternating room-activations. This has to be done better in future versions.
    raw_sensor_data_motion['noise'] = ((raw_sensor_data_motion['room_major'].shift(periods=1)) !=
                                       raw_sensor_data_motion['room_major']) & \
                                      ((raw_sensor_data_motion['room_major'].shift(periods=-1)) !=
                                       raw_sensor_data_motion['room_major']) & \
                                      ((raw_sensor_data_motion['room_major'].shift(periods=-1)) == (
                                          raw_sensor_data_motion['room_major'].shift(periods=1)))

    raw_sensor_data_motion['room_major'] = np.where(raw_sensor_data_motion['noise'],
                                                    raw_sensor_data_motion['room_major'].shift(periods=-1),
                                                    raw_sensor_data_motion['room_major'])

    # noise column no longer required. Drop it to save memory.
    raw_sensor_data_motion = raw_sensor_data_motion.drop(['noise'], axis=1)

    # stop timer
    t1_convert_raw2trace = timeit.default_timer()
    # calculate method runtime
    runtime_convert_raw2trace = np.round(t1_convert_raw2trace - t0_convert_raw2trace, 1)
    # logging
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)
    logger.info("Transforming log took %s seconds.", runtime_convert_raw2trace)

    # write traces to storage
    utils.write_csv_file(data=raw_sensor_data_motion, filedir=data_sources_path + dir_runtime_files,
                         filename=filename_traces_raw, separator=csv_delimiter_traces,
                         logging_level=settings.logging_level)
    # logging
    logger.info("Traces were saved as csv file '../%s", data_sources_path + dir_runtime_files + filename_traces_raw)

    return raw_sensor_data_motion


def partition_log_into_traces(traces_raw_pd,
                              dir_runtime_files,
                              hyp_trace_partition_method,
                              hyp_number_of_activations,
                              hyp_trace_duration):
    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)

    # boolean which shows if the trace partition was already executed with same parameters
    same_params_executed = False
    # path in which the traces file is located
    dir_same_param = ""

    # the sensor gets divided into traces by a fixed number of sensor activations
    if hyp_trace_partition_method == 'FixedSensorActivations':
        # checks if FixedSensorActivations trace partition method was already executed with same number of activations
        same_params_executed, dir_same_param = \
            helper.param_combination_already_executed(path_data_sources=settings.path_data_sources,
                                                      dir_export_files=
                                                      settings.dir_ecc_trace_partition_method_sensor_activations,
                                                      current_params={
                                                          'trace_partition_method': hyp_trace_partition_method,
                                                          'number_of_activations': hyp_number_of_activations},
                                                      step='fixed sensor activations trace partition')
        if not same_params_executed:
            # this is the first time the trace partition gets executed with the given number of activations
            # count the number of activations so every activated sensor is for now its own 'cluster'
            traces_raw_pd['Case'] = (traces_raw_pd['Active'].groupby(traces_raw_pd['Active']).cumsum()).astype(int)

            traces_raw_pd['Case'] = np.where(traces_raw_pd['Case'] == 0,
                                             0,
                                             ((traces_raw_pd['Case'] - 1) / hyp_number_of_activations).astype(int) + 1)

    # the sensor gets divided into traces by the sensor activation time
    elif hyp_trace_partition_method == 'FixedActivationTime':
        same_params_executed, dir_same_param = \
            helper.param_combination_already_executed(path_data_sources=settings.path_data_sources,
                                                      dir_export_files=
                                                      settings.dir_ecc_trace_partition_method_activation_time,
                                                      current_params={
                                                          'trace_partition_method': hyp_trace_partition_method,
                                                          'trace_duration': hyp_trace_duration},
                                                      step='fixed activation time trace partition')
        if not same_params_executed:
            # this is the first time the trace partition gets executed with the given activation time

            # cumulate the duration (not individual activation time of sensors) and divide it by the desired duration
            # only fill in active values. the deactivating has to be done in the cluster where the sensor was activated
            # in the first place
            traces_raw_pd['Case'] = ((traces_raw_pd['Duration'].fillna(0).cumsum() / hyp_trace_duration + 1) *
                                     traces_raw_pd['Active']).astype(int)

    # Divide sensor activations into traces. Activations in the same room and at the same time have the same case id.
    elif hyp_trace_partition_method == 'RoomsSimple':
        # checks if FixedSensorActivations trace partition method was already executed with same number of activations
        same_params_executed, dir_same_param = \
            helper.param_combination_already_executed(path_data_sources=settings.path_data_sources,
                                                      dir_export_files=
                                                      settings.dir_ecc_trace_partition_method_rooms_simple,
                                                      current_params={
                                                          'trace_partition_method': hyp_trace_partition_method},
                                                      step='rooms simple trace partition')
        if not same_params_executed:
            traces_raw_pd['Case'] = ((traces_raw_pd.room_major != traces_raw_pd.room_major.shift()).cumsum() *
                                     traces_raw_pd['Active']).astype(int)

    if not same_params_executed:
        # create directory if it not exists and export traces_raw_pd dataframe to drive
        path = pathlib.Path(dir_same_param)
        path.mkdir(parents=True, exist_ok=True)
        traces_raw_pd.to_pickle(dir_same_param + settings.filename_ecc_traces_raw)
        # write action into log
        logger.info(
            "Performed %s trace partition. Exported raw traces into '../%s'",
            hyp_trace_partition_method, dir_same_param)
    else:
        # The trace partition with the parameters was already executed in a previous iteration.
        # read in the result of it
        traces_raw_pd = pd.read_pickle(dir_same_param + settings.filename_ecc_traces_raw)
        # write action into log
        logger.info("Imported raw traces from '../%s'", dir_same_param)

    # need to sort the array because the later column comparison/merging does not keep index numbers
    traces_raw_pd_sorted = traces_raw_pd.sort_values(by=['SensorID', 'Timestamp'])

    # sort event log by SensorID, to get the corresponding log entry for deactivating the sensor again and
    # associate this with the same cluster
    # execute twice in order to mend recording errors
    for _ in range(2):
        traces_raw_pd_sorted['Case'] = np.where(traces_raw_pd_sorted['Case'] == 0,
                                                traces_raw_pd_sorted['Case'].shift(1),
                                                traces_raw_pd_sorted['Case'])

    # ensures the case numbers are consecutive numbers (especially when using the 'duration'-approach, cases would be
    # spread further apart
    traces_raw_pd_sorted['Case'] = (traces_raw_pd_sorted['Case'].rank(method='dense')).astype(int)

    # go back to old order
    traces_raw_pd_sorted = traces_raw_pd_sorted.sort_values(by=['Timestamp'])

    return traces_raw_pd_sorted


def transform_log_to_vectors(log_with_case_id,
                             dir_runtime_files,
                             vectorization_method):
    # for none of the methods the deactivation entry is needed.
    # drop those rows and keep only activated rows
    log_with_case_id = log_with_case_id[log_with_case_id.Active == 1]

    # count how many times a sensor has been activated in a case:
    # do not count M0 because this one will be activated for every case the same amount anyways (roughly)
    # this is because in a 'good' cluster the sensor should turn off just as much as it turns on
    if vectorization_method == 'quantity' or vectorization_method == 'quantity_time':
        quantity_vector = pd.pivot_table(log_with_case_id,
                                         index=['Case'], columns=['Sensor_Added'],
                                         values=['Active'], aggfunc='count', fill_value=0)
        quantity_vector.columns = ["Quantity " + str(x) for x in range(quantity_vector.shape[1])]

    # count how long a sensor has been activated in a case:
    if vectorization_method == 'time' or vectorization_method == 'quantity_time':
        time_vector = pd.pivot_table(log_with_case_id,
                                     index=['Case'],
                                     columns=['Sensor_Added'],
                                     values=['SensorActivationTime'], aggfunc='sum', fill_value=0)
        time_vector.columns = ["Time " + str(x) for x in range(time_vector.shape[1])]

    if vectorization_method == 'time':
        return time_vector
    elif vectorization_method == 'quantity':
        return quantity_vector
    elif vectorization_method == 'quantity_time':
        quantity_time_vector = pd.concat([quantity_vector, time_vector], axis=1)
        return quantity_time_vector

    return None
