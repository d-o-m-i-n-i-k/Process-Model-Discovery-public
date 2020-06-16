# Zuordnung Cluster und Aktivitäten
import numpy as np
import pandas as pd
import z_setting_parameters as settings


def create_event_log_files(dir_runtime_files,
                           cluster,
                           traces_vectorised,
                           output_case_traces_cluster,
                           hyp_week_separator,
                           hyp_number_of_day_partitions):
    traces_vectorised['Cluster'] = cluster
    output_case_traces_cluster['Cluster'] = \
        traces_vectorised['Cluster'][output_case_traces_cluster['Case']].values

    # drop column where cluster = nan ->
    # output_case_traces_cluster = output_case_traces_cluster.dropna(subset=['Cluster'])

    traces_vectorised = traces_vectorised.sort_values(by=['Cluster'])
    traces_vectorised = traces_vectorised.reset_index(drop=True)

    # add column with routine description to the log
    output_case_traces_cluster_routines = add_routine(output_case_traces_cluster,
                                                      hyp_week_separator=hyp_week_separator,
                                                      hyp_number_of_day_partitions=hyp_number_of_day_partitions)

    # write traces to disk
    # time of sensor activations grouped by case
    traces_vectorised.to_csv(settings.path_data_sources + dir_runtime_files + settings.filename_cluster,
                             sep=settings.csv_delimiter_cluster)

    output_case_traces_cluster_routines.to_csv(settings.path_data_sources + dir_runtime_files +
                                               settings.filename_cases_cluster,
                                               sep=settings.csv_delimiter_cases_cluster)

    return output_case_traces_cluster_routines


def add_routine(output_case_traces_cluster,
                hyp_week_separator,
                hyp_number_of_day_partitions):
    # add 'routine'-column to dataframe (morning/evening-routine or workday/weekend-routine)

    # output_case_traces_cluster['Routine'] = output_case_traces_cluster.apply(routine_classifier, axis=1)

    if hyp_week_separator == 'workday':
        output_case_traces_cluster['Routine'] = np.where(output_case_traces_cluster['Timestamp'].dt.dayofweek < 5,
                                                         'weekday', 'weekend')
        # if day is partitioned, add underscore
        if hyp_number_of_day_partitions > 0:
            output_case_traces_cluster['Routine'] += '_'
    elif hyp_week_separator == 'weekday':
        output_case_traces_cluster['Routine'] = output_case_traces_cluster['Timestamp'].dt.day_name()
        # if day is partitioned, add underscore
        if hyp_number_of_day_partitions > 0:
            output_case_traces_cluster['Routine'] += '_'
    else:
        output_case_traces_cluster['Routine'] = ''

    if hyp_number_of_day_partitions == 2:
        output_case_traces_cluster['Routine'] += output_case_traces_cluster.apply(partition_day_in_two, axis=1)
    elif hyp_number_of_day_partitions == 3:
        output_case_traces_cluster['Routine'] += output_case_traces_cluster.apply(partition_day_in_three, axis=1)
    elif hyp_number_of_day_partitions == 4:
        output_case_traces_cluster['Routine'] += output_case_traces_cluster.apply(partition_day_in_four, axis=1)
    elif hyp_number_of_day_partitions == 5:
        output_case_traces_cluster['Routine'] += output_case_traces_cluster.apply(partition_day_in_five, axis=1)
    else:
        pass

    if hyp_number_of_day_partitions == 1 and not hyp_week_separator:
        output_case_traces_cluster['Routine'] = 'None'

    # need to sort the array to match the off sensors to the same 'routine' as the on sensors
    output_case_traces_cluster_sorted = output_case_traces_cluster.sort_values(by=['LC_Activity', 'Timestamp'])

    # sort event log by SensorID, to get the corresponding log entry for deactivating the sensor again and
    # associate this with the same cluster
    # execute twice in order to mend recording errors
    for _ in range(2):
        output_case_traces_cluster_sorted['Routine'] = np.where(output_case_traces_cluster_sorted['LC'] == 'c',
                                                                output_case_traces_cluster_sorted['Routine'].shift(1),
                                                                output_case_traces_cluster_sorted['Routine'])

    # go back to old order
    output_case_traces_cluster_sorted = output_case_traces_cluster_sorted.sort_values(by=['Timestamp'])
    return output_case_traces_cluster_sorted


def partition_day_in_two(row):
    if 0 < row['Timestamp'].hour <= 11:
        val = 'AM'
    else:
        val = 'PM'
    return val


def partition_day_in_three(row):
    if 0 < row['Timestamp'].hour <= 8:
        val = 'night'
    elif 8 < row['Timestamp'].hour <= 16:
        val = 'day'
    else:
        val = 'evening'
    return val


def partition_day_in_four(row):
    if 0 < row['Timestamp'].hour <= 6:
        val = 'night'
    elif 6 < row['Timestamp'].hour <= 12:
        val = 'morning'
    elif 12 < row['Timestamp'].hour <= 18:
        val = 'afternoon'
    else:
        val = 'evening'
    return val


def partition_day_in_five(row):
    if 5 < row['Timestamp'].hour <= 10:
        val = 'morning'
    elif 10 < row['Timestamp'].hour <= 14:
        val = 'noon'
    elif 14 < row['Timestamp'].hour <= 17:
        val = 'afternoon'
    elif 17 < row['Timestamp'].hour <= 23:
        val = 'evening'
    else:
        val = 'night'
    return val


def create_event_log_files_deprecated(trace_data_time, output_case_traces_cluster, k_means_cluster_ids,
                                      path_data_sources,
                                      dir_runtime_files, sm, km, filename_cluster, csv_delimiter_cluster,
                                      filename_cases_cluster,
                                      csv_delimiter_cases_cluster):
    # write best matching units (BMU) to output file
    trace_data_time['BMU'] = np.transpose(sm._bmu[0, :]).astype(int)

    # write k-Means Cluster to output file
    cluster_list = km.labels_[trace_data_time.BMU]
    trace_data_time['Cluster'] = pd.DataFrame(cluster_list)

    # write vanilla kmeans to output file
    trace_data_time['kMeansVanilla'] = np.transpose(k_means_cluster_ids).astype(int)

    # write BMU and Cluster to output_case file
    # so the raw_data from the beginning now also has clusters
    output_case_traces_cluster['Cluster'] = trace_data_time['Cluster'][output_case_traces_cluster['Case']].values
    output_case_traces_cluster['BMU'] = trace_data_time['BMU'][output_case_traces_cluster['Case']].values
    output_case_traces_cluster['kMeansVanilla'] = k_means_cluster_ids[output_case_traces_cluster['Case'] - 1]
    # drop column where cluster = nan
    output_case_traces_cluster = output_case_traces_cluster.dropna(subset=['Cluster'])

    trace_data_time = trace_data_time.sort_values(by=['Cluster', 'BMU'])
    trace_data_time = trace_data_time.reset_index(drop=True)

    # write traces to disk
    # time of sensor activations grouped by case
    trace_data_time.to_csv(path_data_sources + dir_runtime_files + filename_cluster, sep=csv_delimiter_cluster)

    output_case_traces_cluster.to_csv(path_data_sources + dir_runtime_files + filename_cases_cluster,
                                      sep=csv_delimiter_cases_cluster)

    return output_case_traces_cluster
