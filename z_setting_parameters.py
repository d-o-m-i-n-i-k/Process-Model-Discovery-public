import logging
from hyperopt import tpe

# ############################################################
# ADJUSTABLE VARIABLES

# folder name containing sensor data (relative from directory of sources)
rel_dir_name_sensor_data = '5-Kyoto/'
# rel_dir_name_sensor_data = '19-Aruba/'
# filename of the file that contains the sensor data
filename_sensor_data = '5-Kyoto-Data.txt'
# filename_sensor_data = '19-Aruba_Data'

# path of sources and outputs
path_data_sources = 'z_Data-Sources/' + rel_dir_name_sensor_data
# folder containing files read and written during runtime
dir_runtime_files = 'runtime-files/'
# folder of one iteration containing files read and written during runtime
dir_runtime_files_iteration = '%Y-%m-%d_%H-%M-%S/'
# filename of room separation
filename_room_separation = 'Room-separation.csv'
# filename of Adjacency-Matrix
filename_adjacency_matrix = 'Adjacency-Matrix.csv'
# filename of parameters file
filename_parameters_file = '0-Parameters.csv'
# filename of log file coming from logging module
filename_log_file = '1-LogFile.log'
# filename of Adjacency-plot
filename_adjacency_plot = 'adjacency_plot.pdf'

# csv configuration for sensor data file
# delimiter of the columns in csv file of sensor data (input)
csv_delimiter_sensor_data = '\t'
# indicator at which line the data starts
csv_header_sensor_data = 0
# columns that should get parsed as a date
csv_parse_dates_sensor_data = ['DateTime']
# data type of columns in the file
csv_dtype_sensor_data = {'Active': float}

# filename of cases benchmark file
filename_benchmark = 'benchmark.csv'
# csv delimiter of benchmark file
csv_delimiter_benchmark = ';'
# indicator at which line the data starts
csv_header_benchmark = 0

# the way in which the result of an iteration is evaluated
metric_to_be_maximised = 'F1'
metric_to_be_maximised_list = ['Precision', 'Fitness', 'entropia:Precision', 'entropia:Fitness', 'F1']

# number of times the process model discovery gets executed
number_of_runs = 200

# upper limit for input_data (set to None if there is no limit)
max_number_of_raw_input = None

# sklearn SOM settings hyperopt parameter tuning
# optimization algorithm (representative Tree of Parzen Estimators (TPE))
som_opt_algorithm = tpe.suggest
som_opt_attempts = 20
# The initial step size for updating the SOM weights. (default = 1)
min_lr = 1
max_lr = 1
# Parameter for magnitude of change to each weight. Does not update over training (as does learning rate)more aggressive
# updates to weights. (default = 1)
min_sigma = 1
max_sigma = 1

# number of motion sensors
number_of_motion_sensors = 51
# prefix of motion sensor IDs
prefix_motion_sensor_id = 'M'

# set a level of logging
logging_level = logging.INFO

# maximum number of persons which were in the house while the recording of sensor data
max_number_of_people_in_house = 1

# Specifies the linkage method for clustering with custom distance calculation
linkage_method_for_clustering = 'ward'
linkage_method_for_clustering_list = ['single', 'complete', 'average', 'weighted', 'median', 'centroid', 'ward']

# threshold for filtering out sensors in dfg relative to max occurrences of a sensor (value in range 0-1)
rel_proportion_dfg_threshold = 0.5

# miner used for process model creation - choose between: heuristic, inductive
miner_type = 'heuristic'
miner_type_list = ['heuristic', 'inductive']

# event case correlation export files
# folder containing files read and written during ecc trace partition method (number_of_activations)
dir_ecc_trace_partition_method_sensor_activations = 'ecc/' \
                                                    'trace_partition/' \
                                                    'method-{trace_partition_method}/' \
                                                    'number_of_activations-{number_of_activations}/'
# folder containing files read and written during ecc trace partition method (trace_duration)
dir_ecc_trace_partition_method_activation_time = 'ecc/' \
                                                 'trace_partition/' \
                                                 'method-{trace_partition_method}/' \
                                                 'trace_duration-{trace_duration}/'
# folder containing files read and written during ecc trace partition method (room separation)
dir_ecc_trace_partition_method_rooms_simple = 'ecc/' \
                                              'trace_partition/' \
                                              'method-{trace_partition_method}/'
# filename of ecc raw trace file
filename_ecc_traces_raw = 'traces_raw.pickle'

# filename of trace data file
filename_trace_data_time = 'trace_data_time.pickle'
# filename of traces cluster file
filename_output_case_traces_cluster = 'o_c_t_cluster.pickle'

# output files
# filename of trace file
filename_traces_raw = 'traces_raw.csv'
# csv delimiter of trace file
csv_delimiter_traces = ';'

# filename of divided trace file
filename_traces_basic = 'traces_basic.csv'
# csv delimiter of divided trace file
csv_delimiter_traces_basic = ';'

# filename of cluster file
filename_cluster = 'Cluster.csv'
# csv delimiter of cluster file
csv_delimiter_cluster = ';'

# filename of cases cluster file
filename_cases_cluster = 'Cases_Cluster.csv'
# csv delimiter of cases_cluster file
csv_delimiter_cases_cluster = ';'

# filename of log export file
filename_log_export = 'log_export.xes'

# folder containing petri net file
dir_petri_net_files = 'petri_nets/'
# filename of petri net .pnml file
filename_petri_net = 'petri_net.pnml'
# filename of petri net image file
filename_petri_net_image = 'ProcessModelHM.png'

# folder containing dfg png files
dir_dfg_files = 'directly_follows_graphs/'
# filename of dfg file (per cluster)
filename_dfg_cluster = 'DFG_Cluster_{cluster}.png'
# filename of dfg file
filename_dfg = 'DFG.png'

# # # # Hyperparameter Search # # # #
# Program execution type - Choose between the possible types:
# 'fixed_params' (the parameters are set before the program is executed),
# 'param_optimization' (uses a search space in which the parameters are optimized during execution)
execution_type = 'param_optimization'

# hyperopt parameter tuning - optimization algorithm (representative Tree of Parzen Estimators (TPE))
opt_algorithm = tpe.suggest

# how is the path of a person through the environment split
#    'FixedSensorActivations': only count how many sensors are being activated
#    'FixedActivationTime': Count the time between first and last activation in a trace
#    'RoomsSimple': Split the traces when the person leaves a certain area (rooms)
trace_partition_method = ['FixedSensorActivations', 'FixedActivationTime', 'RoomsSimple']

# range for number of sensor activations per trace
number_of_activations_per_trace_min = 5
number_of_activations_per_trace_max = 50
number_of_activations_per_trace_step_length = 5

# range for number of cumulated sensor duration time per trace
trace_duration_min = 120
trace_duration_max = 1800
trace_duration_step_length = 120

# set distance for zero to other sensors
# used in creation of the distance_matrix_real_world matrix
zero_distance_value_min = 1
zero_distance_value_max = 1

# number of clusters
hyp_min_number_clusters = 6
hyp_max_number_clusters = 20

# range for vectorization type
# possible types: 'quantity', 'time', 'quantity_time'
vectorization_type_list = ['quantity', 'time', 'quantity_time']

# # Routines
# separate the day into various segments
# 1: No segmentation
# 2: 0-12: Am, 12-23: PM
# 3: 0-8: Night, 8-16: Day, 16-23: Evening
# 4: 0-6: Night, 6-12: Morning, 12-18: Afternoon, 18-24: Night
# 5: 5-10: Morning, 10-14: Noon, 14-17: Afternoon, 17-23:Evening, 23-5:Night
hyp_number_of_day_partitions_list = [1, 2, 3, 4, 5]

# differentiate between different days of the week
# weekday: Mo, Tue, Wed, Thu, Fri, Sat, Sun
# workday: weekend/workday
# None: no day differentiation
# possible values: ['weekday', 'workday', None]
hyp_week_separator_list = ['weekday', 'workday', None]
# #

# range for clustering method (parameter optimization)
# possible methods: 'sklearn-SOM', 'k-Means', 'k-Medoids'
clustering_method_list = ['sklearn-SOM', 'k-Means', 'k-Medoids']

# # Use fixed values for debug # #
# fixed params execution parameters
fixed_params = {'zero_distance_value': 1,
                'traces_time_out_threshold': 300,
                'trace_length_limit': 6,
                'custom_distance_number_of_clusters': 10,
                'distance_threshold': 1.2,
                'max_errors_per_day': 100,
                'vectorization_type': 'quantity_time',
                'event_case_correlation_method': 'Classic',
                'clustering_method': 'sklearn-SOM',
                'hyp_number_of_day_partitions': 3,
                'hyp_week_separator': 'workday'}
# #
# # # # # # # # # # # # # # # # # # #
