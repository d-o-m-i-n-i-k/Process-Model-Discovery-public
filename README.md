# Process Model Discovery from Sensor Event Data

Discovers process models from raw event log data. 

Based on the approach presented in: <br>
D. Janssen, A. Koschmider, F. Mannhardt, S. v. Zelst. <br>
Process Model Discovery from Sensor Event Data. 


## Installation
### Clone
Clone this repository to your local machine using https://github.com/d-o-m-i-n-i-k/Process-Model-Discovery-public.git
### Requirements
- Python 3.5+
- Dataset with time stamps, sensor IDs and sensor status 
- Adjacency matrix of the sensors' location
- Define parameters:
  * zero_distance_value_list (set pseudo distance for no sensors activated to other sensors)
  * max_number_of_raw_input (upper limit for input_data (set to None if there is no limit))
  * max_number_of_people_in_house (maximum number of persons which were in the house while the recording of sensor data)
  * traces_time_out_threshold_list (the time in seconds in which a sensor activation is assigned to a existing trace)
  * max_trace_length_list (maximum length of traces (in case length mode is used to separate raw-traces))
  * data_types_list (choose between the similarity measure: quantity, time, quantity_time)
  * k_means_number_of_clusters (number of k-means cluster)

## Usage

### Step 0: z_create_distance_matrix
- Distance matrix is created from the adjacency matrix using the Floyd-Warshall-Algorithm

<br>
<br>
The following steps are performed in several iterations with different parameters. The parameters are determined by hyperparameter optimization using "hyperopt".
 
### Step 1: a_EventCaseCorrelation
Transform raw-data to traces
- Read in the sensor data
- Create trace from the raw data by associating every event log entry to an entity (def convert_raw_data_to_traces)
- Split up the raw traces to shorter traces (def divide_raw_traces)

### Step 2: b_ActivityDiscovery
Activity Discovery
- To establish a baseline, cluster traces with a plain k-means algorithm
- Apply a self-organising-map algorithm on the traces (Code by: Vahid Moosavi and Sebastian Packmann)

### Step 3: c_EventActivityAbstraction
Event Activity Abstraction
- Match the discovered clusters by both the vanilla k-means and the som-algorithm to the raw data input

### Step 4: d_ProcessDiscovery
Process Discovery
- Create a directly follows graph of the process using the visualization tool "graphviz"
- Determine the accuracy of a process model using "PM4Py"
