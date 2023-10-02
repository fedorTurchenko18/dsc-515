# Federated Learning Concept

## Classic Machine Learning Approach
- The training data we work with doesn’t originate on the machine we train the model on – it gets created somewhere else
- "Somewhere else" is usually not just one place, it’s many places. It could be several devices all running the same app. But it could also be several organizations, all generating data for the same task
- So to use machine learning, or any kind of data analysis, the approach that has been used in the past was to collect all data on a central server. This server can be somewhere in a data center, or somewhere in the cloud
- Once all the data is collected in one place, we can finally use machine learning algorithms to train our model on the data. This approach could be described as *"bring data to the computation"*

## Challenges of Classic ML
1. Tasks variety and resulting complications
   - Classic ML solves tasks with centralized data really well (e.g. web traffic analysis)
   - Cases, where the data is not available on a centralized server, cause issues, since it complicates "bringing data to the computation" through obligatory complex data pipelines execution. Examples of these include:
       - Sensitive healthcare records from multiple hospitals to train cancer detection models
       - Financial information from different organizations to detect financial fraud
       - Location data from your electric car to make better range prediction
       - End-to-end encrypted messages to train better auto-complete models
2. Legal regulations, causing even inability to fetch data from certain instances
3. User preference, associated with compliance to the full data privacy, which again does not allow for smooth data extraction
4. Data volume: Some sensors, like cameras, produce such a high data volume that it is neither feasible nor economic to collect all the data (due to, for example, bandwidth or communication efficiency). These data sometimes are not stored in a geographical location, thus may cause issues of access

## Federated Learning Solution
Reverse the Classic ML pipeline: *"bring computation to the data"*

### Step 0: Initialize the model
- Initialize the global model on the global server
<img src="https://flower.dev/docs/framework/_images/fl-initialize-global-model.png" alt="Step 0" width="400"/>

### Step 1: Send the parameters of the global model to the connected client nodes
- This is to ensure that each participating node starts their local training using the same model parameters
<img src="https://flower.dev/docs/framework/_images/fl-send-global-model.png" alt="Step 1" width="400"/>

### Step 2: Train model locally on the data of each organization/device (client node)
- They use their own local dataset to train their own local model
- They don’t train the model until full convergence, but they only train for a little while
- This could be as little as one epoch on the local data, or even just a few steps (mini-batches)
<img src="https://flower.dev/docs/framework/_images/fl-send-model-updates.png" alt="Step 2" width="400"/>

### Step 3: Return model updates back to the server
- After local training, each client node has a slightly different version of the model parameters they originally received
- The parameters are all different because each client node has different examples in its local dataset
- The client nodes then send those model updates back to the server
- The model updates they send can either be the full model parameters or just the gradients that were accumulated during local training
<img src="https://flower.dev/docs/framework/_images/fl-send-model-updates.png" alt="Step 3" width="400"/>
