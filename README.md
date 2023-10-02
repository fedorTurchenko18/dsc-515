# Amazon Web Catalogue Image Search Engine

## Project Goal

Content-Based Image Retrieval (CBIR): Implement a system that allows users to search for products using images as queries. Users can upload an image, and the system finds similar products in your catalog. When dealing with CBIR task, convolutional neural networks (CNNs) are commonly used due to their effectiveness in image feature extraction.

Additionally, the study will employ Flower framework for Federated Learning implementation to explore how different Deep Learning frameworks, namely JAX/Pytorch/TensorFlow, influence the performance of a single model architecture.

## Data Source

[Amazon Berkeley Objects (ABO)](https://registry.opendata.aws/amazon-berkeley-objects/) is a collection of 147,702 product listings with multilingual metadata and 398,212 unique catalog images. 8,222 listings come with turntable photography (also referred as "spin" or "360º-View" images), as sequences of 24 or 72 images, for a total of 586,584 images in 8,209 unique sequences. For 7,953 products, the collection also provides high-quality 3d models, as glTF 2.0 files.

This project requires only the usage of downscaled (max 256 pixels) catalog images and metadata (3 Gb).

## Project Board

The [following](https://miro.com/app/board/uXjVMh9LxnY=/?share_link_id=535088326052) Miro board provides a graphical description of the project.

## Federated Learning Concept

### Classic Machine Learning Approach
- The training data we work with doesn’t originate on the machine we train the model on – it gets created somewhere else
- "Somewhere else" is usually not just one place, it’s many places. It could be several devices all running the same app. But it could also be several organizations, all generating data for the same task
- So to use machine learning, or any kind of data analysis, the approach that has been used in the past was to collect all data on a central server. This server can be somewhere in a data center, or somewhere in the cloud
- Once all the data is collected in one place, we can finally use machine learning algorithms to train our model on the data. This approach could be described as *"bring **data** to the computation"*

### Challenges of Classic ML
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

### Federated Learning Solution
Reverse the Classic ML pipeline: *"bring **computation** to the data"*

#### Step 0: Initialize the model
- Initialize the global model on the global server
<img src="https://flower.dev/docs/framework/_images/fl-initialize-global-model.png" alt="Step 0" width="400"/>

#### Step 1: Send the parameters of the global model to the connected client nodes
- This is to ensure that each participating node starts their local training using the same model parameters
<img src="https://flower.dev/docs/framework/_images/fl-send-global-model.png" alt="Step 1" width="400"/>

#### Step 2: Train model locally on the data of each organization/device (client node)
- They use their own local dataset to train their own local model
- They don’t train the model until full convergence, but they only train for a little while
- This could be as little as one epoch on the local data, or even just a few steps (mini-batches)
<img src="https://flower.dev/docs/framework/_images/fl-send-model-updates.png" alt="Step 2" width="400"/>

#### Step 3: Return model updates back to the global server
- The parameters are all different because each client node has different examples in its local dataset
- The client nodes then send those model updates back to the server
- The model updates they send can either be the full model parameters or just the gradients that were accumulated during local training
<img src="https://flower.dev/docs/framework/_images/fl-send-model-updates.png" alt="Step 3" width="400"/>

#### Step 4: Aggregate model updates into a new global model
- If the server selected 100 client nodes, it now has 100 slightly different versions of the original global model, each trained on the local data of one client
- *Federated Averaging* algorithm is applied to compile a single model:
     - It takes the weighted average of the model updates, weighted by the number of examples each client used for training
     - The weighting is important to make sure that each data example has the same “influence” on the resulting global model
     - If one client has 10 examples, and another client has 100 examples, then - without weighting - each of the 10 examples would influence the global model ten times as much as each of the 100 examples
<img src="https://flower.dev/docs/framework/_images/fl-aggregate-model-updates.png" alt="Step 4" width="400"/>

#### Step 5: Repeat steps 1 to 4 until the model converges
- The loop is wrapped up as follows:
     - The global model parameters get sent to the participating client nodes (step 1)
     - The client nodes train on their local data (step 2)
     - They send their updated models to the server (step 3)
     - The server then aggregates the model updates to get a new version of the global model (step 4)
- After the aggregation step (step 4), we have a model that has been trained on all the data of all participating client nodes, but only for a little while
- We then have to repeat this training process over and over again to eventually arrive at a fully trained model that performs well across the data of all client nodes

## Flower Frameweork
Flower provides the infrastructure to do exactly that in an easy, scalable, and secure way. In short, Flower presents a unified approach to federated learning, analytics, and evaluation. It allows the user to federate any workload, any ML framework, and any programming language.

### "Hello World" Example
This sequence of commands allows to run the simulation of federated learning with 2 clients (you will need 3 terminals running simultaneously in total)

Terminal 1:

Install the dependencies first

```
# could also be pip3 | pip3.x depending on your local configuration
# make sure you are in the root directory of the project
pip install -r requirements.txt
```

Then go to the example directory

`cd flower-helloworld`

Start the server

```
# could also be python3 | python3.x depending on your local configuration
# IMPORTANT: you need to run this first, before starting the clients
python server.py
```

Terminal 2:

Go to the example directory

`cd flower-helloworld`

Start the client

```
# could also be python3 | python3.x depending on your local configuration
python client.py
```

Terminal 3:

Repeat the same actions as for Terminal 2
