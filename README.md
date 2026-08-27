# MLOps PyTorch Pipeline

An end-to-end MLOps pipeline for CIFAR-10 image classification using PyTorch,
Docker, and Kubernetes.

## 1. Project Overview

This project demonstrates the complete deployment lifecycle of a PyTorch
image classification model:

1. Model development and training with PyTorch
2. Automated testing with pytest
3. Containerized model training with Docker
4. Containerized model serving with FastAPI
5. Kubernetes-based model training
6. Persistent storage using Kubernetes PersistentVolumeClaim
7. Kubernetes model serving with two replicas
8. Health and prediction endpoint validation

## 2. Architecture

```mermaid
flowchart TD
    A[CIFAR-10 Dataset] --> B[Training PVC<br>/app/data]

    B --> C[Kubernetes Training Job]
    C --> D[mlops-train:v1]

    D --> E[Best Model Checkpoint]
    E --> F[Checkpoint PVC<br>/app/checkpoints]

    F --> G[Serving Pod 1<br>mlops-serve:v1]
    F --> H[Serving Pod 2<br>mlops-serve:v1]

    G --> I[ClusterIP Service<br>port 80]
    H --> I

    I --> J[GET /health]
    I --> K[POST /predict]

## 3. Repository Structure
   CIFAR-10
      |
      v
   PersistentVolumeClaim (/app/data)
      |
      v
   Kubernetes Training Job
      |
      v
   mlops-train:v1
      |
      v
   classifier_v1.pt
      |
      v
   PersistentVolumeClaim (/app/checkpoints)
      |
      +--------------------------+
      |                          |
      v                          v
   Serving Pod 1             Serving Pod 2
   mlops-serve:v1            mlops-serve:v1
      |                          |
      +------------+-------------+
                  |
                  v
         ClusterIP Service :80
                  |
            +-----+------+
            |            |
            v            v
         /health      /predict

## 4. Environment

The pipeline was developed and validated using:

Windows 11
WSL2
Ubuntu 26.04 LTS
Python 3.11.16
Docker Desktop
Docker Desktop Kubernetes
Kubernetes v1.36.1
8 CPU cores
11.68 GiB Docker memory
CPU-only development machine
No NVIDIA GPU

## 5. Model
Dataset: CIFAR-10
Architecture: ResNet-18
Number of classes: 10
Input size: 3 × 32 × 32
Optimizer: Adam
Loss function: CrossEntropyLoss
Training device: CPU
Early stopping: supported
Best model checkpoint: classifier_v1.pt

The ResNet-18 model is adapted for CIFAR-10 using a 3×3 first convolution
and no initial max-pooling layer.

## 6. Python Setup

The project uses Python 3.11.16 through pyenv.

pyenv local 3.11.16
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt

Run tests:

pytest -v
## 7. Local Training
python src/train.py

The training configuration is read from:

configs/training_config.yaml

Training metrics are printed as JSON lines containing:

epoch
training loss
training accuracy
validation loss
validation accuracy

The best checkpoint is saved as:

checkpoints/classifier_v1.pt
## 8. Tests

The test suite validates:

ResNet-18 model creation
CIFAR-10 output shape
CPU forward pass
Invalid architecture handling
Trainable parameters
CIFAR-10 preprocessing
CIFAR-10 class names

Validated result:

7 passed
## 9. Docker Training

Build the training image:

docker build -f docker/Dockerfile.train -t mlops-train:v1 .

Run training:

docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1

The container uses:

MLOPS_DATA_DIR=/app/data
MLOPS_CHECKPOINT_DIR=/app/checkpoints

The CPU-only PyTorch wheel is used because the development system does not
have an NVIDIA GPU.

## 10. Docker Serving

Build:

docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

Run:

docker run --rm \
  -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  mlops-serve:v1

Health:

curl http://localhost:8080/health

Prediction:

curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"

The serving container runs as the non-root appuser user and includes a
Docker HEALTHCHECK.

## 11. Kubernetes Setup

Create the namespace:

kubectl apply -f k8s/namespace.yaml

Create the ConfigMap:

kubectl apply -f k8s/configmap.yaml

Create the PVC:

kubectl apply -f k8s/pvc.yaml

The local Docker Desktop Kubernetes cluster uses the hostpath StorageClass.

## 12. Kubernetes Training

Run:

kubectl apply -f k8s/training-job.yaml

Monitor the Job:

kubectl logs -f job/model-training -n ml-training

Verify:

kubectl get jobs -n ml-training
kubectl get pods -n ml-training
kubectl get pvc -n ml-training

The training Job uses:

Image: mlops-train:v1

/app/configs
/app/data
/app/checkpoints

Resource requests and limits:

CPU:    2
Memory: 4Gi
## 13. Kubernetes Model Serving

Deploy:

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml

Verify:

kubectl get deployment model-serving -n ml-training
kubectl get pods -n ml-training
kubectl get svc model-serving -n ml-training

The Deployment uses:

2 replicas
RollingUpdate strategy
maxSurge: 1
maxUnavailable: 0
liveness probe on /health
readiness probe on /health
read-only checkpoint PVC
CPU request: 500m
memory request: 1Gi
CPU limit: 1
memory limit: 2Gi
## 14. Kubernetes End-to-End Validation

Port-forward the Service:

kubectl port-forward svc/model-serving 8080:80 -n ml-training

Health:

curl http://localhost:8080/health

Expected:

{"status":"ok","model_loaded":true}

Prediction:

curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"

Validated result:

True class:      cat
Predicted class: cat
Confidence:      0.966809
## 15. GPU Bonus

The GPU bonus is provided separately in:

k8s/gpu-training-job.yaml

The manifest requests:

nvidia.com/gpu: 1

and uses an NVIDIA GPU node selector.

The local development machine has no NVIDIA GPU, so the GPU Job was not
executed locally. The manifest was validated using:

kubectl apply --dry-run=client -f k8s/gpu-training-job.yaml
## 16. Assignment Clarifications

The implementation follows the instructor's clarifications:

pvc.yaml is included.
GPU bonus uses a separate gpu-training-job.yaml.
.github/workflows/ci.yml is omitted.
k8s/hpa.yaml is omitted.
tests/test_model.py contains actual tests.
Docker and Kubernetes workflows use CPU-only PyTorch because the development
laptop has no NVIDIA GPU.
## 17. Validation Summary
Component	Result
PyTorch model	PASS
Automated tests	7 passed
Local CPU training	PASS
Docker training	PASS
Docker checkpoint persistence	PASS
Docker /health	PASS
Docker /predict	PASS
Docker non-root user	PASS
Docker HEALTHCHECK	PASS
Kubernetes namespace	PASS
Kubernetes ConfigMap	PASS
Kubernetes PVC	Bound
Kubernetes training Job	Complete 1/1
Kubernetes serving Deployment	2/2
Kubernetes Service	ClusterIP :80
Kubernetes /health	PASS
Kubernetes /predict	PASS
GPU bonus manifest	Dry-run PASS
## 18. Reflection

The main challenge in this project was maintaining a reproducible MLOps workflow across local Python, Docker, and Kubernetes on a CPU-only development machine. The laptop did not have an NVIDIA GPU, so the project used CPU-only PyTorch wheels and kept the GPU implementation as a separate Kubernetes bonus manifest.

A second challenge was dependency management. The initial Docker build attempted to install CUDA-related packages even though the system had no NVIDIA GPU. This resulted in a very large dependency download and a failed installation. The issue was resolved by explicitly using the CPU PyTorch package index and pinning the CPU-specific PyTorch and torchvision versions.

Kubernetes introduced another practical challenge when the training Job initially failed while attempting to download CIFAR-10 from inside the cluster. The already verified CIFAR-10 dataset was therefore placed into the Kubernetes persistent volume before training. This removed the runtime dependency on downloading the dataset and allowed the Kubernetes training Job to complete successfully.

The project was developed incrementally: model testing was completed before dataset loading, local training was validated before Docker training, Docker serving was tested before Kubernetes serving, and the final Kubernetes API was validated only after the deployment and Service were healthy. This approach made failures easier to isolate and provided clear evidence for each deployment stage.

The final system demonstrates a complete pipeline from CIFAR-10 data and PyTorch training to persistent model storage and two-replica Kubernetes inference. The model successfully served predictions through the Kubernetes Service, including a correct cat prediction with high confidence.
Another useful lesson was the value of persistent storage and health-based deployment management. The training Job and serving Deployment have different responsibilities, while the PVC provides a durable boundary for both training data and model artifacts. The readiness and liveness probes also make the serving layer observable and allow Kubernetes to determine whether a replica is ready to receive requests. Using separate Git branches and pull requests for each major stage made the development history easier to review and helped keep the implementation organized.
