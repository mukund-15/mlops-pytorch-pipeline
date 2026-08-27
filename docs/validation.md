# Validation Evidence

This document records the validation results obtained during development of the
MLOps PyTorch pipeline. No training workload needs to be rerun solely for
documentation.

## 1. Automated Tests

Command:

```bash
pytest -v

Result:

7 passed

The test suite covers:

ResNet-18 model creation
CIFAR-10 output shape
CPU forward pass
Invalid architecture handling
Trainable parameters
CIFAR-10 preprocessing
CIFAR-10 class names
2. Local Model Validation

The ResNet-18 model was validated on CPU.

Input shape : torch.Size([4, 3, 32, 32])
Output shape: torch.Size([4, 10])
Torch       : 2.13.0+cpu
CUDA        : False
3. Local Training

The local training pipeline completed successfully for 5 epochs.

Best result:

Best epoch           : 4
Best validation loss : 0.604884410572052
Best validation acc  : 0.7949

The training script emitted JSON-line metrics and saved:

checkpoints/classifier_v1.pt
4. Docker Training

Image:

mlops-train:v1

The training container was executed with:

docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1

Successful validation results:

Epoch 1 validation accuracy : 0.4856
Epoch 2 validation accuracy : 0.6821
Epoch 3 validation accuracy : 0.7803
Epoch 4 validation accuracy : 0.7949
Epoch 5 validation accuracy : 0.7946

Best validation loss       : 0.6049
Best validation accuracy   : 0.7949

The container saved:

/app/checkpoints/classifier_v1.pt

The checkpoint was persisted through the mounted volume and was available on
the host as:

checkpoints/classifier_v1.pt
5. Docker Serving

The serving image was successfully validated.

Health:

GET /health

{"status":"ok","model_loaded":true}

Prediction:

POST /predict

True class      : cat
Predicted class : cat
Confidence      : 0.966809

Container validation:

Container user : appuser
UID            : 1000
HEALTHCHECK    : healthy
Port           : 8080
6. Kubernetes Training

Namespace:

ml-training

PVC:

Name         : mlops-storage
Status       : Bound
Capacity     : 5Gi
StorageClass : hostpath

Training Job:

Name       : model-training
Completion : 1/1
Status     : Complete

The Job used:

Image           : mlops-train:v1
ConfigMap mount : /app/configs
Data mount      : /app/data
Checkpoint mount: /app/checkpoints
CPU request     : 2
Memory request  : 4Gi

The Kubernetes training Job successfully produced:

/app/checkpoints/classifier_v1.pt
7. Kubernetes Model Serving

Deployment:

Name           : model-serving
Ready replicas : 2/2
Available      : 2

Rolling update:

maxSurge       : 1
maxUnavailable : 0

Resources:

Requests:
  CPU    : 500m
  Memory : 1Gi

Limits:
  CPU    : 1
  Memory : 2Gi

Probes:

Liveness  : GET /health
Readiness : GET /health

Checkpoint mount:

/app/checkpoints
readOnly: true
8. Kubernetes Service

Service:

Name         : model-serving
Type         : ClusterIP
Service port : 80
Target port  : 8080

Two serving endpoints were registered through the EndpointSlice.

9. Kubernetes End-to-End API Validation

The Service was port-forwarded with:

kubectl port-forward svc/model-serving 8080:80 -n ml-training

Health:

{"status":"ok","model_loaded":true}

Prediction:

True class      : cat
Predicted class : cat
Confidence      : 0.966809

This validated the full path:

Kubernetes Service
        |
        v
Serving Pod
        |
        v
Read-only Checkpoint PVC
        |
        v
classifier_v1.pt
        |
        v
FastAPI /predict
10. GPU Bonus

The separate GPU bonus manifest is:

k8s/gpu-training-job.yaml

It requests:

nvidia.com/gpu: 1

and uses an NVIDIA GPU node selector.

It was validated using:

kubectl apply --dry-run=client -f k8s/gpu-training-job.yaml

The GPU Job was not executed because the development laptop has no NVIDIA GPU.

11. Instructor Clarifications

The implementation follows the instructor's clarifications:

pvc.yaml is included.
The GPU configuration is in a separate YAML file.
ci.yml is omitted.
hpa.yaml is omitted.
tests/test_model.py contains automated tests.
12. Evidence Notes

These results come from actual validation runs during development.

The expensive workloads do not need to be rerun solely to produce
documentation or evidence.