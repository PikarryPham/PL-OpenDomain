# Model Quantization API Documentation

## Introduction

This document describes the new API functionality for managing the quantization of transformer models in the system. Quantization is the process of reducing the precision of weights in a model to decrease model size and increase inference speed while maintaining an acceptable level of prediction quality.

## New API: `/models/quantize`

### Description
This API allows users to actively initiate the quantization process for a specific transformer model, helping to reduce model size and increase inference speed.

### Method
POST

### Parameters
Requires a JSON body with the structure:
```json
{
  "model_type": "<model_name>"
}
```

Where `<model_name>` can be one of the following values:
- `roberta` - RoBERTa model
- `xlm-roberta` - XLM-RoBERTa model
- `distilbert` - DistilBERT model

### Responses

#### Success - Model quantized
```json
{
  "status": "success",
  "message": "Successfully quantized <model_name> model",
  "is_quantized": true
}
```

#### Success - Model was already quantized
```json
{
  "status": "success",
  "message": "Model <model_name> is already quantized",
  "already_quantized": true
}
```

#### Error - Model does not exist or is not supported
```json
{
  "status": "error",
  "message": "Model <model_name> not found or not supported for quantization"
}
```

#### Error - Error during quantization
```json
{
  "status": "error",
  "message": "Error during quantization: <error_details>"
}
```

## How to check quantization status

### Using the `/models/active` API

Send a GET request to the `/models/active` API and check the `is_quantized` field in the model metadata:

```json
{
  "roberta": {
    "status": "active",
    "metadata": {
      "model_name": "roberta-base",
      "is_quantized": true
    }
  },
  "xlm-roberta": {
    "status": "active",
    "metadata": {
      "model_name": "xlm-roberta-base",
      "is_quantized": false
    }
  }
}
```

## Technical Changes

### Quantization Process Improvements

1. **Separate quantization process**: Quantization is now handled through a dedicated API instead of being performed automatically during model loading.

2. **Race condition handling**: Modifications help avoid race conditions during container startup when multiple models try to quantize simultaneously.

3. **Better error management**: The system logs more details about the quantization process and related errors.

4. **Status check before quantization**: The system checks if a model has already been quantized before initiating the quantization process, avoiding redundant operations.

### Quantization Implementation

The system uses Dynamic Post-Training Quantization (Dynamic PTQ) through the `torch.quantization` library:

```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

This method only quantizes the Linear layers in the model with the data type `torch.qint8` (8-bit integer), which helps:
- Reduce model size by approximately 4 times
- Increase inference speed
- Maintain prediction quality at a reasonable level

### Metadata Storage

When a model is quantized, the system updates the `is_quantized` field in the model's metadata.json file:

```json
{
  "model_name": "roberta-base",
  "model_type": "roberta",
  "version": "hf_20250417_084022",
  "created_at": "2025-04-17T08:40:22.516901",
  "is_quantized": true,
  "source": "HuggingFace (downloaded)",
  "dimensions": 768
}
```

## Using the quantize API

### Example: Quantizing a RoBERTa model

```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "roberta"}'
```

### Example: Quantizing an XLM-RoBERTa model

```bash
curl -X POST http://localhost:8000/models/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_type": "xlm-roberta"}'
```

## Benefits of Quantization

1. **Memory savings**: Quantized models have smaller sizes, reducing RAM requirements when running.
2. **Increased processing speed**: Computations with integers are faster than with floating-point numbers, leading to faster inference speed.
3. **Energy efficiency**: Consumes less energy, suitable for devices with limited power sources.
4. **Equivalent performance**: In many cases, quantized models still maintain performance nearly equivalent to the original model.