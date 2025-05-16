# This documentation is only used for:
1. Troubleshooting common errors
2. Recommended error-free loading and fine-tuning processes

-> Only run it whenever you meets the errors as follows

## Log in to the container
``` docker exec -it chatbot-api bash ``` 

## Load models
``` python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('roberta-base'); model = AutoModel.from_pretrained('roberta-base'); print('Successfully loaded roberta-base')"

python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base'); model = AutoModel.from_pretrained('xlm-roberta-base'); print('Successfully loaded xlm-roberta-base')"

python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased'); model = AutoModel.from_pretrained('distilbert-base-uncased'); print('Successfully loaded distilbert-base-uncased')"

``` 

# Troubleshooting errors during model loading and fine-tuning

## Common errors and solutions

### 1. Connection errors to Hugging Face

If you encounter errors like:
```
"We couldn't connect to 'https://huggingface.co' to load the files..."
```
or other timeout errors when loading transformer models.

**Solution:**
- Ensure the server has a stable internet connection to Hugging Face.
- Use the `force_download: true` parameter when calling the `POST /models/load` API to try downloading again. This command will attempt to download from the source and save a new version.
```
bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest", "force_download": true}'
```

### 2. Error "'torch.dtype' object has no attribute 'data_ptr'" (Already handled by fix_transformer_models.py)

This error typically occurs with older PyTorch versions or incorrect environment configurations when loading RoBERTa or XLM-RoBERTa.

**Solution:**
- Run the `fix_transformer_models.py` script before loading or fine-tuning.
- **Important:** Use the load API with `force_download: true` **after running the fix script**. The `load` API with `force_download` will reload the model from Hugging Face, ensuring you have the most compatible version.
```bash
# Step 1: Run the fix script (if not already run)
# python src/fix_transformer_models.py --force

# Step 2: Reload the model with force_download
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "roberta", "version": "latest", "force_download": true}'
```

### 3. Error "does not appear to have a file named pytorch_model.bin..."

Usually occurs when the model has not been fully downloaded or there was an error during the download/local save process.

**Solution:**
- Use the load API with `force_download: true` to redownload the entire model from Hugging Face.
```bash
curl -X POST http://localhost:8000/models/load \
   -H "Content-Type: application/json" \
   -d '{"model_type": "xlm-roberta", "version": "latest", "force_download": true}'
```

### 4. Error "BMX model is not available"

**Solution:**
- BMX is an optional dependency. If you don't need to use the `bmx` or `hybrid_bmx_bert` models, you can ignore this error. The system will still work with other models.
- If you *do* need to use BMX, make sure you've installed the `baguetter` library correctly according to their instructions. Installation may require C++ compilation steps.

### 5. Error "'bool' object is not iterable" when calling /models/load

This error occurs because the API handler function returns a boolean value instead of the expected JSON format.

**Solution:**
- **Already fixed** in the latest code version by modifying the `load_transformer_model` function in `embeddings.py` to always return a tuple `(success, result)`. Make sure you're running the updated code.
- If you still encounter the error, check the changes in `embeddings.py` and ensure the Docker container has been rebuilt with the latest code.

## Error-free loading and fine-tuning process (Recommended)

1.  **(Optional but recommended)** Run the transformer fix script first:
    ```bash
    python src/fix_transformer_models.py --force
    ```
2.  Load basic models (TF-IDF, BM25):
    ```bash
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "tfidf", "version": "latest"}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "bm25", "version": "latest"}'
    ```
3.  Load transformer models with `force_download: true` (only needed for first run or when updating/fixing):
    ```bash
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "roberta", "version": "latest", "force_download": false}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "xlm-roberta", "version": "latest", "force_download": false}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "distilbert", "version": "latest", "force_download": false}'
    ```
4.  Load hybrid models:
    (Note: These commands will use the *latest* versions of the component models loaded in the previous steps)
    ```bash
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_tfidf_bert", "version": "latest"}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_bm25_bert", "version": "latest"}'
    # curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_bmx_bert", "version": "latest"}' # If BMX is installed
    ```
5.  Proceed with fine-tuning (will use the latest model versions available):
    ```bash
    curl -X POST http://localhost:8000/models/fine-tune \
      -H "Content-Type: application/json" \
      -d '{"path": "data/history_learning_data.json", "sample": 10000, "version": "custom_v1"}'
    ```
6.  Check model status:
    ```bash
    curl -X GET http://localhost:8000/models/active
    ```

# Simplest approach: Use automatic scripts

## Method 1: Load all models (Using download_all_models.sh)
This script will automatically call the `/models/load` API for all models, including using `force_download: true` for transformer models to ensure they are loaded correctly.

```bash
# Give execute permission to the script
chmod +x src/download_all_models.sh

# Run the script (optional parameters: [host] [port])
./src/download_all_models.sh localhost 8000
```

## Method 2: Fix specific transformer model issues (Using fix_transformer_models.py)
This script helps create the necessary directory structure and files to fix some issues when loading RoBERTa and XLM-RoBERTa. **You should run this script before loading transformer models for the first time.**

```bash
# Give execute permission to the script
chmod +x src/fix_transformer_models.py

# Run the script to fix issues (create dummy files and directory structure)
python src/fix_transformer_models.py --force

# Check if models have been "fixed" (check directory structure)
python src/fix_transformer_models.py --check
```
**Note:** After running this script, you still need to call the `/models/load` API with `force_download: true` to actually download the model data from Hugging Face.

## Method 3: Automate the entire process (Using fix_and_finetune.sh - Best method)
This script combines running `fix_transformer_models.py`, `download_all_models.sh` (loading all models with `force_download`), and then automatically calls the fine-tune API.

```bash
# Give execute permission to the script
chmod +x src/fix_and_finetune.sh

# Run the script (optional parameters: [host] [port] [finetune_version])
./src/fix_and_finetune.sh localhost 8000 my_finetuned_v1
```

This script will:
1.  Run `fix_transformer_models.py --force`.
2.  Run `download_all_models.sh` to load all models (with `force_download` for transformers).
3.  Call the `/models/fine-tune` API with the specified version.
4.  Monitor and display the results of the fine-tuning task.
5.  Check the final status of the models via the `/models/active` API.

This is the simplest and most comprehensive way to ensure models are properly set up and fine-tuned.