# This documentation is an overall for:
1. How to reset and restart Docker containers
2. Data import process
3. API test commands
4. Steps to delete and reload models
5. How to fine-tune and load different models (transformer, hybrid)
6. Guide to checking model status and version
7. How to compare embedding models
8. Guide to running model evaluation script
9. How to use models to extract data

## Reset all docker containers
```
  bash
  cd chatbot/backend
  docker compose down -v
  docker system prune -a
```
##  Restart docker
1. docker compose build --no-cache  # Build takes about 750s 
2. docker compose up -d
3. docker compose ps #All containers should be in 'Up (healthy)' status
4. docker compose logs -f chatbot-api
5. docker compose logs -f chatbot-worker (another terminal tab)

-> At this point, the source has automatically downloaded models from hugging face according to the hf_timestamps and latest folder.

##  Test API
- You can run curl commands directly in the terminal
### Remove models folder
sudo rm -rf models
## 1. Test loading models with models/load 
Models that are not yet in the models folder need to be fine-tuned before loading, such as tfidf
```
curl -X POST http://localhost:8000/models/fine-tune \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 100000}'
```
After fine-tuning is complete, you can load the model
```
curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "tfidf", "version": "latest"}'
```

Load transformer models with `force_download: true` (only needed for the first run or when updating/fixing bugs):

    ```
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "roberta", "version": "latest", "force_download": true}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "xlm-roberta", "version": "latest", "force_download": true}'
    curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "distilbert", "version": "latest", "force_download": true}'
    ```

Load hybrid models:
    (Note: These commands will use the *latest* versions - based on the timestamp of component models that were loaded in the previous steps)
```
curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_tfidf_bert", "version": "latest"}'
curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_type": "hybrid_bm25_bert", "version": "latest"}'
```

## 2. Check model status:
```
curl -X GET http://localhost:8000/models/active
```
## 3. Get/check all versions of all models (including metadata information)
```
curl -X GET http://localhost:8000/models/versions \
  -H "Content-Type: application/json"
```
Delete a specific model version
```
curl -X DELETE http://localhost:8000/models/roberta/hf_20250416_163351
```

## 4. Compare embedding models
```
curl -X POST http://localhost:8000/models/compare \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 100000}'
```
```
curl -X GET http://localhost:8000/models/fine-tune/4474c114-032b-4fd0-a328-6e1dd90e8e65
```
### Use a specific model to extract data
```
curl -X POST http://localhost:8000/dbpedia/extract-data \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 100000, "embedding_model": "bm25"}'

curl -X GET http://localhost:8000/dbpedia/extract-data/c51a3967-f5d1-455a-9591-4f20dbb1d091 \
  -H "Content-Type: application/json"
```
### Use a specific model to extract data
```
curl -X POST http://localhost:8000/dbpedia/extract-data \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 1000000, "embedding_model": "bm25"}'
```
## 5. Run evaluation script:
Connect to the chatbot-api container
```
docker exec -it chatbot-api bash
```
Navigate to the src directory
```
cd /usr/src/app/src
```
Run the model evaluation script (will automatically download NLTK resources)
```
python evaluate_models.py --data_path data/history_learning_data.json --output_dir results --sample 10000 --limit 15
```

## 6. Use a specific model to extract data (after loading and indexing data)
```
curl -X POST http://localhost:8000/dbpedia/extract-data \
  -H "Content-Type: application/json" \
  -d '{"path": "data/history_learning_data.json", "sample": 100000, "embedding_model": "roberta"}'
```