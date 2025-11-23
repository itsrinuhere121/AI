```Curl
curl --location 'http://localhost:12434/engines/llama.cpp/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "ai/qwen3:0.6B-Q4_0",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Tell me about the fall of Rome."
        }
    ]
}'
```

