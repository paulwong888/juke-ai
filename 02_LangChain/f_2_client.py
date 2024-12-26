import requests
import json

response = requests.post(
    url="http://localhost:9999/joke/invoke",
    # json={"input": {"topic": "小明"}}
    json={"topic": "小明"}
)
print(
    json.dumps(
        response.json(),
        indent=4,
        ensure_ascii=False
    )
)