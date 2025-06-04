import anthropic
import time
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

start_time = time.time()

client = anthropic.Anthropic()

# 1. Submit batch
model = "claude-3-haiku-20240307"
BATCH_SIZE = 1000
requests = []
for i in range(BATCH_SIZE):
    new_request = Request(
        custom_id=f"my-first-request-{i}",
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello, world"}]
        )
    )
    requests.append(new_request)
message_batch = client.messages.batches.create(
    requests=requests
)

batch_id = message_batch.id
print(f"Submitted batch {batch_id}")

# 2. Poll until the batch is complete
while True:
    message_batch = client.messages.batches.retrieve(batch_id)
    if message_batch.processing_status == "ended":
        print("Batch processing complete.")
        break
    print(f"Batch {batch_id} is still processing...")
    time.sleep(1)

# 3. Retrieve results
results = client.messages.batches.results(batch_id)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")


# 3. Retrieve results
# 3. Retrieve results (again)
# Stream results file in memory-efficient chunks, processing one at a time
for result in results:
    print(result)
