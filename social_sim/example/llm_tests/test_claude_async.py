import asyncio
import random
import time
from anthropic import AsyncAnthropic, RateLimitError

client = AsyncAnthropic()

NUM_PROMPTS = 1000
prompt_array = [f"Prompt {i}" for i in range(NUM_PROMPTS)]

MAX_CONCURRENCY = 40
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

MAX_TOKENS = 1000
MODEL = "claude-3-haiku-20240307"

async def safe_call(prompt, agent_id, max_retries=3):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text if response.content else "<empty>"
                return f"[Agent {agent_id}] {text}"
            except RateLimitError:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"⚠️ Rate limit hit for Agent {agent_id}, retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
        return f"[Agent {agent_id}] ❌ FAILED after retries"

async def main():
    start_time = time.time()

    tasks = [
        safe_call(prompt, i)
        for i, prompt in enumerate(prompt_array)
    ]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    print(f"\n✅ Finished {len(prompt_array)} agents in {total_time:.2f} seconds.\n")

    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
