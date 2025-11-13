import asyncio
import httpx
import time

# this is for testing and speeding up async gemini api calls, reuse in bulk_truck_cleaning api

async def call_gemini(i):
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "http://localhost:8000/api/clean_truck",  # or directly Gemini
            json={"raw_truck_name": f"{i} MT"}
        )
        print(i, res.status_code)
        return res.json()

async def main():
    start = time.time()
    results = await asyncio.gather(*[call_gemini(i) for i in range(10)])
    print("Total time:", time.time() - start)
    print(results, "results")   


asyncio.run(main())
