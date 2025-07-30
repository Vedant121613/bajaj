import aiohttp
import aiofiles
import os

async def download_pdf(url: str, save_path: str = "temp.pdf") -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download PDF: {response.status}")
            
            async with aiofiles.open(save_path, "wb") as f:
                await f.write(await response.read())

    return save_path
