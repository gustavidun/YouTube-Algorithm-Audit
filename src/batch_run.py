import asyncio

from puppet import YTPuppet

N = 10
SLANTS = [-1, -0.5, 0, 0.5, 1]

async def main():

    if N % len(SLANTS) != 0:
        raise Exception("SLANTS length must be divisible with N")

    k = N // len(SLANTS)
    partitions = SLANTS * k

    puppets = [YTPuppet(f"puppet-", slant=s) for i, s in enumerate(partitions)]

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(puppet.run()) for puppet in puppets]

    return puppets

if __name__ == "__main__":
    puppets = asyncio.run(main())

