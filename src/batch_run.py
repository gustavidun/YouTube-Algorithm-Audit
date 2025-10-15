import asyncio
import argparse

from puppet import YTPuppet
import shortuuid

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=36)
parser.add_argument("--train", type=int, default=100)
parser.add_argument("--drift", type=int, default=200)
args = parser.parse_args()

N = args.n
TRAIN = args.train
DRIFT = args.drift

SLANTS = [(-1, 0), (-1, 1), (0, 1), (0, -1), (1, 0), (1,-1), (-1,-1), (0,0), (1,1)] 

async def main():

    if N % len(SLANTS) != 0:
        raise Exception("SLANTS length must be divisible with N")

    k = N // len(SLANTS)
    partitions = SLANTS * k

    puppets = [YTPuppet(f"p-{shortuuid.uuid()}", slant=s[0], target_slant=s[1],train_depth=TRAIN, drift_depth=DRIFT) for i, s in enumerate(partitions)]

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(puppet.run()) for puppet in puppets]

    return puppets

if __name__ == "__main__":
    puppets = asyncio.run(main())

