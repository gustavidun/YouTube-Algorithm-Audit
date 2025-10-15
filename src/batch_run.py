import asyncio
import argparse

from puppet import YTPuppet
import shortuuid

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=36)
parser.add_argument("--train", type=int, default=100)
parser.add_argument("--drift", type=int, default=200)
parser.add_argument("--headless", type=bool, default=True)
args = parser.parse_args()

N = args.n
TRAIN = args.train
DRIFT = args.drift
HEADLESS = args.headless

SLANTS = [(-1, 0), (-1, 1), (0, 1), (0, -1), (1, 0), (1,-1), (-1,-1), (0,0), (1,1)] 

async def safe_run(puppet : YTPuppet):
    try:
        await puppet.run()
    except Exception as e:
        puppet.logger.exception("Puppet crashed", exc_info=e)

async def main():

    if N % len(SLANTS) != 0:
        raise Exception("SLANTS length must be divisible with N")

    k = N // len(SLANTS)
    partitions = SLANTS * k

    puppets = [
        YTPuppet(
            f"p-{shortuuid.uuid()}",
            slant=s[0],
            target_slant=s[1],
            train_depth=TRAIN,
            drift_depth=DRIFT,
            headless=HEADLESS
        ) 
        for i, s in enumerate(partitions)
    ]

    async with asyncio.TaskGroup() as tg:
        for p in puppets:
            tg.create_task(safe_run(p))

    return puppets

if __name__ == "__main__":
    puppets = asyncio.run(main())

