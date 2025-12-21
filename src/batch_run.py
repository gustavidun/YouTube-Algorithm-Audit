import asyncio
import argparse

from puppet import YTPuppet
import shortuuid
import random

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=45)
parser.add_argument("--train", type=int, default=100)
parser.add_argument("--drift", type=int, default=200)
parser.add_argument("--headless", type=bool, default=True)
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

N = args.n
TRAIN = args.train
DRIFT = args.drift
HEADLESS = args.headless
RANDOM = args.random

if not RANDOM: SLANTS = [(-1, 0), (-1, 1), (0, 1), (0, -1), (1, 0), (1,-1), (-1,-1), (0,0), (1,1), (-99,-99)] #-99 = random 
else: SLANTS = [(-99,-99)]

crashes = 0

async def safe_run(puppet : YTPuppet):
    global crashes
    await asyncio.sleep(random.uniform(0,30))
    try:
        await puppet.run()
    except Exception as e:
        puppet.logger.exception("Puppet crashed", exc_info=e)
        crashes += 1

async def main():

    if N % len(SLANTS) != 0:
        raise Exception("SLANTS length must be divisible with N")

    k = N // len(SLANTS)
    partitions = SLANTS * k

    puppets = []
    for i, s in enumerate(partitions):
        if s[0] == -99:
            num = random.uniform(-1,1)
            s = (num, num)

        if RANDOM:
            util = True if (i+1)%2 == 0 else False

        puppets.append(
            YTPuppet(
                f"p-{shortuuid.uuid()}",
                slant=s[0],
                target_slant=s[1],
                train_depth=TRAIN,
                drift_depth=DRIFT,
                headless=HEADLESS,
                utility=util
            )
        )

    async with asyncio.TaskGroup() as tg:
        for p in puppets:
            tg.create_task(safe_run(p))

    return puppets

if __name__ == "__main__":
    puppets = asyncio.run(main())
    print(f"Finished run. Lost {crashes} puppets")

