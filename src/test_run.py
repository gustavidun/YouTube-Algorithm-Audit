import asyncio

from puppet import YTPuppet

if __name__ == "__main__":
    puppet = YTPuppet("test", 0, 1, False, drift_depth=20, train_depth=1, wt=15)
    asyncio.run(puppet.run())

