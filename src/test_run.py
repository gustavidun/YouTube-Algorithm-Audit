import asyncio

from puppet import YTPuppet

if __name__ == "__main__":
    puppet = YTPuppet("test", 0, 1, drift_depth=3, train_depth=1, wt=10, headless=False)
    asyncio.run(puppet.run())

