from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from puppet import YTPuppet 

@dataclass
class Video():
    id : str
    slant : Optional[float] = None
    title : Optional[str] = None
    channel : Optional[str] = None

    def __str__(self):
        return f"ID: {self.id}, slant: {self.slant}"

@dataclass
class Watch():
    state : str
    puppet : YTPuppet
    puppet_slant : float
    depth : int
    video : Video
    recs : list[Video]

    def __str__(self):
        return f"{self.video}. Depth: {self.depth}, puppet state: {self.state}"