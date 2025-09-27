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
    description : Optional[str] = None
    tags : Optional[list[str]] = None
    category : Optional[str] = None
    blacklist : Optional[bool] = False

    def __str__(self):
        return f"Video ID: {self.id}, video slant: {self.slant}"

@dataclass
class Watch():
    state : str
    puppet : "YTPuppet"
    puppet_slant : float
    depth : int
    video : Video
    recs : list[Video]

    def __str__(self):
        return f"{self.video}. Depth: {self.depth}, puppet slant: {self.puppet_slant}, puppet state: {self.state}"