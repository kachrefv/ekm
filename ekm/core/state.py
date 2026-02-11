import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class FocusItem(BaseModel):
    aku_id: str
    timestamp: float
    relevance_score: float = 1.0
    current_weight: float = 1.0
    frequency: int = 1

class FocusBuffer(BaseModel):
    items: Dict[str, FocusItem] = Field(default_factory=dict)
    max_size: int = 20
    decay_rate: float = 0.05

    def update(self, activated_ids: List[str], action: str = 'activate'):
        now = time.time()
        boost = 1.5 if action == 'activate' else 1.0
        
        # Decay existing
        for item in self.items.values():
            if item.aku_id not in activated_ids:
                item.current_weight *= (1.0 - self.decay_rate)
        
        # Add/Update new
        for _id in activated_ids:
            if _id in self.items:
                self.items[_id].frequency += 1
                self.items[_id].timestamp = now
                self.items[_id].relevance_score += (0.2 * boost)
                self.items[_id].current_weight = self.items[_id].relevance_score
            else:
                self.items[_id] = FocusItem(
                    aku_id=_id,
                    timestamp=now,
                    relevance_score=1.0 * boost,
                    current_weight=1.0 * boost
                )
        
        # Prune
        if len(self.items) > self.max_size:
            sorted_items = sorted(
                self.items.values(), 
                key=lambda x: 0.7 * x.current_weight + 0.3 * (x.timestamp / now),
                reverse=True
            )
            self.items = {item.aku_id: item for item in sorted_items[:self.max_size]}
