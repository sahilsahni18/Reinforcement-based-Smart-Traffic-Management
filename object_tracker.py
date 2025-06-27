from sort import Sort # type: ignore

class VehicleTracker:
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections):
        # detections: list of [x1, y1, x2, y2, score]
        tracks = self.tracker.update(detections)
        # returns list of [x1, y1, x2, y2, track_id]
        return tracks