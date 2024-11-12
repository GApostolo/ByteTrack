import numpy as np
from collections import OrderedDict

# STATE MACHINE FOR TRACKS - The state changes depending on whether the track matches or not 
# New - Track is new and not yet associated with any detection
# Tracked - Track is currently matched
# Lost - Track was matched but not matched in the current frame (but will be able to match in the next frames)
# Removed - Track was lost for too long
class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

# Base class for all tracks
class BaseTrack(object):
    _count = 0 ############

    # TRACK PROPERTIES
    track_id = 0
    is_activated = False # Track is activated when it can be matched in the next frame (i.e. not lost/removed) 
    state = TrackState.New

    # Track features from current detection 
    history = OrderedDict() # Previously matched detections
    features = []           # Re-ID of previously matched features
    curr_feature = None     # Current feature
    score = 0               # Current confidence score 

    # TRACK FRAME PROPERTIES 
    start_frame = 0         # Frame where the track started
    frame_id = 0            # Current frame
    time_since_update = 0   # Number of frames since the track was last matched 
                            # (Starts to increment when track becomes lost, if exceeds some limit, the track goes from lost to removed)

    # multi-camera
    location = (np.inf, np.inf) ###########

    ###########
    @property
    def end_frame(self):
        return self.frame_id

    ###########
    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    # Change track state
    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
