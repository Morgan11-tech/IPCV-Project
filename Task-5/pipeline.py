"""
Task 5 - perception: everything the camera frame tells us, per frame.

    Task 1 faces -> Task 2 hands + signs -> Task 3 hand-to-player
    -> adapters.player_inputs() -> one Task 4 PlayerInput per player

stabilize.py sits on top of Task 1 and Task 3 to reduce jitter and identity
or hand-owner flips (smoothing + hysteresis).

No drawing and no game rules here.
"""
from dataclasses import dataclass

import bootstrap
import config
import registration
import tracking
from recognition import build_reference_embedding, detect_faces, recognize_faces
from src.recognizer import SignRecognizer
from task3_player_association import PlayerAssociator, PoseBodyShapes, faces_from_rush_players
from task4_game import LEVEL_WORDS

import adapters
from stabilize import FaceTracks, HandOwners


@dataclass
class Perception:
    players: dict       # Task 1 {pid: PlayerState}
    hands: list         # Task 3 hand list (pixels), same order as Task 2's hands
    owners: dict        # Task 3 {hand index: pid}
    inputs: dict        # Task 4 {pid: PlayerInput}


class PerceptionPipeline:
    def __init__(self, detector, recognizer, references, use_pose=True, fake_p2=False):
        self.detector, self.recognizer, self.references = detector, recognizer, references
        self.fake_p2 = fake_p2
        self.players = tracking.create_players()
        self.faces = FaceTracks(self.players)
        self.hand_owners = HandOwners()
        self.signs = SignRecognizer(mirror_input=config.MIRROR_VIEW)   # frames arrive mirrored
        # the model knows more words than the levels use; only game words may be answered
        adapters.restrict_vocabulary(self.signs, [w for words in LEVEL_WORDS.values() for w in words])
        self.pose = PoseBodyShapes(model_path=bootstrap.POSE_MODEL) if use_pose else None
        self.assoc = PlayerAssociator()

    def step(self, frame, frame_id, t_s, timer):
        """Run Tasks 1-3 on one frame; timer.lap() records each stage."""
        H, W = frame.shape[:2]

        faces = detect_faces(self.detector, frame)
        assignments, similarities = {}, None
        if faces and frame_id % config.RECOGNITION_INTERVAL == 0:
            assignments, similarities = recognize_faces(self.recognizer, frame, faces, self.references)
        self.faces.update(faces, assignments, similarities, frame_id, t_s)
        timer.lap("T1 faces")

        t2_out = self.signs.update(frame, frame_id, timestamp=t_s)
        timer.lap("T2 hands")

        hands = adapters.hands_for_task3(t2_out, W, H)
        bodies = self.pose.detect(frame) if self.pose else None
        self.assoc.process(frame_id, faces_from_rush_players(self.players), hands, bodies=bodies)
        owners = self.hand_owners.update(self.assoc.current_owners, self.signs.signs)
        self.assoc.current_owners = dict(owners)        # debug view shows the sticky owners
        timer.lap("T3 assoc")

        inputs = adapters.player_inputs(self.players, owners, t2_out, self.signs.signs, W, H)
        if self.fake_p2:
            inputs[2] = adapters.fake_player_input(W, H)
        return Perception(self.players, hands, owners, inputs)


def get_references(cap, window, detector, recognizer, register, fake_p2):
    """Task 1 face references {1: ref, 2: ref}, registering first if needed.
    Returns None if registration was quit."""
    pids = (1,) if fake_p2 else (1, 2)
    embeddings = {} if register else {pid: registration.load_player_data(pid) for pid in pids}
    if register or any(e is None for e in embeddings.values()):
        embeddings = {}
        for pid in pids:
            other = build_reference_embedding(embeddings[1]) if pid == 2 else None
            result = registration.register_player(cap, window, detector, recognizer, pid,
                                                  other_reference=other)
            if result is None:
                return None
            embeddings[pid] = result[0]
            registration.save_player_data(pid, *result)

    references = {pid: build_reference_embedding(e) for pid, e in embeddings.items()}
    if fake_p2:
        # Nobody's face can match the opposite of Player 1's reference, so the
        # real Player 2 slot stays empty and the fake player is used instead.
        references[2] = -references[1]
    return references
