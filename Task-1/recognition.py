"""
Face detection, face embeddings, cosine similarity and identity decision.

Pipeline for one frame:
    frame --YuNet--> face boxes + 5 landmarks
          --align--> 112x112 face crops
          --SFace--> 128-D embedding per face
          --cosine similarity vs. Player 1 / Player 2 references-->
          --one-to-one assignment + threshold--> PLAYER_1 / PLAYER_2 / UNKNOWN
"""
import cv2
import numpy as np

import config


def load_models():
    """Load the pretrained face detector (YuNet) and recognizer (SFace)."""
    detector = cv2.FaceDetectorYN.create(
        config.DETECTOR_MODEL, "", (320, 320),
        config.DETECTION_SCORE_THRESHOLD, config.DETECTION_NMS_THRESHOLD)
    recognizer = cv2.FaceRecognizerSF.create(config.RECOGNIZER_MODEL, "")
    return detector, recognizer


def detect_faces(detector, frame):
    """
    Detect all faces in a BGR frame.

    YuNet is a small CNN that predicts, for every face, a bounding box, five
    landmarks (right eye, left eye, nose tip, right/left mouth corner, from
    the subject's point of view) and a confidence score.

    Returns a list of dicts:
        box       (x, y, w, h) in full-frame pixels
        landmarks 5x2 array of (x, y)
        score     detector confidence 0..1
        row       the raw 15-value YuNet row, needed later for alignment
    """
    h, w = frame.shape[:2]
    # Detect on a smaller image for speed, then scale coordinates back.
    scale = min(1.0, config.DETECTION_WIDTH / w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, rows = detector.detect(small)

    faces = []
    if rows is None:
        return faces
    for row in rows:
        row = row.copy()
        row[:14] /= scale            # boxes and landmarks back to full size
        faces.append({
            "box": tuple(float(v) for v in row[0:4]),
            "landmarks": row[4:14].reshape(5, 2),
            "score": float(row[14]),
            "row": row,
        })
    return faces


def get_face_embedding(recognizer, frame, face):
    """
    Turn one detected face into a 128-D identity embedding.

    1. alignCrop uses the eye/nose/mouth landmarks to rotate and scale the face
       into a canonical 112x112 pose (same geometry the network was trained on),
       so head roll and distance do not change the input much.
    2. The SFace CNN maps the crop to a 128-D vector. Faces of the same person
       end up close together in this space, different people far apart.
    3. We L2-normalise the vector (length 1) so that cosine similarity becomes
       a simple dot product.
    """
    aligned = recognizer.alignCrop(frame, face["row"])
    embedding = recognizer.feature(aligned).flatten().astype(np.float32)
    return embedding / np.linalg.norm(embedding)


def cosine_similarity(a, b):
    """
    cos(theta) = (a . b) / (|a| |b|)

    1.0  -> same direction (very similar faces)
    0.0  -> unrelated
    <0   -> opposite
    Only the direction of an embedding carries identity information, not its
    length, which is why cosine similarity is used for SFace/ArcFace-style models.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_reference_embedding(embeddings):
    """
    The "training" / preparation step: average one player's registration
    embeddings into a single reference vector.

    Each sample contains the identity PLUS noise from pose, expression and
    lighting. Averaging keeps the shared identity direction and cancels part of
    the sample-specific variation. We re-normalise because the mean of unit
    vectors is shorter than 1.

    No neural network is trained here: SFace stays frozen; we only store a
    template in its embedding space.
    """
    mean = np.mean(np.asarray(embeddings), axis=0)
    return mean / np.linalg.norm(mean)


def assign_faces_to_players(similarities, threshold=config.SIMILARITY_THRESHOLD):
    """
    One-to-one assignment of faces to the two players.

    similarities: array of shape (num_faces, 2); column 0 = Player 1,
                  column 1 = Player 2.

    Rules:
      * a player can be given to at most one face,
      * a face can be given to at most one player,
      * a pair is only allowed if its similarity >= threshold,
      * among all allowed combinations pick the one with the highest total
        similarity.

    With only two players we can simply try every combination: Player 1 gets
    face i (or nobody), Player 2 gets face j != i (or nobody). For n faces that
    is (n+1)^2 options, which is tiny. Faces left over are UNKNOWN.

    Returns {player_id: (face_index, similarity)} for assigned players only.
    """
    num_faces = similarities.shape[0]
    options = [None] + list(range(num_faces))   # None = player not assigned

    best_total, best = 0.0, {}
    for i in options:
        for j in options:
            if i is not None and i == j:
                continue                          # one face cannot be both
            total, choice = 0.0, {}
            if i is not None:
                if similarities[i, 0] < threshold:
                    continue
                total += similarities[i, 0]
                choice[1] = (i, float(similarities[i, 0]))
            if j is not None:
                if similarities[j, 1] < threshold:
                    continue
                total += similarities[j, 1]
                choice[2] = (j, float(similarities[j, 1]))
            if total > best_total:
                best_total, best = total, choice
    return best


def recognize_faces(recognizer, frame, faces, references):
    """
    Compute embeddings for all detected faces, compare them with both player
    references and decide who is who.

    references: {1: reference_embedding, 2: reference_embedding}

    Returns (assignments, similarities):
        assignments  {player_id: (face_index, similarity)}
        similarities (num_faces, 2) array, useful for debug display
    """
    similarities = np.zeros((len(faces), 2), dtype=np.float32)
    for f, face in enumerate(faces):
        embedding = get_face_embedding(recognizer, frame, face)
        similarities[f, 0] = cosine_similarity(embedding, references[1])
        similarities[f, 1] = cosine_similarity(embedding, references[2])
    return assign_faces_to_players(similarities), similarities
