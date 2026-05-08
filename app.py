"""
Rock Paper Scissors — AI Prediction Game
=========================================
Architecture:
  - Flask web server (routes + API)
  - GameEngine  : rules, scoring, session state
  - MarkovChain : sequence-based prediction
  - FrequencyModel : bias-based prediction
  - EnsembleAI  : combines both models + difficulty control
"""

from flask import Flask, render_template, request, jsonify, session
import random
import json
from collections import defaultdict

app = Flask(__name__)
app.secret_key = "rps_secret_key_2024"   # needed for session storage

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MOVES = ["rock", "paper", "scissors"]

# What beats what: BEATS[x] = the move that beats x
BEATS = {
    "rock":     "paper",
    "paper":    "scissors",
    "scissors": "rock",
}

# What loses to what: COUNTERED_BY[x] = what x loses to
# (same as BEATS, just named for clarity)
COUNTERED_BY = BEATS

# Emoji for display
EMOJI = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}


# ─────────────────────────────────────────────
# MODEL 1: MARKOV CHAIN PREDICTOR
# ─────────────────────────────────────────────

class MarkovChain:
    """
    Learns transition probabilities from move sequences.

    A Markov Chain asks: "Given the last N moves the player threw,
    what move are they most likely to throw next?"

    Example with order=2:
      History: [rock, paper, rock, paper, scissors, rock, paper]
      After seeing 'scissors → rock', what comes next?
      It looks up that bigram in its transition table.

    Data structure:
      transitions = {
        "scissors|rock": {"paper": 3, "rock": 1},
        "rock|paper":    {"scissors": 2},
        ...
      }
    """

    def __init__(self, order=2):
        self.order = order
        # One transition table per order level (e.g. order-1 and order-2)
        self.transitions = {
            n: defaultdict(lambda: defaultdict(int))
            for n in range(1, order + 1)
        }

    def _make_key(self, history_slice):
        """Joins a list of moves into a string key: ['rock','paper'] -> 'rock|paper'"""
        return "|".join(history_slice)

    def update(self, history):
        """
        Record the transition that just happened at every order level.
        Called with the full history AFTER the new move is appended.
        For each order n, looks at the last (n+1) moves:
          context[:-1] = the 'given' sequence (length n)
          context[-1]  = what the player just threw
        """
        for n in range(1, self.order + 1):
            if len(history) < n + 1:
                continue
            context   = history[-(n + 1):]
            key       = self._make_key(context[:-1])
            next_move = context[-1]
            self.transitions[n][key][next_move] += 1

    def predict(self, history):
        """
        Try highest order first; fall back to lower orders if no data exists.
        E.g. tries order-2 ('rock|paper'), then order-1 ('paper') before giving up.
        This prevents blind spots when a specific sequence hasn't been seen yet.
        """
        for n in range(self.order, 0, -1):
            if len(history) < n:
                continue
            key    = self._make_key(history[-n:])
            counts = self.transitions[n].get(key)
            if counts:
                return max(counts, key=counts.get)
        return None

    def confidence(self, history):
        """
        Returns confidence from whichever order level would be used to predict.
        """
        for n in range(self.order, 0, -1):
            if len(history) < n:
                continue
            key    = self._make_key(history[-n:])
            counts = self.transitions[n].get(key)
            if counts:
                total = sum(counts.values())
                best  = max(counts.values())
                return round(best / total, 3)
        return 0.0


# ─────────────────────────────────────────────
# MODEL 2: FREQUENCY / BIAS TRACKER
# ─────────────────────────────────────────────

class FrequencyModel:
    """
    Detects if the player has a favourite move.

    Simple idea: count how often each move appears in the full history.
    If 'rock' appears 60% of the time, predict rock and counter it.

    This catches players who subconsciously prefer one move.
    """

    def __init__(self):
        self.counts = {"rock": 0, "paper": 0, "scissors": 0}

    def update(self, move):
        self.counts[move] += 1

    def predict(self):
        """Returns the player's most frequent move (what we expect them to throw)."""
        total = sum(self.counts.values())
        if total < 3:
            return None  # not enough data
        return max(self.counts, key=self.counts.get)

    def confidence(self):
        """
        Returns confidence based on how dominant the top move is.
        If all three are equal (33% each), confidence is 0.
        If one is thrown 100% of the time, confidence is 1.
        """
        total = sum(self.counts.values())
        if total == 0:
            return 0.0
        best_count = max(self.counts.values())
        # Normalise: 0 = perfectly even, 1 = completely biased
        baseline = total / 3
        confidence = (best_count - baseline) / (total - baseline) if total > baseline else 0
        return round(max(0.0, confidence), 3)

    def distribution(self):
        """Returns percentage distribution of moves for the stats screen."""
        total = sum(self.counts.values())
        if total == 0:
            return {m: 0 for m in MOVES}
        return {m: round(self.counts[m] / total * 100, 1) for m in MOVES}


# ─────────────────────────────────────────────
# ENSEMBLE AI
# ─────────────────────────────────────────────

class EnsembleAI:
    """
    Combines MarkovChain + FrequencyModel with a weighted vote.

    Decision logic:
      1. Ask MarkovChain for its prediction + confidence
      2. Ask FrequencyModel for its prediction + confidence
      3. Weight-average their confidences
      4. Pick the model with higher weighted confidence
      5. Counter the predicted move (that's the AI's throw)
      6. On Easy mode, randomly ignore prediction 40% of the time

    This is the 'brain' of the AI.
    """

    def __init__(self, markov_weight=0.6, freq_weight=0.4):
        self.markov = MarkovChain(order=2)
        self.freq = FrequencyModel()
        self.markov_weight = markov_weight
        self.freq_weight = freq_weight

    def update(self, history):
        """Called after every round with the full history."""
        self.markov.update(history)
        if history:
            self.freq.update(history[-1])

    def predict(self, history, difficulty="hard"):
        """
        Returns a dict with:
          - ai_move: what the AI will throw
          - predicted_player_move: what the AI thinks the player will throw
          - confidence: 0–100% confidence
          - model_used: which model drove the decision
          - reasoning: human-readable explanation
        """
        # --- On Easy mode, occasionally be 'dumb' ---
        if difficulty == "easy" and random.random() < 0.40:
            return {
                "ai_move": random.choice(MOVES),
                "predicted_player_move": None,
                "confidence": 0,
                "model_used": "random",
                "reasoning": "Easy mode: AI made a random choice."
            }

        # --- Get predictions from both models ---
        markov_pred = self.markov.predict(history)
        markov_conf = self.markov.confidence(history) * self.markov_weight

        freq_pred = self.freq.predict()
        freq_conf = self.freq.confidence() * self.freq_weight

        # --- Choose the better model ---
        predicted_player_move = None
        model_used = "random"
        reasoning = "Not enough data yet — making a random choice."
        combined_confidence = 0

        if markov_pred and markov_conf >= freq_conf:
            predicted_player_move = markov_pred
            combined_confidence = markov_conf + freq_conf
            model_used = "markov"
            reasoning = f"Markov chain detected pattern: predicts you'll throw {markov_pred}."
        elif freq_pred:
            predicted_player_move = freq_pred
            combined_confidence = freq_conf + markov_conf
            model_used = "frequency"
            reasoning = f"Frequency model: you throw {freq_pred} most often."

        # --- Counter the predicted move ---
        if predicted_player_move:
            ai_move = COUNTERED_BY[predicted_player_move]
        else:
            ai_move = random.choice(MOVES)

        return {
            "ai_move": ai_move,
            "predicted_player_move": predicted_player_move,
            "confidence": round(min(combined_confidence * 100, 99)),  # cap at 99%
            "model_used": model_used,
            "reasoning": reasoning
        }

    def get_distribution(self):
        return self.freq.distribution()


# ─────────────────────────────────────────────
# GAME ENGINE
# ─────────────────────────────────────────────

class GameEngine:
    """
    Manages game state: score, history, outcome logic.

    State is stored in Flask's session (a server-side dict tied to the browser).
    On each request we load state → play a round → save state back.
    """

    def determine_outcome(self, player_move, ai_move):
        """
        Returns 'player', 'ai', or 'tie'.
        BEATS[x] = the move that beats x.
        So if BEATS[player_move] == ai_move, the AI's move beats the player's → AI wins.
        If BEATS[ai_move] == player_move, the player's move beats the AI's → player wins.
        """
        if player_move == ai_move:
            return "tie"
        if BEATS[player_move] == ai_move:
            return "ai"
        return "player"

    def build_result(self, player_move, ai_prediction):
        """
        Plays one round. Returns a complete result dict.
        """
        ai_move = ai_prediction["ai_move"]
        outcome = self.determine_outcome(player_move, ai_move)

        return {
            "player_move": player_move,
            "ai_move": ai_move,
            "outcome": outcome,
            "predicted_player_move": ai_prediction["predicted_player_move"],
            "confidence": ai_prediction["confidence"],
            "model_used": ai_prediction["model_used"],
            "reasoning": ai_prediction["reasoning"],
        }


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

game_engine = GameEngine()


def get_session_state():
    """Load or initialise session state."""
    if "rps" not in session:
        session["rps"] = {
            "history": [],       # player's move history
            "scores": {"player": 0, "ai": 0, "tie": 0},
            "rounds": [],        # full round log
            "difficulty": "hard",
            "ai_correct_predictions": 0,
        }
    return session["rps"]


def save_session_state(state):
    session["rps"] = state
    session.modified = True


@app.route("/")
def index():
    """Serve the main game page."""
    return render_template("index.html")


@app.route("/play", methods=["POST"])
def play():
    """
    Main game endpoint.
    Receives: { "move": "rock"|"paper"|"scissors", "difficulty": "easy"|"hard" }
    Returns:  full round result + updated scores + stats
    """
    data = request.get_json()
    player_move = data.get("move")
    difficulty = data.get("difficulty", "hard")

    if player_move not in MOVES:
        return jsonify({"error": "Invalid move"}), 400

    state = get_session_state()
    state["difficulty"] = difficulty

    # ── Rebuild AI from history (stateless server approach) ──
    # We replay the full history by feeding progressively longer slices.
    # Using enumerate gives the correct positional index — unlike .index(move)
    # which always finds the FIRST occurrence of that move value, producing
    # wrong slices when the same move appears multiple times (e.g. rock,paper,rock).
    ai = EnsembleAI()
    for i, move in enumerate(state["history"]):
        ai.markov.update(state["history"][:i + 1])
        ai.freq.update(move)

    # ── AI makes prediction BEFORE seeing player's move ──
    prediction = ai.predict(state["history"], difficulty)

    # ── Resolve the round ──
    result = game_engine.build_result(player_move, prediction)

    # ── Update state ──
    state["history"].append(player_move)
    state["scores"][result["outcome"]] += 1

    if (prediction["predicted_player_move"] == player_move and
            prediction["model_used"] != "random"):
        state["ai_correct_predictions"] += 1

    state["rounds"].append({
        "round": len(state["rounds"]) + 1,
        "player": player_move,
        "ai": result["ai_move"],
        "outcome": result["outcome"],
    })

    # ── Compute statistics ──
    total_rounds = sum(state["scores"].values())
    predictable_rounds = sum(
        1 for r in state["rounds"] if r["outcome"] != "tie"
    )
    ai_accuracy = (
        round(state["ai_correct_predictions"] / max(total_rounds, 1) * 100, 1)
    )

    # Rebuild distribution from fresh AI
    ai_fresh = EnsembleAI()
    for move in state["history"]:
        ai_fresh.freq.update(move)

    save_session_state(state)

    return jsonify({
        "result": result,
        "scores": state["scores"],
        "total_rounds": total_rounds,
        "ai_accuracy": ai_accuracy,
        "distribution": ai_fresh.get_distribution(),
        "history_length": len(state["history"]),
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Returns full session statistics for the end-screen."""
    state = get_session_state()
    total = sum(state["scores"].values())

    ai_fresh = EnsembleAI()
    for move in state["history"]:
        ai_fresh.freq.update(move)

    return jsonify({
        "scores": state["scores"],
        "total_rounds": total,
        "rounds_log": state["rounds"][-10:],  # last 10 rounds
        "distribution": ai_fresh.get_distribution(),
        "ai_accuracy": round(
            state["ai_correct_predictions"] / max(total, 1) * 100, 1
        ),
        "difficulty": state["difficulty"],
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Clears session state and starts a fresh game."""
    session.pop("rps", None)
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    # debug=True auto-reloads when you save the file — great for development
    app.run(debug=True)
