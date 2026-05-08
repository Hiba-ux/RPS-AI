# Rock Paper Scissors — AI Prediction Game

A Python/Flask web game where an AI learns your move patterns and tries to beat you.

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py

# 4. Open your browser
http://127.0.0.1:5000
```

## Project Structure

```
rps_game/
├── app.py              # All Python logic: Flask server + AI models + Game Engine
├── templates/
│   └── index.html      # Animated single-page game UI (HTML + CSS + JS)
├── requirements.txt    # Python packages needed
└── README.md
```

## System Architecture

### Layer 1 — UI (index.html)
The frontend is a single HTML page with:
- CSS animations for move reveals, win/loss effects, glows
- JavaScript that calls the Flask API and updates the UI
- No frameworks needed — pure HTML/CSS/JS

### Layer 2 — Game Engine (app.py: GameEngine class)
- Stores game state in Flask's `session` (a browser-tied server dict)
- Determines round outcomes using the BEATS dictionary
- Tracks scores, move history, round log

### Layer 3 — AI Brain (app.py: MarkovChain, FrequencyModel, EnsembleAI)

**MarkovChain**: Learns move sequences.
  - Order 2: looks at your last 2 moves to predict your next one
  - Builds a transition table: {"rock|paper": {"scissors": 3, "rock": 1}}

**FrequencyModel**: Detects which move you throw most often.
  - Simple count of rock/paper/scissors in full history

**EnsembleAI**: Combines both with weighted voting (60% Markov, 40% Frequency).
  - On Easy mode: ignores prediction 40% of the time (random)
  - On Hard mode: always uses the best available prediction

## API Endpoints

| Route    | Method | Purpose                          |
|----------|--------|----------------------------------|
| /        | GET    | Serve the HTML game page         |
| /play    | POST   | Play a round, get result + stats |
| /stats   | GET    | Get full session statistics      |
| /reset   | POST   | Clear all session data           |

## 

- **Markov Chain**: A probabilistic model where the next state depends only on the current state (memoryless property). Order-N Markov chains look at N previous states.
- **Ensemble model**: Combining multiple weak models to produce a stronger prediction.
- **Flask session**: Server-side key/value store tied to a browser cookie — used here as in-memory game state.
- **REST API**: The frontend and backend communicate via JSON over HTTP — a standard web architecture pattern.
