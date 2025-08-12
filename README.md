# ⚾ MLB Matchup Analysis Tool

## Overview
The **MLB Matchup Analysis Tool** is a data-driven baseball analytics platform that helps users evaluate game matchups, player performance, and hitting probabilities with precision.  
It integrates **confirmed starting lineups**, **stadium-specific park factors**, **real-time weather conditions**, and **player-vs-pitcher splits** to produce actionable insights for analysts, bettors, and fans.

---

## Key Features
- **Confirmed Starting Lineups**  
  Only displays players who are confirmed to start (batting order 1–9). Excludes bench, IL, or TBD players.

- **Weather-Driven Hit Odds**  
  Generates per-player probabilities for:
  - Single (1B)
  - Double (2B)
  - Triple (3B)
  - Home Run (HR)  
  Adjusted for temperature, wind speed & direction, humidity, barometric pressure, precipitation chance, and roof status.

- **Park Factors**  
  Accounts for stadium-specific tendencies for extra-base hits and home runs (handedness-aware if available).

- **Player Performance Baselines**  
  Incorporates rolling averages and projection systems for hitter vs. pitcher handedness.

- **Game Metadata**  
  Includes:
  - Game start time (stadium local time)
  - Stadium name
  - Home/Away status
  - Doubleheader game number (if applicable)

---

## Data Sources
The tool integrates data from:
- **Official MLB Schedule & Lineup Feeds**
- **Weather APIs** for game-time forecasts
- **Park Factor Databases**
- **Player Performance Statistics** (season-to-date, rolling splits, projections)

---

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/mlb-matchup-analysis.git
cd mlb-matchup-analysis
