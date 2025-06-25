# BeatSaberBot

Beat Saber Bot is a Python script that generates replay (.bsor) files for Beat Saber, a VR rhythm game. It tries to mimic a human (it's not good at it, very good). Bot receives map data from BeatSaver, processes it and creates a replay file that can be used in the game.

How the bot works

Map retrieval: The bot loads the map from BeatSaver by its ID, retrieves info.dat and difficulty files and allows the user to select the difficulty.
Simulation: The bot analyzes the notes, walls, and bombs of the map, calculating their timing based on the BPM and jump speed of the note. The bot simulates saber and head movements, adjusting for walls and bombs, and makes human-like errors (e.g., misses or failed hits).
Replay Generation: The bot creates a .bsor file containing:
Information: Metadata such as song title, player information, and score.
Frames: Positions/rotations of the head and saber at 90 frames per second.
Notes: Events such as successful/unsuccessful hits, misses or bombs, with score details.

Key Features

Error Mode: Optionally introduces misses, failed hits or bomb hits to simulate real player gameplay (This works very poorly).
Repeat Saving: Saves repeats in a structured folder (BeatSaberDemos/<song_name>/<difficulty>.bsor) for easy access and compatibility with Beat Saber.
