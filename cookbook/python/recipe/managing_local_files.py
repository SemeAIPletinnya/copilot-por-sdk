#!/usr/bin/env python3

from copilot import CopilotClient
import os

POR_ABSTAIN_SIGNAL = "[[POR_ABSTAIN]]"

def handle_por_event(event):
    if event["type"] != "assistant.message":
        return False
    content = event["data"].get("content") or ""
    if POR_ABSTAIN_SIGNAL in content:
        print("🛑 Copilot abstained (PoR signal received).")
        return True
    return False

# Create and start client
client = CopilotClient()
client.start()

# Create session
session = client.create_session(model="gpt-5")

# Event handler
def handle_event(event):
    if handle_por_event(event):
        return
    if event["type"] == "assistant.message":
        print(f"\nCopilot: {event['data']['content']}")
    elif event["type"] == "tool.execution_start":
        print(f"  → Running: {event['data']['toolName']}")
    elif event["type"] == "tool.execution_complete":
        print(f"  ✓ Completed: {event['data']['toolCallId']}")

session.on(handle_event)

# Ask Copilot to organize files
# Change this to your target folder
target_folder = os.path.expanduser("~/Downloads")

session.send(prompt=f"""
Analyze the files in "{target_folder}" and organize them into subfolders.

1. First, list all files and their metadata
2. Preview grouping by file extension
3. Create appropriate subfolders (e.g., "images", "documents", "videos")
4. Move each file to its appropriate subfolder

Please confirm before moving any files.
""")

session.wait_for_idle()

session.destroy()
client.stop()
