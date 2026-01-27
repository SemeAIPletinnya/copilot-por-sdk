#!/usr/bin/env python3

from copilot import CopilotClient

POR_ABSTAIN_SIGNAL = "[[POR_ABSTAIN]]"

def handle_por_event(event):
    if event["type"] != "assistant.message":
        return False
    content = event["data"].get("content") or ""
    if POR_ABSTAIN_SIGNAL in content:
        print("🛑 Copilot abstained (PoR signal received).")
        return True
    return False

client = CopilotClient()

try:
    client.start()
    session = client.create_session(model="gpt-5")

    response = None
    def handle_message(event):
        nonlocal response
        if handle_por_event(event):
            return
        if event["type"] == "assistant.message":
            response = event["data"]["content"]

    session.on(handle_message)
    session.send(prompt="Hello!")
    session.wait_for_idle()

    if response:
        print(response)

    session.destroy()
except Exception as e:
    print(f"Error: {e}")
finally:
    client.stop()
