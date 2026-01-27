# Session Persistence and Resumption

Save and restore conversation sessions across application restarts.

## Example scenario

You want users to be able to continue a conversation even after closing and reopening your application.

> **Runnable example:** [recipe/persisting_sessions.py](recipe/persisting_sessions.py)
>
> ```bash
> cd recipe && pip install -r requirements.txt
> python persisting_sessions.py
> ```

### Creating a session with a custom ID

```python
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

def handle_message(event):
    if handle_por_event(event):
        return
    if event["type"] == "assistant.message":
        print(f"Copilot: {event['data']['content']}")

client = CopilotClient()
client.start()

# Create session with a memorable ID
session = client.create_session(
    session_id="user-123-conversation",
    model="gpt-5",
)
session.on(handle_message)

session.send(prompt="Let's discuss TypeScript generics")

# Session ID is preserved
print(session.session_id)  # "user-123-conversation"

# Destroy session but keep data on disk
session.destroy()
client.stop()
```

### Resuming a session

```python
client = CopilotClient()
client.start()

# Resume the previous session
session = client.resume_session("user-123-conversation")
session.on(handle_message)

# Previous context is restored
session.send(prompt="What were we discussing?")

session.destroy()
client.stop()
```

### Listing available sessions

```python
sessions = client.list_sessions()
for s in sessions:
    print("Session:", s["sessionId"])
```

### Deleting a session permanently

```python
# Remove session and all its data from disk
client.delete_session("user-123-conversation")
```

### Getting session history

```python
messages = session.get_messages()
for msg in messages:
    print(f"[{msg['type']}] {msg['data']}")
```

## Best practices

1. **Use meaningful session IDs**: Include user ID or context in the session ID
2. **Handle missing sessions**: Check if a session exists before resuming
3. **Clean up old sessions**: Periodically delete sessions that are no longer needed
