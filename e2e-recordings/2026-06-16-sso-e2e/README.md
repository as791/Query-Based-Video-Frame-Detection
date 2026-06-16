# VideoVault SSO E2E Recording

Created: 2026-06-16T17:03:54.909Z

Capture mode: Chrome tab screenshots encoded into MP4

Sample video: /Users/aryaman.sinha/Downloads/5977704-hd_1366_586_30fps.mp4

## Flow Covered
- Signed-out landing page
- Google SSO account chooser
- Authenticated Library page
- Video upload
- Async processing until Ready (3/3 chunks indexed)
- Frame search for low motion
- Clickable frame modal and previous-frame navigation
- Ask page WebSocket connection
- Fresh chat prompt and streamed assistant response

## Frames
- 001: 01 signed out landing — Frontend landing page before Google SSO
- 002: 02 google account chooser — Google account chooser for OAuth SSO
- 003: 03 authenticated library — Authenticated Library after Google OAuth redirect
- 004: 04 upload started — Sample video upload started
- 005: 05 processing 1 — normalized · 0/3 indexed · 0 failed
- 006: 05 processing 2 — indexing · 1/3 indexed · 0 failed
- 007: 05 processing 3 — Ready (3/3 chunks indexed)
- 008: 06 search query entered — Search query: low motion
- 009: 07 search results — High confidence frame search results returned
- 010: 08 frame modal open — Frame modal opened from search result
- 011: 09 frame modal previous — Previous frame navigation moved the modal timeline
- 012: 10 ask websocket connected — Ask page with WebSocket connected
- 013: 11 chat prompt entered — Chat prompt: summarize low motion
- 014: 12 chat streaming — Assistant response streaming over WebSocket
- 015: 13 chat completed — Assistant response completed and input reset

## Notes
- This is a visible-browser feature recording from Google SSO through upload, indexing, search, frame modal navigation, and WebSocket chat.
- Chrome clipboard fill is still unavailable in this environment, so typed inputs were entered with keypresses.
- The local VLM may fall back to retrieval-only answers when host model service is unavailable.
