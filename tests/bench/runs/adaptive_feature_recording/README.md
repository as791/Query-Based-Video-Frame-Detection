# Adaptive Learning Feature Recording

Primary artifact: `adaptive_learning_real_ui_actions.webm`

Previous slide-style artifact: `adaptive_learning_feature_walkthrough.webm`

The primary recording is an action-only UI walkthrough. It shows:

1. Features page domain selection.
2. Existing domain labels and a real uploaded fall example clip.
3. Playback of the actual MP4 clip used for the few-shot example.
4. Library domain selection.
5. Domain-scoped search for `person falling down`.
6. Real search result cards with extracted frame evidence.
7. `Relevant` and `Not relevant` feedback clicks.
8. Opening an extracted frame modal.

The v1 "trained model" is not LoRA or a neural checkpoint. It is a lightweight domain profile with labels, example references, prototype centroids, feedback centroids, ranker config, and model version history.

## Runtime Notes

- Chrome captured the live app states for Features and Library during real UI actions.
- `SPHAR_p01faint_fall_1.mp4` is included as the actual source clip used in the walkthrough.
- The final video was produced in a local Chrome tab using the browser `MediaRecorder` API over a canvas replay of captured app states and the actual clip playback.
- The current UI user has metadata for the demo domain and labels; benchmark-backed domains remain separate under the benchmark auth user.
